import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import os
from hippo_model import HippoModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import create_causal_mask, create_sliding_window_causal_mask

class LoRALinear(nn.Module):
    """
    自定义LoRA线性层，支持增量式LoRA微调
    """
    def __init__(self, in_features, out_features, r=8, lora_alpha=32, lora_dropout=0.1, bias=False, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = r
        
        # 原始权重
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None
            
        # LoRA参数 - 维度符合LoRA标准：A将输入映射到低维，B将低维映射到输出
        self.lora_A = nn.Parameter(torch.empty((r, in_features)))  # (rank, in_features)：将in_features映射到rank
        self.lora_B = nn.Parameter(torch.empty((out_features, r)))  # (out_features, rank)：将rank映射到out_features
        self.scaling = lora_alpha / r  # 缩放因子
        
        # Dropout（用于LoRA分支的输入）
        self.lora_dropout = nn.Dropout(lora_dropout)
        
        # 初始化参数
        self.reset_parameters()
        
    def reset_parameters(self):
        """初始化参数 - 使用LoRA论文推荐的初始化策略"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        # LoRA初始化：A用高斯分布，B用零初始化（保证初始时LoRA分支不干扰基础模型）
        nn.init.kaiming_normal_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        """
        前向传播：output = 基础模型输出 + LoRA分支输出
        修正点：
        1. 修复LoRA分支的线性层维度逻辑（去掉错误的转置）
        2. 增加dropout在LoRA分支的应用
        """
        # 基础模型输出：W * x + bias
        result = F.linear(x, self.weight, self.bias)
        
        # LoRA分支输出：(B * A * x_dropout) * scaling
        if self.rank > 0:
            # 步骤1：对输入x应用dropout（稳定训练）
            x_dropout = self.lora_dropout(x)
            
            # 步骤2：低秩映射（A将输入压缩到低维）
            # lora_A维度：(rank, in_features)，F.linear会自动计算 x_dropout @ lora_A.T
            # 输出维度：(batch, seq_len, rank)
            lora_mid = F.linear(x_dropout, self.lora_A)
            
            # 步骤3：低秩映射（B将低维还原到输出维度）
            # lora_B维度：(out_features, rank)，F.linear会自动计算 lora_mid @ lora_B.T
            # 输出维度：(batch, seq_len, out_features)（与基础模型输出维度一致）
            lora_output = F.linear(lora_mid, self.lora_B)
            
            # 步骤4：添加LoRA分支结果（乘缩放因子）
            result += self.scaling * lora_output
            
        return result
    
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}'


class HippoLoRAQwen(nn.Module):
    """
    基于LoRA微调的Qwen模型，集成Hippo状态空间建模
    特点：在fusion_layers中使用LoRA进行参数高效微调，其他层保持冻结
    """
    def __init__(self, base_model_name_or_path, fusion_layers=None, seq_len=4096, lora_rank=8, lora_alpha=32, lora_dropout=0.1, cache_dir="qwen3-8b-custom-module-training", last_n_tokens=0, attn_lora_config=None, ffn_lora_config=None):
        super().__init__()
        
        self.base_model_name_or_path = base_model_name_or_path
        self.seq_len = seq_len
        # 存储LoRA配置参数
        self.attn_lora_config = attn_lora_config
        self.ffn_lora_config = ffn_lora_config
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name_or_path,
            trust_remote_code=True
        )
        # 加载基础模型
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True  # 减少CPU内存使用
        )

        self.config = self.base_model.config

        self.hidden_size = self.config.hidden_size
        
        self.num_transformer_layers = self.config.num_hidden_layers

        self.fusion_layers = fusion_layers if fusion_layers is not None else [0]
        
        # 配置LoRA参数 - 支持独立配置注意力层和FFN层
        self.lora_config = self._setup_lora_config(
            lora_rank, lora_alpha, lora_dropout, attn_lora_config, ffn_lora_config
        )
        
        print("LoRA配置:")
        print(f"- 默认配置: rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}")
        if attn_lora_config:
            print(f"- 注意力层配置: {attn_lora_config}")
        if ffn_lora_config:
            print(f"- FFN层配置: {ffn_lora_config}")
        
        # 默认融合层（选择几个关键层）
        if fusion_layers is None:
            # 选择中后层的几个关键位置进行微调
            num_layers = self.base_model.config.num_hidden_layers
            fusion_layers = [0,6]
        
        # 扩展fusion_layers，因为Hippo输出是插入在层与层之间的
        # 如果在第n层后插入（即在n和n+1层之间），需要对n层和n+1层都进行微调
        extended_fusion_layers = set()
        for layer_idx in fusion_layers:
            # 添加当前层
            extended_fusion_layers.add(layer_idx)
            # 添加下一层（因为Hippo插入在layer_idx和layer_idx+1之间）
            if layer_idx < self.base_model.config.num_hidden_layers - 1:
                extended_fusion_layers.add(layer_idx + 1)
        
        self.fusion_layers = sorted(list(extended_fusion_layers))
        print(f"原始插入层: {fusion_layers}")
        print(f"扩展后LoRA微调层: {self.fusion_layers}")
        
        # 添加last_n_tokens参数，控制Hippo模型处理的token数量
        self.last_n_tokens = last_n_tokens
        
        # 初始化Hippo模型
        # 添加融合层索引越界校验
        for layer_idx in self.fusion_layers:
            if not (0 <= layer_idx < self.base_model.config.num_hidden_layers):
                raise ValueError(
                    f"融合层索引{layer_idx}超出有效范围（0到{self.base_model.config.num_hidden_layers-1}）"
                )
        
        # 初始化Hippo模型
        self.hippo_model = self._initialize_hippo_model()
        
        # 初始化门控机制字典，为每个融合位置创建一个门控
        self.gate_mechanisms = nn.ModuleDict()
        
        # 为每个融合层创建门控机制 (轻量级单层设计)
        for layer_idx in self.fusion_layers:
            # 使用单层轻量门控：简单加权融合 + sigmoid激活
            self.gate_mechanisms[f"layer_{layer_idx}"] = nn.Sequential(
                nn.Linear(self.config.hidden_size * 2, 64),  # 降维到1维权重
                nn.LeakyReLU(negative_slope=0.01),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        
        # 初始化隐藏状态，用在推理时
        self.hidden_h = None
        
        # 在指定的fusion_layers中应用LoRA
        self._apply_lora_to_fusion_layers()
        
        # 初始化门控参数 - 使用LoRA优化的初始化策略
        self._initialize_gate_mechanisms()
        
        # 保存当前训练阶段
        self.current_stage = 0  # 0: 仅训练hippo模型, 1: 微调transformer相邻层
        
        # 冻结所有参数
        self.freeze_all()
        
        # 初始阶段：LoRA参数从一开始就是可训练的
        self.unfreeze_hippo_model()
        self.unfreeze_adjacent_layers()  # 在LoRA中，这确保LoRA参数可训练
        
        # 验证LoRA是否正确应用
        self.verify_lora_applied()
        
        print(f"HippoLoRAQwen初始化完成，LoRA参数数量: {self._count_lora_parameters()}")
        
        # 打印Hippo模型参数信息
        hippo_trainable = sum(p.numel() for p in self.hippo_model.parameters() if p.requires_grad)
        hippo_total = sum(p.numel() for p in self.hippo_model.parameters())
        print(f"Hippo模型参数数量: {hippo_trainable} (可训练) / {hippo_total} (总计)")
        
        # 打印门控机制参数信息
        gate_trainable = sum(p.numel() for p in self.gate_mechanisms.parameters() if p.requires_grad)
        gate_total = sum(p.numel() for p in self.gate_mechanisms.parameters())
        print(f"门控机制参数数量: {gate_trainable} (可训练) / {gate_total} (总计)")
        
        # 打印总体参数统计
        base_trainable = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        base_total = sum(p.numel() for p in self.base_model.parameters())
        total_trainable = base_trainable + hippo_trainable + gate_trainable + self._count_lora_parameters()
        total_params = base_total + hippo_total + gate_total + self._count_lora_parameters()
        print(f"总体参数统计: {total_trainable} (可训练) / {total_params} (总计), 可训练比例: {total_trainable/total_params:.4f}")
        print("训练阶段设置:")
        print("- 阶段0: Hippo组件 + 门控机制 + LoRA参数")
        print("- 阶段1: 在阶段0基础上，无额外参数解冻（LoRA已全部参与训练）")
        print("初始化策略:")
        print("- LoRA_A: 高斯分布初始化（稳定训练）")
        print("- LoRA_B: 零初始化（基础模型主导）")
        print("- 门控: 保守Xavier初始化 + 强负偏置（偏向基础模型）")
        
    def _setup_lora_config(self, lora_rank, lora_alpha, lora_dropout, attn_lora_config, ffn_lora_config):
        """配置LoRA参数，支持独立配置不同层类型"""
        # 默认配置
        default_config = {
            'rank': lora_rank,
            'alpha': lora_alpha,
            'dropout': lora_dropout
        }
        
        # 注意力层配置（如果提供则覆盖默认配置）
        attn_config = default_config.copy()
        if attn_lora_config:
            attn_config.update(attn_lora_config)
            
        # FFN层配置（如果提供则覆盖默认配置）
        ffn_config = default_config.copy()
        if ffn_lora_config:
            ffn_config.update(ffn_lora_config)
            
        return {
            'default': default_config,
            'attn': attn_config,
            'ffn': ffn_config
        }
    
    def _initialize_hippo_model(self):
        """初始化Hippo模型，与基座模型精度保持一致"""
        return HippoModel(
            input_dim=self.config.hidden_size,
            output_dim=self.config.hidden_size,
            seq_len=self.seq_len,
            hippo_scale=0.01,
            dtype=self.base_model.dtype  # 传入基座模型精度
        ).to(self.base_model.device)
    
    def _apply_lora_to_fusion_layers(self):
        """在fusion层应用LoRA"""
        print("在fusion层应用LoRA...")
        for i, layer in enumerate(self.base_model.model.layers):
            if i in self.fusion_layers:
                print(f"  处理第{i}层 (fusion层)...")
                # 为fusion层使用默认配置
                lora_config = self.lora_config['default']
                print(f"    - rank (LoRA秩): {lora_config['rank']}")
                print(f"    - alpha (缩放参数): {lora_config['alpha']}")
                print(f"    - dropout (Dropout率): {lora_config['dropout']}")
                print(f"    - 目标层类型: FFN (Transformer Feed-Forward Network)")
                self._replace_linear_with_lora(layer, 'attn')
    
    def _replace_linear_with_lora(self, layer, component_type):
        """将线性层替换为LoRA层"""
        try:
            # 获取配置
            lora_config = self.lora_config.get(component_type, self.lora_config['default'])
            
            # 定义替换映射
            layer_mapping = {
                'fusion': [
                    ('mlp.gate_proj', 'gate_proj'),
                    ('mlp.down_proj', 'down_proj')
                ],
                'attn': [
                    ('self_attn.q_proj', 'Q_proj'),
                    ('self_attn.k_proj', 'K_proj'),
                    ('self_attn.v_proj', 'V_proj'),
                    ('self_attn.o_proj', 'O_proj')
                ],
                'ffn': [
                    ('mlp.gate_proj', 'gate_proj'),
                    ('mlp.up_proj', 'up_proj'),
                    ('mlp.down_proj', 'down_proj')
                ]
            }
            
            mapping = layer_mapping.get(component_type, [])
            
            for attr_path, name in mapping:
                self._replace_single_layer(layer, attr_path, name, lora_config)

        except Exception as e:
            print(f"替换{component_type}组件失败: {e}")
            
    def _replace_single_layer(self, layer, attr_path, display_name, lora_config):
        """替换单个线性层"""
        try:
            # 获取层对象
            parts = attr_path.split('.')
            obj = layer
            for part in parts[:-1]:
                obj = getattr(obj, part)
            
            attr_name = parts[-1]
            if not hasattr(obj, attr_name):
                return
                
            old_linear = getattr(obj, attr_name)
            
            # 创建LoRA层
            lora_linear = LoRALinear(
                in_features=old_linear.in_features,
                out_features=old_linear.out_features,
                r=lora_config['rank'],
                lora_alpha=lora_config['alpha'],
                lora_dropout=lora_config['dropout'],
                bias=old_linear.bias is not None
            )
            
            # 复制权重
            if lora_linear.weight.shape == old_linear.weight.shape:
                lora_linear.weight.data = old_linear.weight.data.clone()
            else:
                print(f"警告: {display_name}权重维度不匹配，期望{lora_linear.weight.shape}，实际{old_linear.weight.shape}")
                nn.init.kaiming_uniform_(lora_linear.weight, a=math.sqrt(5))
            
            # 复制偏置
            if old_linear.bias is not None:
                if lora_linear.bias is not None and lora_linear.bias.shape == old_linear.bias.shape:
                    lora_linear.bias.data = old_linear.bias.data.clone()
                else:
                    print(f"警告: {display_name}偏置维度不匹配，期望{lora_linear.bias.shape if lora_linear.bias is not None else 'None'}，实际{old_linear.bias.shape}")
                    nn.init.zeros_(lora_linear.bias)
            
            # 替换层
            setattr(obj, attr_name, lora_linear)
            print(f"  - 替换{display_name}: rank={lora_config['rank']}, alpha={lora_config['alpha']}")
            
        except Exception as e:
            print(f"替换{display_name}失败: {e}")


    def freeze_all(self):
        """冻结所有参数"""
        # 冻结基础模型参数
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # 冻结Hippo模型参数
        for param in self.hippo_model.parameters():
            param.requires_grad = False
        
        # 冻结门控机制参数
        for param in self.gate_mechanisms.parameters():
            param.requires_grad = False
    
    def unfreeze_hippo_model(self):
        """解冻Hippo模型和门控机制参数"""
        for param in self.hippo_model.parameters():
            param.requires_grad = True
        for param in self.gate_mechanisms.parameters():
            param.requires_grad = True

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """启用梯度检查点以节省内存"""
        # 将梯度检查点功能传递给基础模型
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
    
    def gradient_checkpointing_disable(self):
        """禁用梯度检查点"""
        # 将梯度检查点功能传递给基础模型
        if hasattr(self.base_model, 'gradient_checkpointing_disable'):
            self.base_model.gradient_checkpointing_disable()
    
    def unfreeze_adjacent_layers(self):
        """确保LoRA参数可训练（LoRA版本：保持函数名一致性，LoRA参数始终参与训练）"""
        print("确保LoRA适配器参数可训练...")
        
        # LoRA参数在LoRA微调中从一开始就是可训练的
        # 这个函数保持与model_customization.py的命名一致性
        for layer_idx in self.fusion_layers:
            # 检查当前层和下一层是否在模型范围内
            current_layer_idx = layer_idx
            next_layer_idx = layer_idx + 1
            
            # 确保当前层的LoRA参数可训练
            if current_layer_idx < len(self.base_model.model.layers):
                layer = self.base_model.model.layers[current_layer_idx]
                for name, module in layer.named_modules():
                    if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
                        # 确保LoRA参数可训练 (LoRA A和B是Parameter，不是Module)
                        if hasattr(module, 'lora_A') and module.lora_A is not None:
                            module.lora_A.requires_grad = True
                        if hasattr(module, 'lora_B') and module.lora_B is not None:
                            module.lora_B.requires_grad = True
            
            # 确保下一层的LoRA参数可训练（如果存在）
            if next_layer_idx < len(self.base_model.model.layers):
                layer = self.base_model.model.layers[next_layer_idx]
                for name, module in layer.named_modules():
                    if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
                        # 确保LoRA参数可训练 (LoRA A和B是Parameter，不是Module)
                        if hasattr(module, 'lora_A') and module.lora_A is not None:
                            module.lora_A.requires_grad = True
                        if hasattr(module, 'lora_B') and module.lora_B is not None:
                            module.lora_B.requires_grad = True
    
    def advance_training_stage(self):
        """推进训练阶段（LoRA版本：保持函数名一致性，实际无需额外参数解冻）"""
        if self.current_stage >= 1:
            print(f"已经在最高训练阶段：{self.current_stage}")
            return
        
        self.current_stage += 1
        print(f"训练阶段推进到：{self.current_stage}")
        
        if self.current_stage == 1:
            print("LoRA微调：所有可训练参数（Hippo、门控、LoRA）已在阶段0参与训练")
            print("阶段1主要调整训练策略，无额外参数解冻")
            
            # 在LoRA中，所有可训练参数（Hippo、门控、LoRA）都已参与训练
            # 推进阶段主要为了与model_customization.py保持调用接口一致
    
    def _is_layer_frozen(self, layer_idx):
        """检查指定层的Transformer层是否有可训练参数"""
        if layer_idx >= len(self.base_model.model.layers):
            return True  # 超出范围认为冻结
        
        layer = self.base_model.model.layers[layer_idx]
        
        # 检查是否有任何参数可训练
        for param in layer.parameters():
            if param.requires_grad:
                return False  # 有可训练参数，层未冻结
        
        return True  # 所有参数都冻结
    
    def _is_gate_frozen(self, layer_idx):
        """检查指定层位置的门控机制是否冻结"""
        gate_name = f"layer_{layer_idx}"
        if gate_name not in self.gate_mechanisms:
            return True  # 不存在认为冻结
        
        gate_module = self.gate_mechanisms[gate_name]
        
        # 检查门控机制是否有任何参数可训练
        for param in gate_module.parameters():
            if param.requires_grad:
                return False  # 有可训练参数，门控未冻结
        
        return True  # 所有参数都冻结
    
    def _initialize_gate_mechanisms(self):
        """初始化门控机制参数，为LoRA微调优化的初始化策略（轻量级单层设计）"""
        for gate_module in self.gate_mechanisms.values():
            linear_layer = gate_module[0]  # 获取线性层（现在是唯一的线性层）
            
            # 确保门控参数是可训练的
            linear_layer.weight.requires_grad = True
            if linear_layer.bias is not None:
                linear_layer.bias.requires_grad = True
            
            # LoRA微调场景的轻量级门控初始化策略：
            # 1. 权重使用较小的Xavier初始化，让基础模型主导输出
            # 输入维度: hidden_size * 2，输出维度: 1
            nn.init.xavier_uniform_(linear_layer.weight, gain=0.05)  # 更小的gain，保守初始化
            
            # 2. 偏置初始化为负值，确保初始门控值偏向基础模型
            # 这对LoRA很重要，因为基础模型权重是冻结的
            if linear_layer.bias is not None:
                nn.init.constant_(linear_layer.bias, -3.0)  # 更强的负偏置，确保sigmoid输出接近0
    
    def _count_lora_parameters(self):
        """计算LoRA参数数量"""
        total_params = 0
        for name, param in self.named_parameters():
            # 更灵活的参数名检查，不仅检查'lora_'前缀
            # 直接检查LoRA参数的确切名称: lora_A 和 lora_B
            if 'lora_A' in name or 'lora_B' in name:
                total_params += param.numel()
        return total_params
    
    def verify_lora_applied(self):
        """验证LoRA层是否被正确应用"""
        lora_layers = []
        for name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                lora_layers.append(name)
        
        print(f"成功应用LoRA的层数: {len(lora_layers)}")
        print(f"LoRA层位置: {lora_layers}")
        
        return len(lora_layers) > 0
    
    def get_trainable_params_info(self):
        """获取当前可训练参数信息"""
        # 计算基础模型可训练参数（基础模型参数在LoRA中通常被冻结）
        base_trainable = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        base_total = sum(p.numel() for p in self.base_model.parameters())
        
        # 计算Hippo模型可训练参数
        hippo_trainable = sum(p.numel() for p in self.hippo_model.parameters() if p.requires_grad)
        hippo_total = sum(p.numel() for p in self.hippo_model.parameters())
        
        # 计算门控机制可训练参数
        gate_trainable = sum(p.numel() for p in self.gate_mechanisms.parameters() if p.requires_grad)
        gate_total = sum(p.numel() for p in self.gate_mechanisms.parameters())
        
        # 计算LoRA参数
        lora_trainable = self._count_lora_parameters()
        lora_total = lora_trainable  # LoRA参数全部都是可训练的
        
        # 计算总参数
        trainable_params = base_trainable + hippo_trainable + gate_trainable + lora_trainable
        total_params = base_total + hippo_total + gate_total + lora_total
        
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_ratio": trainable_params / total_params if total_params > 0 else 0.0,
            "base_model_trainable": base_trainable,
            "hippo_model_trainable": hippo_trainable,
            "gate_mechanisms_trainable": gate_trainable,
            "lora_trainable": lora_trainable,
            "current_stage": self.current_stage
        }
    
    def _is_layer_frozen(self, layer_idx):
        """检查指定Transformer层是否被冻结（LoRA版本：检查是否有可训练的LoRA参数）"""
        if 0 <= layer_idx < len(self.base_model.model.layers):
            layer = self.base_model.model.layers[layer_idx]
            # 在LoRA中，基础权重被冻结，检查是否有LoRA参数需要训练
            lora_params = []
            for name, module in layer.named_modules():
                if hasattr(module, 'lora_A') or 'lora_' in name:
                    lora_params.extend([p for p in module.parameters() if p.requires_grad])
            return len(lora_params) == 0
        return True
    
    def _is_gate_frozen(self, layer_idx):
        """检查指定层的门控机制是否被冻结"""
        if f"layer_{layer_idx}" in self.gate_mechanisms:
            return not any(p.requires_grad for p in self.gate_mechanisms[f"layer_{layer_idx}"].parameters())
        return True
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, dialog_histories=None):
        """
        前向传播方法，使用单个HippoModel处理第一层输入，并在指定层间进行门控融合
        
        参数:
            input_ids: 大模型输入ID
            attention_mask: 注意力掩码
            labels: 标签（可选）
            dialog_histories: 对话历史列表（可选）
        """
        is_training = self.training
        batch_size, seq_len = input_ids.shape
        
        if self.hidden_h is None:
            self.hidden_h = self.hippo_model.reset_h(batch_size=batch_size)

        if attention_mask is None:
            # 仅当attention_mask为None时才初始化为全1
            attention_mask = torch.ones_like(input_ids, dtype=self.base_model.dtype, device=input_ids.device)
        else:
            # 确保传入的attention_mask也使用正确的数据类型
            attention_mask = attention_mask.to(dtype=self.base_model.dtype)

        hidden_states = self.base_model.model.embed_tokens(input_ids)
        # print(f"嵌入层输出 min: {hidden_states.min().item()}, max: {hidden_states.max().item()}, has nan: {torch.isnan(hidden_states).any().item()}")

        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).repeat(batch_size, 1)  # [batch_size, seq_len]
        
        # 初始化cache_position（用于缓存机制），按照Qwen3源码方式计算
        cache_position = torch.arange(
            0, seq_len, device=input_ids.device
        )
        
        # 使用与Qwen3相同的方式生成position_embeddings
        position_embeddings = self.base_model.model.rotary_emb(hidden_states, position_ids)
             
        with torch.no_grad():
            # 检查attention_mask是否已经是字典格式（可能是之前已经处理过的）
            if not isinstance(attention_mask, dict):
                # 准备掩码生成参数
                mask_kwargs = {
                    "config": self.base_model.config,
                    "input_embeds": hidden_states,
                    "attention_mask": attention_mask,
                    "cache_position": cache_position,
                    "past_key_values": None,  # 前向传播时没有历史缓存
                    "position_ids": position_ids,
                }
                
                # 生成因果掩码映射
                causal_mask_mapping = {
                    "full_attention": create_causal_mask(**mask_kwargs),
                }
                
                # 如果模型有滑动窗口层，也生成对应的掩码
                if hasattr(self.base_model.config, 'layer_types') and "sliding_attention" in self.base_model.config.layer_types:
                    causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
            else:
                # 如果已经是字典格式，直接使用
                causal_mask_mapping = attention_mask
                
            # 当dialog_histories不为None时，使用对话历史更新HippoModel的隐藏状态
            if dialog_histories is not None:
                # 证明模型是在训练
                is_training = True
                # 解析格式维度
                batch_size, num_turns, seq_len = dialog_histories.shape
                # 初始化隐藏状态
                h_initial = self.hippo_model.reset_h(batch_size)
                            
                # 按轮次遍历（原逻辑是遍历列表，现在遍历张量的num_turns维度）
                for turn_idx in range(num_turns):
                    history_batch = dialog_histories[:, turn_idx, :]  # 取第turn_idx轮 (batch_size, seq_len)
                    
                    # 跳过全为pad的轮次
                    if (history_batch == self.tokenizer.pad_token_id).all():
                        continue

                    # 计算历史嵌入（仅词嵌入，不需要位置编码，因为历史对话不经过Qwen的transformer层）
                    history_embeds = self.base_model.model.embed_tokens(history_batch)
                    
                    # 更新Hippo隐藏状态，传入last_n_tokens参数
                    _, h_initial = self.hippo_model(history_embeds, h_initial, last_n_tokens=self.last_n_tokens)
                    self.hidden_h = h_initial

        # 使用Hippo模型处理第一层transformer的输入，传入更新后的隐藏状态
        # 始终计算梯度，但只有在非冻结时才会更新参数
        hippo_output, self.hidden_h = self.hippo_model(hidden_states, self.hidden_h, last_n_tokens=self.last_n_tokens)
        # print(f"Hippo输出 min: {hippo_output.min().item()}, max: {hippo_output.max().item()}, has nan: {torch.isnan(hippo_output).any().item()}")
        
        # 处理每一层transformer
        for layer_idx, layer in enumerate(self.base_model.model.layers):
            # 检查当前层是否冻结
            is_layer_frozen = self._is_layer_frozen(layer_idx)
            
            # 对被冻结的层使用no_grad()以节省显存和计算资源
            with torch.set_grad_enabled(is_training):
                
                # 根据层类型选择合适的attention_mask
                if hasattr(layer, 'attention_type'):
                    layer_attention_mask = causal_mask_mapping[layer.attention_type]
                else:
                    # 默认为full_attention
                    layer_attention_mask = causal_mask_mapping["full_attention"]
                
                # 调用Qwen3DecoderLayer的正确参数
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=layer_attention_mask,
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    use_cache=False
                )

                hidden_states = layer_outputs
            
            print(f"hidden_states min: {hidden_states.min().item()}, max: {hidden_states.max().item()}, has nan: {torch.isnan(hidden_states).any().item()}")    
            # 检查当前层是否是需要融合Hippo输出的位置
            if layer_idx in self.fusion_layers:
                # 门控机制是否冻结
                is_gate_frozen = self._is_gate_frozen(layer_idx)
                
                with torch.set_grad_enabled(is_training):
                    # 获取当前层的门控机制
                    gate_mechanism = self.gate_mechanisms[f"layer_{layer_idx}"]
                    
                    # 计算门控权重
                    gate_input = torch.cat([hidden_states, hippo_output], dim=-1)
                    gate_weight = gate_mechanism(gate_input)
                    print(f"相加前hidden_states min: {hidden_states.min().item()}, max: {hidden_states.max().item()}, has nan: {torch.isnan(hidden_states).any().item()}")
                    # 门控加权融合：分别对两个输入加权后求和
                    hidden_states = gate_weight * hippo_output + (1 - gate_weight) * hidden_states
                    print(f"相加后hidden_states min: {hidden_states.min().item()}, max: {hidden_states.max().item()}, has nan: {torch.isnan(hidden_states).any().item()}")
            hidden_states = self.base_model.model.norm(hidden_states)
        
        hidden_states = self.base_model.model.norm(hidden_states)
        logits = self.base_model.lm_head(hidden_states)
        # print(f"norm后hidden_states min: {hidden_states.min().item()}, max: {hidden_states.max().item()}, has nan: {torch.isnan(hidden_states).any().item()}")
        # 计算损失（如果提供了labels）
        loss = None
        if labels is not None:
            # 计算语言模型损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        print(f"logits min: {logits.min().item()}")
        print(f"logits max: {logits.max().item()}")
        print(f"logits has nan: {torch.isnan(logits).any().item()}")
        print(f"logits has inf: {torch.isinf(logits).any().item()}")
        # 构造输出对象（已在顶部导入）
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits
        )
    
    def generate(self, **kwargs):
        """生成方法"""
        return self.base_model.generate(**kwargs)
    
    def save_pretrained(self, save_directory, model_type="lora", save_hippo_components=True):
        """保存模型"""
        import os
        import json
        
        # 确保基础目录存在
        os.makedirs(save_directory, exist_ok=True)
        
        if model_type == "hippo" and save_hippo_components:
            # 只保存Hippo相关组件到hippo_model子目录
            hippo_dir = os.path.join(save_directory, "hippo_model")
            os.makedirs(hippo_dir, exist_ok=True)
            
            # 保存Hippo组件
            torch.save({
                'hippo_state_dict': self.hippo_model.state_dict(),
                'gate_mechanisms': self.gate_mechanisms.state_dict(),
                'fusion_layers': self.fusion_layers
            }, os.path.join(hippo_dir, "hippo_lora_components.pt"))
            
            # 保存配置
            config = {
                'base_model_name_or_path': self.base_model_name_or_path,
                'fusion_layers': self.fusion_layers,
                'model_type': 'hippo_lora_only'
            }
            with open(os.path.join(hippo_dir, "config.json"), 'w') as f:
                json.dump(config, f, indent=2)
                
            print(f"Hippo组件已保存至: {hippo_dir}")
            return
        
        # 完整LoRA模型保存
        base_dir = os.path.join(save_directory, "lora_finetuning")
        os.makedirs(base_dir, exist_ok=True)
        
        # 保存基础模型
        self.base_model.save_pretrained(os.path.join(base_dir, "base_model"))
        
        # 保存Hippo组件和LoRA参数
        torch.save({
            'hippo_state_dict': self.hippo_model.state_dict(),
            'gate_mechanisms': self.gate_mechanisms.state_dict(),
            'fusion_layers': self.fusion_layers,
            'lora_info': {
                'applied_to_layers': self.fusion_layers,
                'lora_rank': 8,
                'lora_alpha': 32,
                'lora_dropout': 0.1
            }
        }, os.path.join(base_dir, "hippo_lora_components.pt"))
        
        # 保存配置
        config = {
            'base_model_name_or_path': self.base_model_name_or_path,
            'fusion_layers': self.fusion_layers,
            'model_type': 'hippo_lora_qwen',
            'save_timestamp': torch.cuda.Event().query() if torch.cuda.is_available() else 0
        }
        
        with open(os.path.join(base_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"LoRA模型已保存到 {base_dir}")
        print("- 基础模型权重")
        print("- Hippo模型参数")
        print("- 门控机制参数")
        print("- LoRA适配器参数")
        print("- 模型配置")
    
    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        """从保存的模型加载"""
        # 加载配置
        with open(os.path.join(model_path, "config.json"), 'r') as f:
            import json
            config = json.load(f)
        
        # 创建模型实例
        model = cls(
            base_model_name_or_path=os.path.join(model_path, "base_model"),
            fusion_layers=config['fusion_layers']
        )
        
        # 加载Hippo组件
        components = torch.load(os.path.join(model_path, "hippo_lora_components.pt"))
        model.hippo_model.load_state_dict(components['hippo_state_dict'])
        model.gate_mechanisms.load_state_dict(components['gate_mechanisms'])
        
        return model


# 使用示例
if __name__ == "__main__":
    # 创建LoRA版本模型
    model = HippoLoRAQwen(
        base_model_name_or_path="Qwen/Qwen2-1.5B-Instruct",
        fusion_layers=[6, 12, 18],  # 在第6, 12, 18层进行LoRA微调
        hippo_model_config={
            'model_dim': 1536,
            'state_dim': 256,
            'kernel_size': 16,
            'num_layers': 2
        }
    )
    
    # 查看可训练参数
    print(model.get_trainable_params_info())
    
    # 简单测试
    import torch
    input_ids = torch.randint(0, 1000, (2, 10))
    attention_mask = torch.ones_like(input_ids)
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    print(f"输出形状: {outputs['logits'].shape}")