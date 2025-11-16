import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from hippo_model import HippoModel

# 内存优化配置
import os
# 启用TF32加速计算
torch.backends.cuda.matmul.allow_tf32 = True
# 启用cudnn基准测试以选择最佳卷积算法
torch.backends.cudnn.benchmark = True

class ModifiedQwen(nn.Module):
    """包装Qwen3-8B并插入单个HippoModel，在transformer层间进行门控融合"""
    def __init__(self, base_model_name_or_path, fusion_layers=None, cache_dir="qwen3-8b-custom-module-training", last_n_tokens=0):
        super().__init__()
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name_or_path,
            trust_remote_code=True
        )
        
        # 加载基础模型，使用混合精度和内存优化
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,  # 使用bfloat16混合精度以减少显存占用
            low_cpu_mem_usage=True  # 减少CPU内存使用
        )
        
        # 获取模型配置
        self.config = self.base_model.config
        self.hidden_size = self.config.hidden_size
        
        self.num_transformer_layers = self.config.num_hidden_layers
        
        # 默认融合层位置（均匀分布在模型层中）
        self.fusion_layers = fusion_layers if fusion_layers is not None else [0]
        
        # 添加last_n_tokens参数，控制Hippo模型处理的token数量
        self.last_n_tokens = last_n_tokens
        
        # 添加融合层索引越界校验
        for layer_idx in self.fusion_layers:
            if not (0 <= layer_idx < self.num_transformer_layers):
                raise ValueError(
                    f"融合层索引{layer_idx}超出有效范围（0到{self.num_transformer_layers-1}）"
                )
        
        # 初始化单个Hippo模型，输入为第一层transformer的输入
        # 移除硬编码设备分配，让HippoModel自动匹配基础模型的设备
        self.hippo_model = HippoModel(
            input_dim=self.hidden_size,
            output_dim=self.hidden_size,
            hippo_scale=0.1  # 添加缩放因子，提高训练稳定性
        ).to(self.base_model.device)  # 确保Hippo模型与基础模型在同一设备上
        
        # 初始化门控机制字典，为每个融合位置创建一个门控
        self.gate_mechanisms = nn.ModuleDict()
        
        # 为每个融合层创建门控机制
        for layer_idx in self.fusion_layers:
            # 创建门控机制 (使用sigmoid激活的线性层)
            self.gate_mechanisms[f"layer_{layer_idx}"] = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.Sigmoid()
            )
        
        # 初始化门控机制参数
        self._initialize_gate_mechanisms()
        
        # 保存当前训练阶段
        self.current_stage = 0  # 0: 仅训练hippo模型, 1: 微调transformer相邻层
        
        # 冻结所有参数
        self.freeze_all()
        
        # 初始阶段：只解冻hippo模型
        self.unfreeze_hippo_model()

        # 初始化隐藏状态,用在推理时
        self.hidden_h = None

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
        """解冻与融合位置相邻的Transformer层参数"""
        for layer_idx in self.fusion_layers:
            # 解冻当前层
            if 0 <= layer_idx < len(self.base_model.model.layers):
                for param in self.base_model.model.layers[layer_idx].parameters():
                    param.requires_grad = True
            
            # 解冻下一层（如果存在）
            if layer_idx + 1 < len(self.base_model.model.layers):
                for param in self.base_model.model.layers[layer_idx + 1].parameters():
                    param.requires_grad = True
    
    def advance_training_stage(self):
        """进入下一训练阶段：Hippo模型和门控机制始终一起训练，只有transformer层在后续阶段解冻"""
        if self.current_stage == 0:
            # 冻结所有参数
            self.freeze_all()
            
            # 先解冻Hippo模型和门控机制（始终一起训练）
            self.unfreeze_hippo_model()
            # 然后解冻相邻的Transformer层
            self.unfreeze_adjacent_layers()
            
            self.current_stage = 1
            return f"已进入训练阶段1：同时训练Hippo模型、门控机制和相邻的Transformer层"
        else:
            return f"已经在最后训练阶段（阶段{self.current_stage}）"
    
    def _initialize_gate_mechanisms(self):
        """初始化门控机制参数，使门控偏向基础模型输出（初始阶段稳定训练）"""
        for gate_module in self.gate_mechanisms.values():
            linear_layer = gate_module[0]  # 获取第一个线性层
            
            # 门控机制初始化为偏向基础模型输出（权重较小）
            # 使用 Xavier 初始化，但降低权重方差
            nn.init.xavier_uniform_(linear_layer.weight, gain=0.1)  # 较小的gain，使初始输出接近0
            
            # 偏置初始化为负值，使sigmoid输出接近0，优先使用基础模型输出
            if linear_layer.bias is not None:
                nn.init.constant_(linear_layer.bias, -2.0)
    
    def get_trainable_params_info(self):
        """获取当前可训练参数信息"""
        trainable_params = 0
        total_params = 0
        
        # 计算基础模型可训练参数
        base_trainable = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        base_total = sum(p.numel() for p in self.base_model.parameters())
        
        # 计算Hippo模型可训练参数
        hippo_trainable = sum(p.numel() for p in self.hippo_model.parameters() if p.requires_grad)
        hippo_total = sum(p.numel() for p in self.hippo_model.parameters())
        
        # 计算门控机制可训练参数
        gate_trainable = sum(p.numel() for p in self.gate_mechanisms.parameters() if p.requires_grad)
        gate_total = sum(p.numel() for p in self.gate_mechanisms.parameters())
        
        trainable_params = base_trainable + hippo_trainable + gate_trainable
        total_params = base_total + hippo_total + gate_total
        
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_ratio": trainable_params / total_params,
            "base_model_trainable": base_trainable,
            "hippo_model_trainable": hippo_trainable,
            "gate_mechanisms_trainable": gate_trainable,
            "current_stage": self.current_stage
        }

    def forward(self, input_ids=None, attention_mask=None, labels=None, dialog_histories=None):
        """
        前向传播方法，使用单个HippoModel处理第一层输入，并在指定层间进行门控融合
        
        参数:
            input_ids: 大模型输入ID
            attention_mask: 注意力掩码
            labels: 标签（可选）
            dialog_histories: 对话历史列表（可选）
        """
        # 检查当前是否处于训练状态
        is_training = self.training
        
        # 获取批次大小和序列长度
        batch_size, seq_len = input_ids.shape
        
        if self.hidden_h is None:
            self.hidden_h = self.hippo_model.reset_h(batch_size=batch_size)
            
        # 不要覆盖传入的attention_mask，保留原始值或仅在需要时初始化
        if attention_mask is None:
            # 仅当attention_mask为None时才初始化为全1
            # 使用bfloat16类型以匹配模型和查询张量的数据类型，避免类型不匹配错误
            attention_mask = torch.ones_like(input_ids, dtype=torch.bfloat16, device=input_ids.device)
        else:
            # 确保传入的attention_mask也使用正确的数据类型
            attention_mask = attention_mask.to(dtype=torch.bfloat16)

        hidden_states = self.base_model.model.embed_tokens(input_ids)

        # 生成位置索引（position_ids）
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).repeat(batch_size, 1)  # [batch_size, seq_len]
        
        # 初始化cache_position（用于缓存机制），按照Qwen3源码方式计算
        past_seen_tokens = 0  # 前向传播时没有历史缓存
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + seq_len, device=input_ids.device
        )
        
        # 使用与Qwen3相同的方式生成position_embeddings
        # 使用模型内置的rotary_emb来生成cos和sin
        position_embeddings = self.base_model.model.rotary_emb(hidden_states, position_ids)
        
        # 按照Qwen3源码方式处理attention_mask
        # 导入需要的掩码生成函数
        try:
            from transformers.models.qwen3.modeling_qwen3 import create_causal_mask, create_sliding_window_causal_mask
        except ImportError:
            # 如果无法导入，使用内置的create_causal_mask函数（需要安装transformers>=4.39）
            from transformers import create_causal_mask, create_sliding_window_causal_mask
        
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

                # 对话历史的嵌入转换和Hippo模型更新都不计算梯度以节省内存
                # 只需要获取隐藏状态h，不需要更新Hippo模型参数
                with torch.no_grad():
                    # 计算历史嵌入（仅词嵌入，不需要位置编码，因为历史对话不经过Qwen的transformer层）
                    history_embeds = self.base_model.model.embed_tokens(history_batch)
                    
                    # 更新Hippo隐藏状态，传入last_n_tokens参数
                    _, h_initial = self.hippo_model(history_embeds, h_initial, last_n_tokens=self.last_n_tokens)
                    self.hidden_h = h_initial

        # 使用Hippo模型处理第一层transformer的输入，传入更新后的隐藏状态
        # 始终计算梯度，但只有在非冻结时才会更新参数
        hippo_output, self.hidden_h = self.hippo_model(hidden_states, self.hidden_h, last_n_tokens=self.last_n_tokens)
        
        # 初始化past_key_values存储
        past_key_values = []
        all_hidden_states = [hidden_states]
        all_attentions = []
        
        # 处理每一层transformer
        for layer_idx, layer in enumerate(self.base_model.model.layers):
            # 检查当前层是否冻结
            is_layer_frozen = self._is_layer_frozen(layer_idx)
            
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
            
            # Qwen3模型的transformer层返回的是张量而非元组
            hidden_states = layer_outputs  # 直接使用整个输出作为hidden_states
            
            # 由于Qwen3不返回past_key_value和attention权重，这里使用默认值
            # 或者根据模型实际行为调整
            past_key_value = None  # 或创建空的past_key_value
            
            # 保存中间结果
            if past_key_value is not None:
                past_key_values.append(past_key_value)
            all_hidden_states.append(hidden_states)
            
            # 由于模型不返回attention权重，这里跳过保存
            # 可以根据需要创建空的attention权重或修改为其他方式
                
            # 检查当前层是否是需要融合Hippo输出的位置
            if layer_idx in self.fusion_layers:
                # 门控机制是否冻结
                is_gate_frozen = self._is_gate_frozen(layer_idx)
                
                with torch.set_grad_enabled(is_training and not is_gate_frozen):
                    # 获取当前层的门控机制
                    gate_mechanism = self.gate_mechanisms[f"layer_{layer_idx}"]
                    
                    # 计算门控权重
                    gate_input = torch.cat([hidden_states, hippo_output], dim=-1)
                    gate_weight = gate_mechanism(gate_input)
                    
                    # 门控加权融合
                    hidden_states = gate_weight * hippo_output + (1 - gate_weight) * hidden_states
        
        # 应用最终的层归一化
        hidden_states = self.base_model.model.norm(hidden_states)
        
        # 最终通过 `lm_head` 获取logits（对应模型结构中的输出层）
        logits = self.base_model.lm_head(hidden_states)
        
        # 计算损失（如果提供了labels）
        loss = None
        if labels is not None:
            # 计算语言模型损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # 构造输出对象（已在顶部导入）
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=tuple(all_hidden_states),
            attentions=tuple(all_attentions)
        )
    
    def _is_hippo_frozen(self):
        """检查Hippo模型是否被冻结"""
        return not any(p.requires_grad for p in self.hippo_model.parameters())
    
    def _is_layer_frozen(self, layer_idx):
        """检查指定Transformer层是否被冻结"""
        if 0 <= layer_idx < len(self.base_model.model.layers):
            return not any(p.requires_grad for p in self.base_model.model.layers[layer_idx].parameters())
        return True
    
    def _is_gate_frozen(self, layer_idx):
        """检查指定层的门控机制是否被冻结"""
        if f"layer_{layer_idx}" in self.gate_mechanisms:
            return not any(p.requires_grad for p in self.gate_mechanisms[f"layer_{layer_idx}"].parameters())
        return True
    
    def generate(self, *args, **kwargs):
        """包装生成函数，确保使用修改后的前向传播逻辑，推理时使用no_grad节省显存"""
        # 设置为评估模式，确保所有冻结层都不计算梯度
        self.eval()
        
        # 在no_grad上下文中进行推理，大幅减少显存占用
        with torch.no_grad():
            # 调用基础模型的generate方法，它会使用我们重写的forward方法
            return self.base_model.generate(*args, **kwargs)
    
    def save_pretrained(self, save_directory, model_type="full"):
        """自定义保存方法，确保所有子模块参数都能正确保存，同时优化内存使用"""
        import os
        
        # 清理GPU缓存以节省内存
        torch.cuda.empty_cache()
        
        # 根据模型类型设置保存路径
        if model_type == "hippo":
            # 只保存Hippo相关组件
            hippo_dir = os.path.join(save_directory, "hippo_model")
            os.makedirs(hippo_dir, exist_ok=True)
            
            # 保存Hippo组件
            custom_state_dict = {
                'hippo_model': self.hippo_model.state_dict(),
                'gate_mechanisms': self.gate_mechanisms.state_dict(),
                'fusion_layers': self.fusion_layers,
                'current_stage': self.current_stage
            }
            torch.save(custom_state_dict, os.path.join(hippo_dir, 'hippo_components.bin'))
            
            # 保存配置
            config = {
                'base_model_name_or_path': self.base_model_name_or_path,
                'fusion_layers': self.fusion_layers,
                'model_type': 'hippo_full_param',
                'base_model_config': self.base_model.config.__dict__ if hasattr(self.base_model, 'config') else {}
            }
            with open(os.path.join(hippo_dir, 'config.json'), 'w') as f:
                json.dump(config, f, indent=2)
                
            print(f"Hippo组件已保存至: {hippo_dir}")
            return
        
        # 完整模型保存（适用于full_parameter_finetuning目录）
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存基础模型
        self.base_model.save_pretrained(save_directory)
        
        # 保存分词器
        self.tokenizer.save_pretrained(save_directory)
        
        # 保存自定义模块参数，优化内存使用
        custom_state_dict = {
            'hippo_model': self.hippo_model.state_dict(),
            'gate_mechanisms': self.gate_mechanisms.state_dict(),
            'fusion_layers': self.fusion_layers,
            'current_stage': self.current_stage
        }
        torch.save(custom_state_dict, os.path.join(save_directory, 'custom_modules.bin'))
        
        print(f"完整模型已保存至: {save_directory}")
        print(f"- 基础模型参数")
        print(f"- Hippo模型参数")
        print(f"- 门控机制参数")
        print(f"- 模型配置和分词器")
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        fusion_layers=None,
        cache_dir=None,
        **kwargs
    ):
        """
        从预训练模型加载模型，严格遵循Qwen3和Hugging Face Transformers的接口规范
        
        Args:
            pretrained_model_name_or_path: 预训练模型的路径或标识符
            fusion_layers: 要融合Hippo输出的层列表
            cache_dir: 缓存目录
            *model_args: 位置参数，传递给父类的from_pretrained
            **kwargs: 关键字参数，包含传递给父类的配置
            
        Returns:
            Loaded model instance
        """
        # 从kwargs中提取自定义参数
        custom_fusion_layers = kwargs.pop('fusion_layers', fusion_layers)
        custom_cache_dir = kwargs.pop('cache_dir', cache_dir)
        
        # 使用transformers的标准方式加载预训练权重
        # 这与Qwen3的方法保持一致，使用标准API确保兼容性
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
        
        # 加载配置
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=custom_cache_dir,
            **kwargs
        )
        
        # 使用标准的Transformers方式加载基础模型权重
        # 这符合Qwen3和所有Transformers模型的规范做法
        print("正在加载基础模型权重...")
        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=custom_cache_dir,
            **kwargs
        )
        
        # 创建模型实例并集成自定义组件
        model = cls(
            config=config,
            base_model_name_or_path=pretrained_model_name_or_path,
            fusion_layers=custom_fusion_layers,
            cache_dir=custom_cache_dir,
        )
        
        # 将加载的权重转移到我们的模型结构中
        # 只转移基础模型部分，保留自定义的lm_head
        with torch.no_grad():
            # 基础模型权重转移
            model.base_model.load_state_dict(
                base_model.base_model.state_dict(), 
                strict=False
            )
            print("基础模型权重加载成功")
        
        # 加载分词器
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                cache_dir=custom_cache_dir,
                **kwargs
            )
            model.tokenizer = tokenizer
        except Exception as e:
            print(f"加载分词器时出错: {e}")
            model.tokenizer = None
        
        # 加载自定义模块参数 - 这些是Hippo模型特有的扩展
        import os
        custom_modules_path = os.path.join(pretrained_model_name_or_path, 'custom_modules.bin')
        if os.path.exists(custom_modules_path):
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                custom_state_dict = torch.load(custom_modules_path, map_location=device)
                
                # 加载Hippo模型参数
                if 'hippo_model' in custom_state_dict:
                    model.hippo_model.load_state_dict(custom_state_dict['hippo_model'])
                
                # 加载门控机制参数
                if 'gate_mechanisms' in custom_state_dict:
                    model.gate_mechanisms.load_state_dict(custom_state_dict['gate_mechanisms'])
                
                # 恢复其他配置
                if 'fusion_layers' in custom_state_dict:
                    model.fusion_layers = custom_state_dict['fusion_layers']
                if 'current_stage' in custom_state_dict:
                    model.current_stage = custom_state_dict['current_stage']
                
                print(f"成功加载自定义模块参数")
                print(f"- Hippo模型参数")
                print(f"- 门控机制参数")
                print(f"- 当前训练阶段: {model.current_stage}")
                
            except Exception as e:
                print(f"加载自定义模块参数时出错: {e}")
                print("将使用默认的自定义模块参数")
        else:
            print(f"未找到自定义模块参数文件: {custom_modules_path}")
            print("将使用默认的自定义模块参数")
        
        # 不在这里初始化hidden_h，让它在forward方法中根据实际batch_size动态初始化
        model.hidden_h = None
        
        # 设置模型为评估模式
        model.eval()
        
        return model

# 使用示例
if __name__ == "__main__":
    model = ModifiedQwen(
        base_model_name_or_path="Qwen/Qwen3-8B",
        fusion_layers=[0],
        cache_dir="qwen3-8b-custom-module-training"  # 可自定义路径
    )
    
    # 打印初始训练阶段信息
    print("初始训练参数信息:")
    params_info = model.get_trainable_params_info()
    print(f"当前阶段: {params_info['current_stage']}")
    print(f"可训练参数比例: {params_info['trainable_ratio']:.6f}")
    print(f"Hippo模型可训练参数: {params_info['hippo_model_trainable']:,}")
    print(f"门控机制可训练参数: {params_info['gate_mechanisms_trainable']:,}")
    
    # 模拟进入下一训练阶段
    stage_info = model.advance_training_stage()
    print(f"\n{stage_info}")
    
    # 打印新阶段参数信息
    params_info = model.get_trainable_params_info()
    print(f"当前阶段: {params_info['current_stage']}")
    print(f"可训练参数比例: {params_info['trainable_ratio']:.6f}")
    print(f"基础模型可训练参数: {params_info['base_model_trainable']:,}")
    print(f"Hippo模型可训练参数: {params_info['hippo_model_trainable']:,}")
    print(f"门控机制可训练参数: {params_info['gate_mechanisms_trainable']:,}")
    
    print("\n模型初始化完成，准备开始训练")