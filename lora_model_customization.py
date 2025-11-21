import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import math
import os
from hippo_model import HippoModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import create_causal_mask, create_sliding_window_causal_mask

class HippoLoRAQwen(nn.Module):
    """
    使用PEFT内置LoRA的Qwen模型，集成Hippo状态空间建模
    """
    def __init__(self, base_model_name_or_path, fusion_layers=None, seq_len=4096, 
                 lora_rank=8, lora_alpha=32, lora_dropout=0.1, cache_dir="qwen3-8b-custom-module-training", 
                 last_n_tokens=0):
        super().__init__()
        
        self.base_model_name_or_path = base_model_name_or_path
        self.seq_len = seq_len
        self.last_n_tokens = last_n_tokens
        
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
            low_cpu_mem_usage=True
        )

        self.config = self.base_model.config
        self.hidden_size = self.config.hidden_size
        self.num_transformer_layers = self.config.num_hidden_layers

        # 设置融合层
        if fusion_layers is None:
            # 默认选择前几层进行微调
            fusion_layers = [6]
        
        # 扩展融合层：在指定层及其后一层应用LoRA
        extended_fusion_layers = set()
        for layer_idx in fusion_layers:
            extended_fusion_layers.add(layer_idx)
            if layer_idx < self.num_transformer_layers - 1:
                extended_fusion_layers.add(layer_idx + 1)
        
        self.fusion_layers = sorted(list(extended_fusion_layers))
        print(f"LoRA微调层: {self.fusion_layers}")
        
        # 应用PEFT内置LoRA
        self._apply_peft_lora(lora_rank, lora_alpha, lora_dropout)
        
        # 初始化Hippo模型
        self.hippo_model = HippoModel(
            input_dim=self.config.hidden_size,
            output_dim=self.config.hidden_size,
            seq_len=self.seq_len,
            hippo_scale=0.01,
            dtype=self.base_model.dtype
        ).to(self.base_model.device)
        
        # 初始化门控机制
        self.gate_mechanisms = nn.ModuleDict()
        for layer_idx in self.fusion_layers:
            self.gate_mechanisms[f"layer_{layer_idx}"] = nn.Sequential(
                nn.Linear(self.config.hidden_size * 2, 16),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        
        # 专门初始化门控机制
        self._initialize_gate_mechanisms()
        
        # 初始化隐藏状态
        self.hidden_h = None
        
        # 冻结基础模型参数，只训练LoRA、Hippo和门控
        self._setup_trainable_parameters()
        
        print(f"HippoLoRAQwen初始化完成")
        print(f"Hippo模型参数: {sum(p.numel() for p in self.hippo_model.parameters() if p.requires_grad)}")
        print(f"门控参数: {sum(p.numel() for p in self.gate_mechanisms.parameters() if p.requires_grad)}")

    def _apply_peft_lora(self, lora_rank, lora_alpha, lora_dropout):
        """使用PEFT内置函数应用LoRA"""
        # 配置LoRA参数
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj",  # 注意力层
                "gate_proj", "up_proj", "down_proj"      # MLP层
            ],
            layers_to_transform=self.fusion_layers  # 只对指定层应用LoRA
        )
        
        # 应用LoRA到基础模型
        self.base_model = get_peft_model(self.base_model, lora_config)
        print(f"PEFT LoRA已应用到层: {self.fusion_layers}")
        self.base_model = self.base_model.model

    def _initialize_gate_mechanisms(self):
        """专门为门控机制设计的初始化策略"""
        for gate_module in self.gate_mechanisms.values():
            # 第一个线性层
            linear1 = gate_module[0]
            nn.init.xavier_uniform_(linear1.weight, gain=0.1)
            if linear1.bias is not None:
                nn.init.constant_(linear1.bias, -2.0)
            
            # 第二个线性层  
            linear2 = gate_module[2]
            nn.init.xavier_uniform_(linear2.weight, gain=0.1)
            if linear2.bias is not None:
                nn.init.constant_(linear2.bias, -3.0)

    def _setup_trainable_parameters(self):
        """设置可训练参数"""
        # PEFT已经处理了LoRA参数的requires_grad
        # 冻结基础模型权重，只训练LoRA适配器、Hippo和门控
        for name, param in self.base_model.named_parameters():
            if 'lora_' in name:
                # LoRA参数保持可训练
                param.requires_grad = True
            else:
                # 基础模型参数冻结
                param.requires_grad = False
        
        # 确保Hippo和门控参数可训练
        for param in self.hippo_model.parameters():
            param.requires_grad = True
        for param in self.gate_mechanisms.parameters():
            param.requires_grad = True

    def _print_labels_text(self, labels):
        """
        辅助函数：将 labels (token ids) 解码为文本并打印
        """
        # 确保 labels 在 CPU 上（避免设备不匹配问题）
        labels_cpu = labels.cpu()
        batch_size = labels_cpu.shape[0]
        
        print("\n" + "="*50)
        print(f"Labels 对应的文本（共 {batch_size} 个样本）：")
        print(labels[0][:8])
        print("="*50)
        
        for idx in range(batch_size):
            # 获取单个样本的 labels，过滤掉 -100（忽略索引）
            sample_labels = labels_cpu[idx]
            valid_labels = sample_labels[sample_labels != -100]  # 只保留有效 token id
            
            if len(valid_labels) == 0:
                print(f"样本 {idx+1}：无有效 labels（全为 -100 或 padding）")
                continue
            
            # 解码为文本（skip_special_tokens=True 跳过 pad_token、bos_token 等特殊 token）
            text = self.tokenizer.decode(
                valid_labels,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True  # 清理多余空格
            )
            
            # 打印结果（包含样本索引、原始 token id 长度、有效 token id 长度、文本内容）
            print(f"\n样本 {idx+1}：")
            print(f"  - 原始 labels 长度：{len(sample_labels)}")
            print(f"  - 有效 token id 长度：{len(valid_labels)}")
            print(f"  - 文本内容：{text}")
        
        print("="*50 + "\n")
    def forward(self, input_ids=None, attention_mask=None, labels=None, dialog_histories=None):
        """
        前向传播方法
        """
        is_training = self.training
        batch_size, seq_len = input_ids.shape

        # self._print_labels_text(labels)
        if self.hidden_h is None:
            self.hidden_h = self.hippo_model.reset_h(batch_size=batch_size)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=self.base_model.dtype, device=input_ids.device)
        else:
            attention_mask = attention_mask.to(dtype=self.base_model.dtype)

        # 正确访问Qwen3的嵌入层 - 使用model.embed_tokens
        hidden_states = self.base_model.model.embed_tokens(input_ids)
        
        # 位置ID
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).repeat(batch_size, 1)
        
        # 生成旋转位置嵌入
        position_embeddings = self.base_model.model.rotary_emb(hidden_states, position_ids)
        
        # 检查attention_mask是否已经是字典格式（可能是之前已经处理过的）
        # 初始化cache_position（用于缓存机制），按照Qwen3源码方式计算
        cache_position = torch.arange(
            0, seq_len, device=input_ids.device
        )
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
            
        # 处理对话历史
        if dialog_histories is not None:
            h_initial = self.hippo_model.reset_h(batch_size)
            for turn_idx in range(dialog_histories.shape[1]):
                history_batch = dialog_histories[:, turn_idx, :]
                
                # 跳过全为pad的轮次
                if (history_batch == self.tokenizer.pad_token_id).all():
                    continue

                # 计算历史嵌入
                history_embeds = self.base_model.model.embed_tokens(history_batch)
                
                # 更新Hippo隐藏状态
                _, h_initial = self.hippo_model(history_embeds, h_initial, last_n_tokens=self.last_n_tokens)
                self.hidden_h = h_initial

        # 使用Hippo模型处理第一层输入
        hippo_output, self.hidden_h = self.hippo_model(hidden_states, self.hidden_h, last_n_tokens=self.last_n_tokens)
        # print(f"Hippo输出 min: {hippo_output.min().item()}, max: {hippo_output.max().item()}, has nan: {torch.isnan(hippo_output).any().item()}")
        
        # 处理每一层transformer
        cache_position = torch.arange(0, seq_len, device=input_ids.device)
        
        for layer_idx, layer in enumerate(self.base_model.model.layers):
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
            # print(f"hidden_states min: {hidden_states.min().item()}, max: {hippo_output.max().item()}, has nan: {torch.isnan(hippo_output).any().item()}")
            
            # 检查是否在融合层位置
            if layer_idx in self.fusion_layers:
                # 获取门控权重
                gate_mechanism = self.gate_mechanisms[f"layer_{layer_idx}"]
                gate_input = torch.cat([hidden_states, hippo_output], dim=-1)
                gate_weight = gate_mechanism(gate_input)
                
                # 门控加权融合
                hidden_states = gate_weight * hippo_output + (1 - gate_weight) * hidden_states

        # 最终归一化和输出
        hidden_states = self.base_model.model.norm(hidden_states)
        logits = self.base_model.lm_head(hidden_states)
        
        # 计算损失
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

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
            # 只保存Hippo相关组件
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
        
        # 完整模型保存
        base_dir = os.path.join(save_directory, "lora_finetuning")
        os.makedirs(base_dir, exist_ok=True)
        
        # 保存基础模型（包含LoRA）
        self.base_model.save_pretrained(os.path.join(base_dir, "base_model"))
        
        # 保存Hippo组件
        torch.save({
            'hippo_state_dict': self.hippo_model.state_dict(),
            'gate_mechanisms': self.gate_mechanisms.state_dict(),
            'fusion_layers': self.fusion_layers,
        }, os.path.join(base_dir, "hippo_lora_components.pt"))
        
        # 保存配置
        config = {
            'base_model_name_or_path': self.base_model_name_or_path,
            'fusion_layers': self.fusion_layers,
            'model_type': 'hippo_lora_qwen',
        }
        
        with open(os.path.join(base_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"模型已保存到 {base_dir}")

    @classmethod
    def from_pretrained(cls, model_path, fusion_layers=None, **kwargs):
        """从保存的模型加载"""
        import json
        
        # 加载配置
        with open(os.path.join(model_path, "config.json"), 'r') as f:
            config = json.load(f)
        
        # 创建模型实例
        model = cls(
            base_model_name_or_path=os.path.join(model_path, "base_model"),
            fusion_layers=config.get('fusion_layers', fusion_layers),
            **{k: v for k, v in kwargs.items() if k in ['seq_len', 'lora_rank', 'lora_alpha', 'lora_dropout', 'last_n_tokens']}
        )
        
        # 加载Hippo组件
        components = torch.load(os.path.join(model_path, "hippo_lora_components.pt"))
        model.hippo_model.load_state_dict(components['hippo_state_dict'])
        model.gate_mechanisms.load_state_dict(components['gate_mechanisms'])
        
        return model