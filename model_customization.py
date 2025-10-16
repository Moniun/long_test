import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from hippo_model import HippoModel

class ModifiedQwen(nn.Module):
    """包装Qwen3-8B并插入自定义模块，支持对话历史记忆融合"""
    def __init__(self, base_model_name_or_path):
        super().__init__()
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name_or_path,
            trust_remote_code=True
        )
        
        # 加载基础模型（4-bit量化）
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        custom_cache_dir = "qwen3-8b-custom-module-training"  # 替换为你的目标路径
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=custom_cache_dir  # 关键参数：指定自定义目录
        )
        
        # 获取模型配置
        self.config = self.base_model.config
        self.hidden_size = self.config.hidden_size
        
        # 初始化Hippo模型，用于处理对话历史
        self.hippo_model = HippoModel(output_dim=self.hidden_size)
        
        # 定位目标MLP层（最后一个Transformer层的MLP）
        self.target_mlp_layer = self.base_model.transformer.h[-1]
        
        # 冻结基础模型参数
        self.freeze_base_model()
        # 解冻需要训练的层
        self.unfreeze_trainable_layers()

    def freeze_base_model(self):
        """冻结除目标MLP外的所有基础模型参数"""
        for name, param in self.base_model.named_parameters():
            # 只保留最后一层的MLP参数可训练
            if not name.startswith("transformer.h.-1.mlp"):
                param.requires_grad = False

    def unfreeze_trainable_layers(self):
        """确保Hippo模型和目标MLP层可训练"""
        # Hippo模型参数
        for param in self.hippo_model.parameters():
            param.requires_grad = True
        # 目标MLP层参数
        for param in self.target_mlp_layer.mlp.parameters():
            param.requires_grad = True

    def forward(self, input_ids=None, attention_mask=None, labels=None, dialog_histories=None):
        """
        前向传播方法，实现对话历史记忆融合逻辑
        
        参数:
            input_ids: 大模型输入ID
            attention_mask: 注意力掩码
            labels: 标签（用于损失计算）
            dialog_histories: 对话历史列表，格式为[["句子1", "句子2"], ["句子3", "句子4", "句子5"], ...]
        """
        # 第一步：处理对话历史，通过HippoModel获取记忆表示
        # 获取大模型的词嵌入层
        embedding_layer = self.base_model.transformer.wte
        
        # 处理每个样本的对话历史
        batch_embedded_histories = []
        for sample_history in dialog_histories:
            if not sample_history:  # 空对话历史
                # 创建零向量作为占位符
                embedded_dialog = torch.zeros(1, 1, self.hidden_size, 
                                            device=self.base_model.device)
                batch_embedded_histories.append(embedded_dialog)
            else:
                # 对每个句子应用embedding
                embedded_sentences = []
                for sentence in sample_history:
                    # 使用大模型的分词器处理句子
                    tokens = self.tokenizer(sentence, return_tensors="pt", 
                                            truncation=True, max_length=512)
                    sent_input_ids = tokens.input_ids.to(self.base_model.device)
                    
                    # 使用大模型的embedding层
                    embedded = embedding_layer(sent_input_ids)
                    embedded_sentences.append(embedded)
                
                # 将同一样本的所有句子嵌入拼接起来
                if embedded_sentences:
                    # 形状: (num_sentences, seq_len, hidden_size)
                    embedded_dialog = torch.cat(embedded_sentences, dim=0)
                    # 添加样本维度: (1, num_sentences, seq_len, hidden_size)
                    embedded_dialog = embedded_dialog.unsqueeze(0)
                    batch_embedded_histories.append(embedded_dialog)
        
        # 将对话历史输入HippoModel，得到记忆表示
        # 输入形状: (batch_size, num_sentences, seq_len, hidden_size)
        memory_representations = self.hippo_model(batch_embedded_histories)
        # memory_representations形状: (batch_size, hidden_size)
        
        # 第二步：将memory_query输入大模型，获取最后一层Transformer的输出
        # 获取基础模型输出（包含隐藏状态）
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        
        # 获取最后一层Transformer的输出（MLP的原始输入）
        last_hidden = outputs.hidden_states[-1]  # 形状: (batch_size, seq_len, hidden_size)
        
        # 第三步：在最后一个MLP层之前，融合HippoModel的输出
        # 将记忆表示扩展为序列长度，以便与每个位置的隐藏状态相加
        # (batch_size, hidden_size) -> (batch_size, seq_len, hidden_size)
        expanded_memory = memory_representations.unsqueeze(1).expand(-1, last_hidden.shape[1], -1)
        
        # 应用LayerNorm（使用模型现有的LayerNorm）
        ln_output = self.target_mlp_layer.ln_2(last_hidden)
        
        # 融合特征：将HippoModel的输出与MLP层的输入相加
        fused_input = ln_output + expanded_memory
        
        # 第四步：将融合后的输入送入MLP层
        mlp_output = self.target_mlp_layer.mlp(fused_input)
        
        # 应用残差连接
        final_output = last_hidden + mlp_output
        
        # 计算logits
        logits = self.base_model.lm_head(final_output)
        
        # 构造输出对象
        return type(outputs)(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

    def generate(self, *args, **kwargs):
        """包装生成函数，保持原始接口"""
        return self.base_model.generate(*args, **kwargs)
    