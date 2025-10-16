import numpy as np
import torch
import torch.nn as nn
from scipy.special import legendre, laguerre
from sentence_transformers import SentenceTransformer
from typing import List, Union


class HippoModel(nn.Module):
    def __init__(self, 
                 input_dim: int = 384, 
                 hidden_dim: int = 128, 
                 hippo_type: str = "LegS", 
                 middle_dim: int = 1024, 
                 ffn_dim: int = 512, 
                 output_dim: int = 512,
                 text_encoder_name: str = 'all-MiniLM-L6-v2'):  # 允许自定义文本编码器
        """
        基于Hippo矩阵的序列建模模型，用于处理变长文本序列并生成固定维度输出
        
        参数:
            input_dim: 文本嵌入的维度（与text_encoder输出匹配）
            hidden_dim: 隐藏状态h的维度
            hippo_type: Hippo矩阵类型，支持'LegT', 'LagT', 'LegS'
            middle_dim: 编码器/解码器中间层维度
            ffn_dim: 解码器FFN层维度
            output_dim: 最终输出维度
            text_encoder_name: 文本编码器模型名或本地路径
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 核心参数
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hippo_type = hippo_type
        self.middle_dim = middle_dim
        self.ffn_dim = ffn_dim
        self.output_dim = output_dim
        
        # 初始化Hippo矩阵（A）：用于隐藏状态的时序更新
        A_np = self._create_hippo_matrix(hidden_dim, hippo_type)
        self.A = nn.Parameter(torch.tensor(A_np, dtype=torch.float32, device=self.device))  # 注册为可学习参数

        # 编码器（替代传统Hippo中的B矩阵）：将文本嵌入映射到隐藏状态空间
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, middle_dim),
            nn.GELU(),
            nn.Linear(middle_dim, hidden_dim)
        ).to(self.device)
                
        # 文本编码器：将原始文本转为向量（使用预训练模型）
        self.text_encoder = SentenceTransformer(text_encoder_name).to(self.device)
        
        
        # 解码器：将最终隐藏状态映射到输出维度
        self.decoder = nn.ModuleDict({
            'input_proj': nn.Linear(hidden_dim, middle_dim),  # 隐藏状态投影
            'layer_norm1': nn.LayerNorm(middle_dim),  # 归一化
            'ffn': nn.Sequential(  # 前馈网络
                nn.Linear(middle_dim, ffn_dim),
                nn.GELU(),
                nn.Linear(ffn_dim, output_dim)
            ),
            'layer_norm2': nn.LayerNorm(output_dim),  # 输出归一化
            'output_layer': nn.Linear(output_dim, output_dim)  # 最终映射
        }).to(self.device)
        
    def _create_hippo_matrix(self, n: int, type_: str) -> np.ndarray:
        """
        创建指定类型的Hippo矩阵（用于捕获时序依赖）
        
        参数:
            n: 矩阵维度（与hidden_dim一致）
            type_: 多项式类型，决定矩阵结构
        返回:
            形状为(n, n)的Hippo矩阵（numpy数组）
        """
        A = np.zeros((n, n), dtype=np.float32)  # 预分配内存，指定float32类型
        if type_ == 'LegT':
            # Legendre多项式变换矩阵（在[-1,1]区间采样）
            for i in range(n):
                P = legendre(i)  # 获取i次Legendre多项式
                for j in range(n):
                    A[i, j] = P(j / (n - 1) * 2 - 1)  # 坐标映射到[-1,1]
        elif type_ == 'LagT':
            # Laguerre多项式变换矩阵（在正整数点采样）
            for i in range(n):
                L = laguerre(i)  # 获取i次Laguerre多项式
                for j in range(n):
                    A[i, j] = L(j + 1)  # 正整数输入
        elif type_ == 'LegS':
            # 带权重的Legendre多项式（对称形式）
            for i in range(n):
                P = legendre(i)
                for j in range(n):
                    A[i, j] = P(2 * j / n - 1) * np.sqrt(2 * i + 1)  # 加权归一化
        else:
            raise ValueError(f"不支持的Hippo矩阵类型: {type_}，可选类型: 'LegT', 'LagT', 'LegS'")
        return A

    def encode_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        将文本或文本列表编码为向量
        
        参数:
            text: 单个文本字符串或文本列表
        返回:
            编码后的张量，形状为(seq_len, input_dim)（列表输入）或(input_dim,)（单文本）
        """
        # 统一输入格式为列表（适配SentenceTransformer的encode方法）
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        # 编码并转换为张量（确保与模型在同一设备）
        vecs = self.text_encoder.encode(texts, convert_to_tensor=True, device=self.device)  # (seq_len, input_dim)
        return vecs  # 与Hippo矩阵A保持设备一致

    def reset_h(self, batch_size: int) -> torch.Tensor:
        """
        初始化批次隐藏状态
        
        参数:
            batch_size: 批次大小
        返回:
            形状为(batch_size, hidden_dim)的初始隐藏状态（全0）
        """
        return torch.zeros(
            batch_size, self.hidden_dim, 
            dtype=torch.float32, 
            device=self.device  # 确保与模型设备一致
        )

    def forward(self, batch_seqs: List[List[str]]) -> torch.Tensor:
        """
        处理批次变长文本序列，生成最终输出
        
        参数:
            batch_seqs: 批次数据，形状为(batch_size, )，每个元素是文本列表
                        示例: [["第一步", "第二步"], ["用户输入1", "用户输入2", "用户输入3"], ...]
        返回:
            批次中每个样本的最终输出，形状为(batch_size, output_dim)
        """
        # -------------------------- 输入校验 --------------------------
        if not isinstance(batch_seqs, list) or not all(isinstance(seq, list) for seq in batch_seqs):
            raise ValueError("输入必须是列表的列表，例如: [['text1', 'text2'], ['text3']]")
        batch_size = len(batch_seqs)
        if batch_size == 0:
            raise ValueError("批次不能为空")
        
        # -------------------------- 文本编码与预处理 --------------------------
        # 1. 编码所有序列并记录长度（处理变长序列）
        encoded_seqs = []  # 存储每个样本的编码结果
        seq_lens = []      # 存储每个样本的序列长度
        for seq in batch_seqs:
            encoded = self.encode_text(seq)  # (seq_len, input_dim)
            encoded_seqs.append(encoded)
            seq_lens.append(encoded.shape[0])
        max_seq_len = max(seq_lens)  # 批次内最大序列长度
        
        # 2. 对序列进行padding（统一长度以便批量处理）
        # 形状: (batch_size, max_seq_len, input_dim)
        padded_x = torch.zeros(
            batch_size, max_seq_len, self.input_dim,
            dtype=torch.float32,
            device=self.device
        )
        for i, seq in enumerate(encoded_seqs):
            padded_x[i, :seq_lens[i]] = seq  # 仅填充有效序列部分
        
        # 3. 通过编码器生成Bx（形状: (batch_size, max_seq_len, hidden_dim)）
        seqs_Bx = self.encoder(padded_x)
        
        # -------------------------- 隐藏状态更新（核心优化） --------------------------
        # 初始化隐藏状态: (batch_size, hidden_dim)
        batch_h = self.reset_h(batch_size)
        
        # 向量化更新：按时间步循环（替代原有的样本+时间步双重循环）
        # 对每个时间步t，仅更新序列长度>t的样本
        for t in range(max_seq_len):
            # 掩码：标记哪些样本在当前时间步t有有效数据（形状: (batch_size,)）
            mask = torch.tensor([t < seq_len for seq_len in seq_lens], device=self.device)
            
            # 提取当前时间步的Bx（形状: (batch_size, hidden_dim)）
            current_Bx = seqs_Bx[:, t, :]
            
            # 批量更新隐藏状态：h = A @ h + Bx（仅对有效样本更新）
            # A.unsqueeze(0) 扩展为(1, hidden_dim, hidden_dim)，支持批次矩阵乘法
            updated_h = torch.bmm(self.A.unsqueeze(0).expand(batch_size, -1, -1), batch_h.unsqueeze(-1)).squeeze(-1)
            updated_h = updated_h + current_Bx  # 加上当前时间步的输入
            
            # 用掩码选择是否更新（无效样本保持原状态）
            batch_h = torch.where(mask.unsqueeze(1), updated_h, batch_h)
        
        # -------------------------- 解码最终隐藏状态 --------------------------
        # 1. 投影与归一化
        projected_h = self.decoder['input_proj'](batch_h)  # (batch_size, middle_dim)
        norm_h = self.decoder['layer_norm1'](projected_h)
        
        # 2. FFN与残差连接
        ffn_output = self.decoder['ffn'](norm_h) + projected_h  # 残差连接增强梯度传播
        norm_ffn_output = self.decoder['layer_norm2'](ffn_output)
        
        # 3. 最终输出
        final_output = self.decoder['output_layer'](norm_ffn_output)  # (batch_size, output_dim)
        
        return final_output


# 使用示例（取消注释可运行）
if __name__ == "__main__":
    # 初始化模型
    model = HippoModel()
    # 示例输入：2个样本，分别包含2个和3个文本
    batch_seqs = [
        ["这是第一个样本的第一句话", "这是第一个样本的第二句话"],
        ["第二个样本的第一句", "第二句", "第三句"]
    ]
    # 前向传播
    outputs = model(batch_seqs)
    print("输出形状:", outputs.shape)  # 应输出 torch.Size([2, 512])