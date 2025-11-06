import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class HippoModel(nn.Module):
    def __init__(self, 
                 input_dim: int = 4096,  # 输入维度（通常为大模型隐藏层维度）
                 hidden_dim: int = 16,   # 隐藏状态维度
                 hippo_type: str = "LegS",  # Hippo矩阵类型
                 ffn_dim: int = 4096*2,     # 前馈网络维度
                 output_dim: int = 4096):  # 输出维度
        """
        Hippo模型的基础实现，基于选择性状态空间模型(SSM)的序列建模
        
        参数:
            input_dim: 输入向量的维度
            hidden_dim: 隐藏状态h的维度
            hippo_type: Hippo矩阵类型，支持'LegT', 'LagT', 'LegS'
            ffn_dim: 前馈网络中间层维度
            output_dim: 最终输出维度
        """
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 核心参数
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hippo_type = hippo_type
        self.ffn_dim = ffn_dim
        self.output_dim = output_dim
        
        # 初始化Hippo矩阵（A）：用于隐藏状态的时序更新
        A_np = self._create_hippo_matrix(hidden_dim, hippo_type)
        self.A = nn.Parameter(torch.tensor(A_np, dtype=torch.float32, device=self.device))
        
        # 基础投影层，用于动态生成SSM参数
        self.b_proj = nn.Linear(input_dim, hidden_dim).to(self.device)
        self.c_proj = nn.Linear(input_dim, hidden_dim * output_dim).to(self.device)
        self.d_proj = nn.Linear(input_dim, output_dim).to(self.device)
        
        # GLU门控机制
        self.b_gate = nn.Linear(input_dim, hidden_dim).to(self.device)
        self.c_gate = nn.Linear(input_dim, hidden_dim * output_dim).to(self.device)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, input_dim)
        ).to(self.device)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(input_dim).to(self.device)
        self.norm2 = nn.LayerNorm(input_dim).to(self.device)
    
    def _create_hippo_matrix(self, n: int, type_: str) -> np.ndarray:
        """
        创建基础的Hippo对角矩阵A，用于SSM计算
        
        参数:
            n: 矩阵维度（与hidden_dim一致）
            type_: 多项式类型
        返回:
            对角向量
        """
        # 基础对角线初始化
        diag = np.zeros(n, dtype=np.float32)
        
        if type_ == 'LegT':
            # Legendre Type T
            for k in range(1, n+1):
                diag[k-1] = -np.pi * (k - 0.5)**2 / 4.0
        elif type_ == 'LegS':
            # Legendre Type S
            for k in range(1, n+1):
                diag[k-1] = -k**2 * np.pi**2 / 4.0
        elif type_ == 'LagT':
            # Laguerre多项式
            for k in range(1, n+1):
                diag[k-1] = -k * 2.0
        else:
            raise ValueError(f"不支持的Hippo矩阵类型: {type_}，可选类型: 'LegT', 'LagT', 'LegS'")
        
        # 确保对角线元素都是负数，保证数值稳定性
        diag = np.minimum(diag, -1e-6)
        
        return diag

    def reset_h(self, batch_size: int) -> torch.Tensor:
        """
        初始化批次隐藏状态
        
        参数:
            batch_size: 批次大小
        返回:
            形状为(batch_size, hidden_dim)的初始隐藏状态
        """
        return torch.zeros(
            batch_size, self.hidden_dim, 
            dtype=torch.float32, 
            device=self.device
        )

    def selective_scan(self, B: torch.Tensor, C: torch.Tensor, A: torch.Tensor, h_initial: torch.Tensor) -> torch.Tensor:
        """
        Mamba基础选择性扫描操作
        
        参数:
            B: 形状为(batch_size, seq_len, hidden_dim)的输入矩阵
            C: 形状为(batch_size, seq_len, hidden_dim, output_dim)的输出矩阵
            A: 形状为(hidden_dim,)的对角向量
            h_initial: 形状为(batch_size, hidden_dim)的初始隐藏状态
        返回:
            形状为(batch_size, seq_len, output_dim)的输出序列
        """
        device = B.device
        
        # 计算指数衰减因子 (hidden_dim,) -> (1, 1, hidden_dim) 用于广播
        delta = torch.exp(A).view(1, 1, hidden_dim)
        
        # 构造时间步索引和累积衰减矩阵，替代循环中的逐步衰减计算
        t = torch.arange(seq_len, device=device).view(1, -1, 1)  # 时间步索引 (1, seq_len, 1)
        decay = delta ** t  # 基础衰减因子 (1, seq_len, hidden_dim)
        # 用三角矩阵掩码实现"仅累加过去时间步"（t >= s）
        decay = decay * torch.triu(torch.ones(seq_len, seq_len, device=device)).view(1, seq_len, seq_len)
        
        # 向量化计算B的贡献：替代循环中h_prev = h_prev * delta + B[:, t]的累积
        B_expanded = B.unsqueeze(1)  # (batch, 1, seq_len, hidden)
        decay_expanded = decay.unsqueeze(-1)  # (1, seq_len, seq_len, 1)
        B_contrib = (B_expanded * decay_expanded).sum(dim=2)  # 并行累加所有过去时间步的贡献
        
        # 向量化计算初始状态的贡献：替代循环中初始状态的逐步衰减
        h_initial_contrib = h_initial.unsqueeze(1) * (delta ** t)  # (batch, seq_len, hidden)
        
        # 总隐藏状态（合并初始状态和B的贡献）
        h = h_initial_contrib + B_contrib
        
        # 用einsum高效计算输出：替代循环中的矩阵乘法
        output = torch.einsum('bth, btho -> bto', h, C)  # (batch, seq_len, output_dim)
        
        return output
    # def selective_scan(self, B: torch.Tensor, C: torch.Tensor, A: torch.Tensor, h_initial: torch.Tensor) -> torch.Tensor:
        
    #     batch_size, seq_len, hidden_dim = B.shape
    #     output_dim = C.shape[-1]  # 获取输出维度
    #     device = B.device
        
    #     # 计算delta值（指数衰减因子）
    #     delta = torch.exp(A)  # 形状为(hidden_dim,)
        
    #     # 初始化隐藏状态
    #     h_prev = h_initial.to(device=device, dtype=B.dtype)
        
    #     # 预分配输出张量
    #     outputs = torch.zeros(batch_size, seq_len, output_dim, device=device, dtype=B.dtype)
        
    #     # 执行选择性扫描
    #     for t in range(seq_len):
    #         # 更新隐藏状态: h_t = h_{t-1} * delta + B_t
    #         h_curr = h_prev * delta + B[:, t]
            
    #         # 2. 计算当前输出：y_t = C_t · h_t（矩阵乘法）
    #         # C[:, t]形状：(batch_size, hidden_dim, output_dim)
    #         # h_curr.unsqueeze(1)形状：(batch_size, 1, hidden_dim)
    #         # 矩阵乘法结果：(batch_size, 1, output_dim)，挤压后为(batch_size, output_dim)
    #         outputs[:, t] = torch.matmul(h_curr.unsqueeze(1), C[:, t]).squeeze(1)
    #         # 准备下一次迭代
    #         h_prev = h_curr
        
    #     return outputs

    def forward(self, 
                batch_vectors: torch.Tensor, 
                h_initial: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mamba模型的前向传播
        
        参数:
            batch_vectors: 形状为(batch_size, seq_len, hidden_size)的张量
        返回:
            批次中每个样本的最终输出，形状为(batch_size, seq_len, output_dim)
        """
        # 输入验证与设备强制对齐，避免设备不匹配错误
        if not isinstance(batch_vectors, torch.Tensor) or batch_vectors.dim() != 3:
            raise ValueError("输入必须是形状为(batch_size, seq_len, input_dim)的张量")
        batch_vectors = batch_vectors.to(self.device)  # 强制输入到模型设备
        batch_size, seq_len, _ = batch_vectors.size()
        
        # 支持流式隐藏状态初始化：允许传入上一轮的隐藏状态
        if h_initial is None:
            h_initial = self.reset_h(batch_size)
        else:
            # 验证隐藏状态形状，增强鲁棒性
            if h_initial.shape != (batch_size, self.hidden_dim):
                raise ValueError(f"h_initial形状应为({batch_size}, {self.hidden_dim})，实际为{h_initial.shape}")
            h_initial = h_initial.to(self.device, dtype=batch_vectors.dtype)  # 设备和类型对齐
        
        # 层归一化
        x = self.norm1(batch_vectors)
        
        # 生成SSM参数（带GLU门控）
        B = self.b_proj(x) * torch.sigmoid(self.b_gate(x))  # (batch, seq_len, hidden_dim)
        C_proj = self.c_proj(x) * torch.sigmoid(self.c_gate(x))  # (batch, seq_len, hidden_dim*output_dim)
        # 将C_proj拆分为矩阵形状，匹配selective_scan的输入要求
        C = C_proj.reshape(batch_size, seq_len, self.hidden_dim, self.output_dim)
        D = self.d_proj(x)  # (batch, seq_len, output_dim)
        
        # 执行选择性扫描（向量化实现）
        ssm_output = self.selective_scan(B, C, self.A, h_initial)
        ssm_output = ssm_output + D  # 结合偏置项
        
        # 残差连接
        x = batch_vectors + ssm_output
        
        # 前馈网络处理
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = x + ffn_out  # 残差连接
        
        # 计算最终隐藏状态（用于流式处理）：避免存储完整h序列，直接公式计算最后一步
        last_h = h_initial * (torch.exp(self.A) ** seq_len) + \
                 torch.sum(B * (torch.exp(self.A) ** (seq_len - 1 - torch.arange(seq_len, device=self.device))).view(1, -1, 1), dim=1)
        
        # 返回输出和最终隐藏状态，支持链式流式调用
        return x, last_h


# 使用示例
if __name__ == "__main__":
    # 初始化模型
    model = HippoModel()
    # 创建示例张量输入：批次大小为2，序列长度为3，输入维度为4096
    batch_vectors = torch.randn(2, 3, 4096, device=model.device)
    # 前向传播
    outputs = model(batch_vectors)
    print("输出形状:", outputs.shape)  # 应输出 torch.Size([2, 3, 4096])
    print("Mamba模型基础功能已成功实现")