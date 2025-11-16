import torch
import numpy as np
import torch.nn as nn
from typing import Optional, Tuple

class HippoModel(nn.Module):
    def __init__(self, 
                 input_dim: int = 4096,  # 输入维度
                 output_dim: int = 4096,  # 输出维度
                 hidden_dim: int = 16,   # 隐藏状态维度
                 seq_len: int = 1024,    # 序列长度
                 ffn_dim: int = 16,
                 hippo_type: str = "LegS",  # Hippo矩阵类型
                 hippo_scale: float = 1.0,
                 dtype: torch.dtype = torch.float16):  # 新增：接收精度类型参数
        """
        基于矩阵的Hippo模型实现，支持与基座模型统一精度
        
        参数:
            input_dim: 输入向量的维度
            output_dim: 输出向量的维度
            hidden_dim: 隐藏状态维度
            seq_len: 序列长度
            ffn_dim: 前馈网络中间维度
            hippo_type: Hippo矩阵类型
            hippo_scale: Hippo矩阵缩放因子
            dtype: 模型计算精度类型（与基座模型保持一致）
        """
        super().__init__()
        
        # 核心参数
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.hippo_type = hippo_type
        self.hippo_scale = hippo_scale
        self.dtype = dtype  # 保存精度类型
        
        # 初始化Hippo矩阵A：(hidden_dim, hidden_dim)
        A_np = self._create_hippo_matrix(hidden_dim, hippo_type)
        A_np = A_np * self.hippo_scale
        self.register_buffer('A', torch.tensor(A_np, dtype=dtype))  # 使用动态精度
        
        # 可学习矩阵参数 - 使用指定精度
        self.B = nn.Parameter(torch.randn(hidden_dim, seq_len, dtype=dtype) * 0.01)
        self.C = nn.Parameter(torch.randn(seq_len, hidden_dim, dtype=dtype) * 0.01)
        
        # D: 线性层序列 - 使用指定精度
        self.D = nn.Sequential(
            nn.Linear(seq_len, 16, dtype=dtype),
            nn.Linear(16, seq_len, dtype=dtype),
            nn.Sigmoid()
        )
        
        # 前馈网络 - 使用指定精度
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ffn_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(ffn_dim, output_dim, dtype=dtype)
        )
        
        # 层归一化 - 使用指定精度
        self.norm1 = nn.LayerNorm(input_dim, dtype=dtype)
        self.norm2 = nn.LayerNorm(input_dim, dtype=dtype)
        
        # 初始化参数
        self._initialize_weights()
        
    def _init_weights(self, module):
        """统一权重初始化方法"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def _initialize_weights(self):
        """自定义权重初始化"""
        self.D.apply(self._init_weights)
        self.ffn.apply(self._init_weights)
        nn.init.xavier_normal_(self.B.data)
        nn.init.xavier_normal_(self.C.data)
    
    def _create_hippo_matrix(self, n: int, type_: str) -> np.ndarray:
        """创建符合SSM稳定性要求的Hippo对角矩阵A"""
        diag = np.zeros(n, dtype=np.float32)
        
        if type_ == 'LegT':
            for k in range(1, n+1):
                raw_val = -np.pi * (k - 0.5)**2 / 4.0
                diag[k-1] = raw_val
        elif type_ == 'LegS':
            for k in range(1, n+1):
                raw_val = -k**2 * np.pi**2 / 4.0
                diag[k-1] = raw_val
        elif type_ == 'LagT':
            for k in range(1, n+1):
                raw_val = -k * 2.0
                diag[k-1] = raw_val
        else:
            raise ValueError(f"不支持的Hippo矩阵类型: {type_}")
        
        max_abs = np.max(np.abs(diag))
        if max_abs == 0:
            max_abs = 1e-6
        scale = 0.9 / max_abs
        diag = diag * scale
        diag = np.minimum(diag, -1e-6)
        return np.diag(diag)

    def reset_h(self, batch_size: int) -> torch.Tensor:
        """初始化批次隐藏状态（使用模型指定精度）"""
        device = next(self.parameters()).device
        return torch.randn(batch_size, self.hidden_dim, self.input_dim, 
                          device=device, dtype=self.dtype) * 0.01  # 动态精度

    def forward(self, 
                batch_vectors: torch.Tensor,  # (batch_size, seq_len, input_dim)
                h_initial: Optional[torch.Tensor] = None,
                last_n_tokens: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        基于矩阵计算的forward过程（所有计算使用统一精度）
        """
        device = next(self.parameters()).device
        
        # 输入验证
        if not isinstance(batch_vectors, torch.Tensor) or batch_vectors.dim() != 3:
            raise ValueError("输入必须是形状为(batch_size, seq_len, input_dim)的张量")
        
        batch_vectors = batch_vectors.to(device, dtype=self.dtype)  # 转换为模型精度
        batch_size, seq_len, input_dim = batch_vectors.size()
        
        # 支持选择性token处理
        if last_n_tokens > 0 and last_n_tokens < seq_len:
            batch_vectors = batch_vectors[:, -last_n_tokens:, :]
            seq_len = last_n_tokens
        
        # 隐藏状态初始化（确保精度一致）
        if h_initial is None:
            h_initial = self.reset_h(batch_size)
        else:
            if h_initial.shape != (batch_size, self.hidden_dim, input_dim):
                raise ValueError(f"h_initial形状应为({batch_size}, {self.hidden_dim}, {input_dim})，实际为{h_initial.shape}")
            h_initial = h_initial.to(device, dtype=self.dtype)  # 统一精度
        
        # 层归一化（使用模型精度）
        x = self.norm1(batch_vectors)  # 已在初始化时指定dtype，无需额外转换
        
        # === 矩阵计算过程（统一使用模型精度） ===
        A = self.A  # 已注册为指定dtype
        B = self.B
        C = self.C
        
        # h = A*h + B*x
        hA = torch.matmul(A, h_initial)
        Bx = torch.matmul(B, x)
        h_new = hA + Bx
        
        # y = C*h + D*x
        Ch = torch.matmul(C, h_new)
        
        # D*x处理
        x_perm = x.permute(0, 2, 1)
        x_flat = x_perm.reshape(-1, seq_len)
        Dx_flat = self.D(x_flat)
        Dx = Dx_flat.reshape(batch_size, input_dim, seq_len)
        Dx = Dx.permute(0, 2, 1)
        
        y = Ch + Dx
        
        # === 前馈网络处理 ===
        x = self.norm2(y)
        ffn_out = self.ffn(x)
        x = x + ffn_out  # 残差连接
        
        return ffn_out, h_new