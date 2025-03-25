import torch
import torch.nn as nn
import math

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim=1024, max_len=512):
        super().__init__()
        
        # 生成旋转角度的频率
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        # 生成位置索引
        positions = torch.arange(max_len).float()
        # 计算每个位置的旋转角度
        freqs = torch.outer(positions, freqs)  # [seq_len, dim/2] outer=>外积
        
        # 预计算旋转角度的cos和sin值
        cos = freqs.cos()  # [seq_len, dim/2]
        sin = freqs.sin()  # [seq_len, dim/2]
        
        # 将cos和sin扩展到与输入维度匹配
        cos = torch.repeat_interleave(cos, 2, dim=1)  # [seq_len, dim]
        sin = torch.repeat_interleave(sin, 2, dim=1)  # [seq_len, dim]
        
        # 将cos和sin值保存为缓冲区，这样它们不会被视为模型参数
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)
        
    def __call__(self, x):
        bs, num_heads, seq_len, _ = x.size()
        cos = self.cos[:seq_len].unsqueeze(0).unsqueeze(1) # [1, 1, seq_len, dim]
        sin = self.sin[:seq_len].unsqueeze(0).unsqueeze(1) # [1, 1, seq_len, dim]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        cos_even = cos[..., 0::2]
        sin_odd = sin[..., 1::2]
        x_rotated = torch.zeros_like(x)
        x_rotated[..., 0::2] = x_even * cos_even - x_odd * sin_odd
        x_rotated[..., 1::2] = x_even * sin_odd + x_odd * cos_even
        return x_rotated
    
ape = RotaryPositionalEmbedding(1024, 512)
x = torch.randn(4, 8, 512, 1024) 
print(ape(x).shape)
        