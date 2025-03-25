import torch
import math

class AbsolutePositionEncoding(torch.nn.Module):
    def __init__(self, dim=1024, max_len=512):
        super().__init__()
        assert max_len%2==0
        pe = torch.zeros(max_len, dim) # 初始化位置编码矩阵
        position = torch.arange(max_len).reshape(max_len, 1) # (max_len, 1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0)/dim)) # (dim//2) 随着维度增加，div_term从1递减至0
        pe[:, 0::2] = torch.sin(position * div_term) # (max_len, dim//2)
        pe[:, 1::2] = torch.cos(position * div_term) # (max_len, dim//2)
        self.register_buffer('pe', pe)
        
    def __call__(self, x):
        bs, num_heads, seq_len, _ = x.size()
        return x + self.pe[:seq_len, :]
    
ape = AbsolutePositionEncoding(1024, 512)
x = torch.randn(4, 8, 32, 1024) 
print(ape(x).shape)
