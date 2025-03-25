import torch
import torch.nn as nn
import torch.nn.functional as F

# LayerNorm简洁实现
class LayerNorm(nn.Module):
    def __init__(self, dim=768, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gemma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x):
        # 计算均值和方差
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        
        # 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        return x_norm * self.gemma + self.beta
    
ln = LayerNorm()
x = torch.randn(4, 64, 8, 768) # (bs,seq_len,num_heads,dim)
print(ln(x).shape)
