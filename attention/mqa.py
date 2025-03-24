import torch
from torch import nn
import torch.nn.functional as F

# 大体参照mha，将k_proj和v_proj换成单头
class MultiQuerySelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads==0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim//num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, self.head_dim)
        self.v_proj = nn.Linear(dim, self.head_dim)
        self.o_proj = nn.Linear(dim, dim)
        
        self.scale_ratio = self.head_dim ** -0.5
        
    def forward(self, x, masks=None):
        bs, seq_len, _ = x.size()
        
        q = self.q_proj(x).reshape(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(bs, seq_len, 1, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(bs, seq_len, 1, self.head_dim).transpose(1, 2)
        
        attn_weights = torch.matmul(q, k.transpose(-1,-2)) * self.scale_ratio
        if masks is not None:
            attn_weights = attn_weights.masked_fill(masks==0, -1e-9)
        attn_weights = F.softmax(attn_weights, -1)
        
        output = torch.matmul(attn_weights, v)
        output = output.transpose(-1,-2).contiguous().view(bs, seq_len, self.dim)
        output = self.o_proj(output)
        return output
    
model = MultiQuerySelfAttention(768, 12) # 以gpt2为例
x = torch.randn(2, 64, 768)
masks = torch.tril(torch.ones(64,64)) # 创建下三角矩阵
x = model(x, masks)
print(x.shape) 
        
    