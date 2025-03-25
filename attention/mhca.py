import torch
from torch import nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim%num_heads==0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim//num_heads # 每个头的维度，forward时才会用到
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.scale_ratio = self.head_dim**-0.5 # 使得训练更稳定，防止注意力分数过大
        
    def forward(self, q, k, v, masks=None):
        # 先获取当前输入的结构信息
        bs, seq_len, _ = q.size() 
        
        # 计算qkv矩阵=>拆成多个头=>转置成 (bs,head,seq_len,head_dim) 的形式，用于计算每个头token之间的注意力分数
        q = self.q_proj(q).reshape(bs, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        k = self.k_proj(k).reshape(bs, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        v = self.v_proj(v).reshape(bs, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        
        # 计算注意力分数
        attn_weights = torch.matmul(q, k.transpose(-1,-2)) * self.scale_ratio
        if masks is not None:
            attn_weights = attn_weights.masked_fill(masks==0, -1e-9)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # 计算输出向量
        output = torch.matmul(attn_weights, v)
        output = output.transpose(-1,-2).contiguous().view(bs, seq_len, self.dim) # 转为(bs,num_heads,head_dim,seq_len)=>contiguous保证内存中连续=>转换回 (bs,seq_len,head_dim)
        output = self.o_proj(output)
        
        return output


model = CrossAttention(768, 12) # 以gpt2为例
q = torch.randn(2, 64, 768)
k = torch.randn(2, 64, 768)
v = torch.randn(2, 64, 768)
masks = torch.tril(torch.ones(64,64)) # 创建下三角矩阵
x = model(q, k, v, masks)
print(x.shape) 
               