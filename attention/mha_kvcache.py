import torch
from torch import nn
import torch.nn.functional as F

# 基于mha修改，增加了kvcache实现
class MultiHeadSelfAttentionKVCache(nn.Module):
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
        
    def forward(self, x, masks=None, use_cache=False, past_key_value=None):
        # 先获取当前输入的结构信息
        bs, seq_len, _ = x.size() 
        
        # 计算qkv矩阵=>拆成多个头=>转置成 (bs,head,seq_len,head_dim) 的形式，用于计算每个头token之间的注意力分数
        q = self.q_proj(x).reshape(bs, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        k = self.k_proj(x).reshape(bs, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        v = self.v_proj(x).reshape(bs, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        
        # 拼接过去的kvcache得到当前kv
        if past_key_value is not None:
            past_key, past_value = past_key_value
            k = torch.cat([past_key, k], dim=2) # 在seq_len层拼接key
            v = torch.cat([past_value, v], dim=2) # 在seq_len层拼接value    
        
        # 保存当前kvcache
        present_key_value = (k, v)
        
        # 计算注意力分数
        attn_weights = torch.matmul(q, k.transpose(-1,-2)) * self.scale_ratio
        if masks is not None:
            attn_weights = attn_weights.masked_fill(masks==0, -1e-9)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # 计算输出向量
        kv_seq_len = k.size(-2)
        output = torch.matmul(attn_weights, v)
        output = output.transpose(-1,-2).contiguous().view(bs, kv_seq_len, self.dim) # 转为(bs,num_heads,head_dim,seq_len)=>contiguous保证内存中连续=>转换回 (bs,seq_len,head_dim)
        output = self.o_proj(output)
        
        if use_cache:
            return output, present_key_value
        
        return output


model = MultiHeadSelfAttentionKVCache(768, 12) # 以gpt2为例

# time 0
x = torch.randn(2, 5, 768)
masks = torch.tril(torch.ones(5,5))
x, past_key_value = model(x, masks, use_cache=True, past_key_value=None)
print(x.shape) 

# time 1
x = torch.randn(2, 1, 768)
masks = torch.tril(torch.ones(6,6))
x, past_key_value = model(x, masks, use_cache=True, past_key_value=past_key_value)
print(x.shape) 

# time 2
x = torch.randn(2, 1, 768)
masks = torch.tril(torch.ones(7,7))
x, past_key_value = model(x, masks, use_cache=True, past_key_value=past_key_value)
print(x.shape) 
