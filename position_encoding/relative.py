import torch
from torch import nn
import math
class RelativePositionEncoding(nn.Module):
    def __init__(self, dim, max_len):
        super().__init__()
        pe = torch.zeros(2*max_len-1, dim)
        position = torch.arange(2*max_len-1).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000/dim)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        
    def __call__(self, x):
        bs, num_heads, seq_len, _ = x.size()
        rel_pos = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
        rel_pos += seq_len - 1
        rel_pos_enc = self.pe[rel_pos]
        return rel_pos_enc # (seq_len,seq_len,dim)
    
rpe = RelativePositionEncoding(1024, 512)
x = torch.randn(4, 8, 32, 1024) 
print(rpe(x).shape)