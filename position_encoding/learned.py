import torch
from torch import nn

class LearnedPositionEncoding(nn.Module):
    def __init__(self, dim=1024, max_len=512):
        super().__init__()
        self.pe = nn.Parameter(
            torch.zeros(max_len, dim)
        )
        nn.init.normal_(self.pe, mean=0, std=0.02)
        
    def __call__(self, x):
        bs, num_heads, seq_len, _ = x.size()
        return x + self.pe[:seq_len, :]
    
lpe = LearnedPositionEncoding()
x = torch.randn(4, 8, 32, 1024) 
print(lpe(x).shape)
