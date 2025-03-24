import torch
import torch.nn as nn
import torch.nn.functional as F

# LayerNorm简洁实现
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        
    def forward(self, x):
        # 计算最后几个维度的均值和方差
        ndim = len(x.shape)
        ndim_affine = len(self.normalized_shape)
        
        # 确定在哪些维度上计算统计量
        dims = tuple(range(ndim - ndim_affine, ndim))
        
        # 计算均值和方差
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, unbiased=False, keepdim=True)
        
        # 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        return x_norm