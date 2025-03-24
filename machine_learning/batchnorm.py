import torch
import torch.nn as nn
import torch.nn.functional as F

# 手动实现BatchNorm
class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # 运行时统计量
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
    def forward(self, x):
        # x: [b, c, h, w]
        if self.training:
            # 在batch维度上计算均值和方差 (对每个通道)
            batch_mean = x.mean(dim=(0, 2, 3))  # [c]
            batch_var = x.var(dim=(0, 2, 3), unbiased=False)  # [c]
            
            # 更新运行时统计量
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            # 使用当前batch的统计量进行归一化
            mean, var = batch_mean, batch_var
        else:
            # 测试时使用运行时统计量
            mean, var = self.running_mean, self.running_var
            
        # 归一化
        x_normalized = (x - mean.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + self.eps)
        # 缩放和平移
        return x_normalized

