import torch

def kmeans(data, k, max_steps=100, tol=1e-4):
    # 获取数据维度
    n = data.size(0)
    
    # 随机初始化聚类中心（从数据中随机选择k个点）
    idx = torch.randperm(n)[:k]
    centers = data[idx]
    
    # 迭代优化
    for i in range(max_steps):
        # 计算每个点到各个中心的距离
        distances = torch.cdist(data, centers, p=2) # (n,k)
        
        # 为每个点分配最近的中心
        labels = torch.argmin(distances, dim=1)
        
        # 保存旧的中心点用于检查收敛
        prev_centers = centers.clone()
        
        # 更新中心点
        for j in range(k):
            # 找出属于当前簇的所有点
            mask = labels == j
            if mask.sum() > 0:  # 确保簇非空
                centers[j] = data[mask].mean(dim=0)
        
        # 检查是否收敛
        if torch.norm(centers - prev_centers) < tol:
            break
    
    return centers, labels

# 生成一些随机数据
n_samples = 300
data1 = torch.randn(n_samples//3, 2) + torch.tensor([2.0, 2.0])
data2 = torch.randn(n_samples//3, 2) + torch.tensor([-2.0, -2.0])
data3 = torch.randn(n_samples//3, 2) + torch.tensor([2.0, -2.0])
data = torch.cat([data1, data2, data3], dim=0)

# 运行K-means
k = 3
centers, labels = kmeans(data, k)

print("聚类中心:")
print(centers)
print("\n各簇数据点数量:")
for i in range(k):
    print(f"簇 {i}: {(labels == i).sum().item()} 个点")
