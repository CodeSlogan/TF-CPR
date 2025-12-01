import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d

class LDSWeightCalculator(nn.Module):
    def __init__(self, all_train_labels, bucket_size=2.0, smoothing_sigma=2.0):
        """
        LDS (Label Distribution Smoothing) 权重计算器
        
        参数:
        ----------
        all_train_labels : np.array or List
            整个训练集中所有的标签值 (例如所有的 SBP 值)。
            注意：必须是【反归一化】后的真实血压值 (mmHg)，或者是全局归一化的值，只要统一即可。
            
        bucket_size : float
            直方图分桶的大小 (默认 2.0 mmHg)。
            
        smoothing_sigma : float
            高斯平滑核的大小 (默认 2.0)。
        """
        super().__init__()
        
        # 1. 预处理数据
        labels = np.array(all_train_labels).flatten()
        min_val, max_val = labels.min(), labels.max()
        
        # 2. 创建分桶边界
        # 例如: [50, 52, 54, ..., 200]
        bins = np.arange(min_val, max_val + bucket_size, bucket_size)
        
        # 3. 统计原始直方图
        hist, _ = np.histogram(labels, bins=bins)
        
        # 4. LDS 核心：高斯平滑
        # 这一步解决了 "140样本多但141样本少" 的邻域相关性问题
        smoothed_hist = gaussian_filter1d(hist, sigma=smoothing_sigma)
        
        # 5. 计算倒数权重
        # 加一个极小值 1e-6 防止除以零
        weights = 1.0 / (smoothed_hist + 1e-6)
        
        # 6. 权重归一化 (关键)
        # 让权重的均值为 1，这样不会改变 Loss 的整体量级，只改变样本间的相对重要性
        weights = weights / weights.mean()
        
        # 7. 注册为 Buffer (保存到 GPU，但不作为模型参数更新)
        # bucket_boundaries: 用于查找当前样本属于哪个桶
        # class_weights: 桶对应的权重表
        self.register_buffer('bucket_boundaries', torch.tensor(bins[:-1]).float())
        self.register_buffer('class_weights', torch.tensor(weights).float())
        
        print(f"LDS Weights Computed: Min Weight={weights.min():.2f}, Max Weight={weights.max():.2f}")

    def get_weights(self, batch_targets):
        """
        根据当前 Batch 的标签值，查表返回对应的权重
        输入: batch_targets [B, ...]
        输出: weights [B, 1]
        """
        # 确保输入是 tensor
        if not isinstance(batch_targets, torch.Tensor):
            batch_targets = torch.tensor(batch_targets)
            
        # 1. 查找桶索引 (Bucketize)
        # 找到每个血压值落在哪个区间里
        device = self.bucket_boundaries.device
        batch_targets = batch_targets.to(device)
        
        # subtract 1 because bucketize returns 1-based index for right boundaries
        indices = torch.bucketize(batch_targets, self.bucket_boundaries) - 1
        
        # 2. 边界截断 (防止越界)
        indices = torch.clamp(indices, 0, len(self.class_weights) - 1)
        
        # 3. 查表得到权重
        weights = self.class_weights[indices]
        
        # 确保维度是 [B, 1] 以便后续广播
        if weights.dim() == 1:
            weights = weights.unsqueeze(1)
            
        return weights