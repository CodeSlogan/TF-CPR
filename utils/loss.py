import torch
import torch.nn as nn
import torch.nn.functional as F

class NegPearsonLoss(nn.Module):
    """
    Pearson Loss: 专注波形形状，忽略幅值和偏移
    """
    def __init__(self):
        super(NegPearsonLoss, self).__init__()

    def forward(self, preds, targets):
        # Flatten: [B, L] -> [B, -1]
        preds_flat = preds.view(preds.shape[0], -1)
        targets_flat = targets.view(targets.shape[0], -1)
        
        # Centering (减均值)
        vx = preds_flat - torch.mean(preds_flat, dim=1, keepdim=True)
        vy = targets_flat - torch.mean(targets_flat, dim=1, keepdim=True)
        
        # Correlation calculation
        cost = torch.sum(vx * vy, dim=1) / (torch.sqrt(torch.sum(vx ** 2, dim=1)) * torch.sqrt(torch.sum(vy ** 2, dim=1)) + 1e-8)
        
        # 1 - PCC (使得 Loss 最小化)
        return 1 - torch.mean(cost)

_pearson_loss_fn = NegPearsonLoss()

def complex_loss(abp_raw, abp_pred, args, target_weights=None):
    """
    极简版 SOTA 损失函数
    保留三要素：加权绝对误差 + 形状相关性 + 自监督辅助
    """
    
    # ------------------------------------------------------
    # 2. 主任务：加权数值误差 (Weighted MSE)
    # ------------------------------------------------------
    if abp_raw.dim() == 3: abp_raw = abp_raw.squeeze(-1)
    
    if target_weights is None:
        target_weights = torch.ones_like(abp_raw[:, 0]).unsqueeze(1) # [B, 1]
    
    if target_weights.dim() == 1:
        target_weights = target_weights.unsqueeze(1)

    mse_per_sample = F.mse_loss(abp_pred, abp_raw, reduction='none').mean(dim=1) # [B]
    loss_mse = (mse_per_sample * target_weights.squeeze()).mean()

    # ------------------------------------------------------
    # 3. 主任务：形态相关性 (Pearson Loss)
    # ------------------------------------------------------
    loss_pcc = _pearson_loss_fn(abp_pred, abp_raw)

    total_loss = (
        args.lambda_mse * loss_mse +   
        args.lambda_pcc * loss_pcc 
    )

    return total_loss