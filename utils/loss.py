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

def complex_loss(ecg_raw, ppg_raw, abp_raw, 
                            ecg_pred, ppg_pred, abp_pred, 
                            args,
                            target_weights=None):
    """
    极简版 SOTA 损失函数
    保留三要素：加权绝对误差 + 形状相关性 + 自监督辅助
    """
    
    # ------------------------------------------------------
    # 1. ALMR 辅助任务 (Auxiliary Loss)
    # ------------------------------------------------------
    # 作用：确保输入端的 Tokenizer 学会了生理特征
    loss_almr = torch.tensor(0.0, device=abp_pred.device)
    if (ecg_pred is not None) and (ppg_pred is not None):
        # 维度对齐与截断
        ecg_target = ecg_raw.squeeze(-1) if ecg_raw.dim() == 3 else ecg_raw
        ppg_target = ppg_raw.squeeze(-1) if ppg_raw.dim() == 3 else ppg_raw
        target_len = ecg_pred.shape[1]
        
        # 简单的 MSE 约束特征重构
        loss_almr = F.mse_loss(ecg_pred, ecg_target[:, :target_len]) + \
                    F.mse_loss(ppg_pred, ppg_target[:, :target_len])

    # ------------------------------------------------------
    # 2. 主任务：加权数值误差 (Weighted MSE)
    # ------------------------------------------------------
    if abp_raw.dim() == 3: abp_raw = abp_raw.squeeze(-1)
    
    # 不平衡权重处理
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
        args.lambda_mse * loss_mse +   # 建议 1.0 (基准)
        args.lambda_pcc * loss_pcc +   # 建议 0.5 - 1.0 (形状约束)
        args.lambda_almr * loss_almr   # 建议 0.1 (辅助约束)
    )

    return total_loss