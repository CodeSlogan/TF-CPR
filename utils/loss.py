import torch
import torch.nn as nn
import torch.nn.functional as F


class NegPearsonLoss(nn.Module):
    def __init__(self):
        super(NegPearsonLoss, self).__init__()

    def forward(self, preds, targets):

        preds_flat = preds.view(preds.shape[0], -1)
        targets_flat = targets.view(targets.shape[0], -1)
        
        vx = preds_flat - torch.mean(preds_flat, dim=1, keepdim=True)
        vy = targets_flat - torch.mean(targets_flat, dim=1, keepdim=True)
        
        cost = torch.sum(vx * vy, dim=1) / (torch.sqrt(torch.sum(vx ** 2, dim=1)) * torch.sqrt(torch.sum(vy ** 2, dim=1)) + 1e-8)
        
        return 1 - torch.mean(cost)

_pearson_loss_fn = NegPearsonLoss()

def complex_loss(ecg_raw, ppg_raw, abp_global_norm, 
                 ecg_pred, ppg_pred, pred_final, pred_shape, pred_stats, 
                 args):
    """
    增强版复合损失函数：加入 PCC 约束与 Log-FFT
    """
    # ---  ALMR 辅助任务 ---
    loss_almr = torch.tensor(0.0, device=pred_final.device)
    
    if (ecg_pred is not None) and (ppg_pred is not None):
        # 确保 target 维度正确
        ecg_target = ecg_raw.squeeze(-1) if ecg_raw.dim() == 3 else ecg_raw
        ppg_target = ppg_raw.squeeze(-1) if ppg_raw.dim() == 3 else ppg_raw
        
        # 简单的长度截断 (假设 pred 长度 <= target 长度)
        target_len = ecg_pred.shape[1]
        loss_almr_ecg = F.mse_loss(ecg_pred, ecg_target[:, :target_len])
        loss_almr_ppg = F.mse_loss(ppg_pred, ppg_target[:, :target_len])
        loss_almr = loss_almr_ecg + loss_almr_ppg

    if abp_global_norm.dim() == 3: abp_global_norm = abp_global_norm.squeeze(-1)
    
    # ------------------------------------------------------
    # 1. 动态计算 Instance 统计量 (On-the-fly)
    # ------------------------------------------------------
    # 注意：这里的 gt_mean 是 "全局归一化数据" 的均值
    # 物理含义：(这个人的平均血压 - Global_Median) / Global_IQR
    gt_mean = abp_global_norm.mean(dim=1, keepdim=True) 
    gt_std  = abp_global_norm.std(dim=1, keepdim=True) + 1e-6
    
    gt_shape = (abp_global_norm - gt_mean) / gt_std
    
    # A. 最终重构 Loss (整体对齐)
    loss_total_mse = F.mse_loss(pred_final, abp_global_norm)
    loss_total_pcc = _pearson_loss_fn(pred_final, abp_global_norm)
    
    # B. 统计量 Loss (让 Scalar Head 学习相对偏移)
    target_stats = torch.cat([gt_mean, gt_std], dim=1) # (B, 2)
    loss_stats = F.mse_loss(pred_stats, target_stats)
    
    # C. 形状 Loss (让 Decoder 学习纯波形)
    loss_shape = F.mse_loss(pred_shape, gt_shape)

    total_loss = (
        args.lambda_mse * loss_total_mse +
        args.lambda_pcc * loss_total_pcc +
        args.lambda_scalar * loss_stats +
        args.lambda_deriv * loss_shape +
        args.lambda_almr * loss_almr
    )
    

    return total_loss