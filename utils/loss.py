import torch
import torch.nn.functional as F
from tqdm import tqdm


def complex_loss(ecg_raw, ppg_raw, abp_raw, 
                           ecg_pred, ppg_pred, abp_pred, 
                           lambdas: dict):
    """
    计算基于时频双域与ALMR自监督的复合损失函数
    """
    # 维度对齐: [B, L, 1] -> [B, L]
    if abp_raw.dim() == 3: abp_raw = abp_raw.squeeze(-1)
    if abp_pred.dim() == 3: abp_pred = abp_pred.squeeze(-1)
    
    # 1.1 数值精度 (MSE)
    loss_bp_mse = F.mse_loss(abp_pred, abp_raw)
    
    # 1.2 波形形态 (一阶差分 L1)
    diff_pred = abp_pred[:, 1:] - abp_pred[:, :-1]
    diff_raw = abp_raw[:, 1:] - abp_raw[:, :-1]
    loss_bp_deriv = F.l1_loss(diff_pred, diff_raw)
    
    # 1.3 频域一致性
    fft_pred = torch.fft.rfft(abp_pred, dim=-1)
    fft_raw = torch.fft.rfft(abp_raw, dim=-1)
    loss_bp_freq = F.mse_loss(fft_pred.abs(), fft_raw.abs())
    
    # --- ALMR 辅助任务 ---
    loss_almr = torch.tensor(0.0, device=abp_pred.device)
    
    if (ecg_pred is not None) and (ppg_pred is not None) and (lambdas['almr'] > 0):
        # 仅当权重 > 0 且有输出时计算
        ecg_target = ecg_raw.squeeze(-1) if ecg_raw.dim() == 3 else ecg_raw
        ppg_target = ppg_raw.squeeze(-1) if ppg_raw.dim() == 3 else ppg_raw
        
        B, N, P = ecg_pred.shape
        target_len = N * P
        
        # 长度适配
        if ecg_target.shape[1] > target_len:
            ecg_target = ecg_target[:, :target_len]
            ppg_target = ppg_target[:, :target_len]
        elif ecg_target.shape[1] < target_len:
            pad_len = target_len - ecg_target.shape[1]
            ecg_target = F.pad(ecg_target, (0, pad_len))
            ppg_target = F.pad(ppg_target, (0, pad_len))
            
        ecg_target_reshaped = ecg_target.view(B, N, P)
        ppg_target_reshaped = ppg_target.view(B, N, P)
        
        loss_almr_ecg = F.mse_loss(ecg_pred, ecg_target_reshaped)
        loss_almr_ppg = F.mse_loss(ppg_pred, ppg_target_reshaped)
        loss_almr = loss_almr_ecg + loss_almr_ppg

    # --- 总损失聚合 ---
    total_loss = (lambdas['bp_mse'] * loss_bp_mse +
                  lambdas['bp_deriv'] * loss_bp_deriv + 
                  lambdas['bp_freq'] * loss_bp_freq +
                  lambdas['almr'] * loss_almr)
    
    loss_dict = {
        "total": total_loss.item(),
        "mse": loss_bp_mse.item(),
        "deriv": loss_bp_deriv.item(),
        "freq": loss_bp_freq.item(),
        "almr": loss_almr.item()
    }
    
    return total_loss, loss_dict