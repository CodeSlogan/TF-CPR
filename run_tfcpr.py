import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import sys
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt

from utils.data_factory import DataFactory, DataConfig, denormalize_abp
from utils.loss import complex_loss
from utils.earlystop import EarlyStopping
from utils.compute_sbp_dbp import calculate_sbp_dbp_from_abp_waveform
from utils.ldsweight import LDSWeightCalculator
from model.tfcpr.TFCPR import TFCPR


def get_args():
    parser = argparse.ArgumentParser(description='TFCPRNet')

    # --- 1. Mode Selection (New) ---
    parser.add_argument('--model_name', type=str, default="tfcpr", help='Model name for saving checkpoints')
    parser.add_argument('--is_train', type=bool, default=True, help='train or test')
    parser.add_argument('--test_checkpoint', type=str, default="./checkpoints/tfcpr_20251130_220345_ep2_mae12.6049.pth", help='Path to checkpoint for testing mode')

    # --- 2.1 Dataset Parameters ---
    parser.add_argument('--data_root', type=str, default='./dataset/mimicbp', help='Data root directory')
    parser.add_argument('--original_length', type=int, default=3750, help='Raw NPY file row length')
    parser.add_argument('--seq_len', type=int, default=625, help='Input window size (slicing length)')
    parser.add_argument("--pred_len", type=int, default=625, help="prediction sequence length")
    parser.add_argument('--stride', type=int, default=625, help='Sliding window stride')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Train set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation set ratio')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')

    # --- 2.2 Model Architecture ---
    parser.add_argument('--topk', type=int, default=8, help='Top-K for BP-Book retrieval')
    parser.add_argument('--num_slots', type=int, default=2048, help='Number of prototypes for BP-Book')
    parser.add_argument("--cwt_channels", type=int, default=32, help="cwt channels")
    parser.add_argument("--d_model", type=int, default=256, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--e_layers", type=int, default=4, help="num of encoder layers")
    parser.add_argument("--d_ff", type=int, default=1024, help="dimension of fcn")
    parser.add_argument("--factor", type=int, default=1, help="attn factor")
    parser.add_argument('--num_beats', type=int, default=16, help='Number of beats for TBR_ALMR_Module')
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument(
        "--output_attention",
        action="store_true",
        help="whether to output attention in encoder",
    )
    parser.add_argument(
        "--patch_len_list",
        type=str,
        default="16,32,64,128",
        help="a list of patch len used in TFCPR",
    )
    parser.add_argument(
        "--single_channel",
        action="store_true",
        help="whether to use single channel patching for TFCPR",
        default=False,
    )
    parser.add_argument(
        "--augmentations",
        type=str,
        default="none",
        help="a comma-seperated list of augmentation types (none, jitter or scale). Append numbers to specify the strength of the augmentation, e.g., jitter0.1",
    )
    
    # --- 2.3 Loss Function Weights (Lambdas) ---
    parser.add_argument('--lambda_mse', type=float, default=1.2, help='Weight for MSE Loss')
    parser.add_argument('--lambda_almr', type=float, default=0, help='Weight for ALMR Self-Supervised Loss')
    parser.add_argument('--lambda_pcc', type=float, default=1.0, help='Weight for Pearson Correlation Coefficient Loss')
    parser.add_argument('--lambda_bp', type=float, default=0, help='Weight for SBP/DBP Loss')

    # --- 2.4 Training Hyperparams ---
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience (epochs)') # New
    
    # --- 2.5 System ---
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:2' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_path', type=str, default='./checkpoints/', help='Path to save models')

    return parser.parse_args()


def validate(model, dataloader, args, config: DataConfig, prefix="Validating"):
    model.eval()
    total_loss = 0
    steps = 0
    
    all_errors = []  
    sbp_errors = [] 
    dbp_errors = []  
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=prefix, leave=False):
            ppg = batch['ppg'].to(args.device).permute(0, 2, 1)
            ecg = batch['ecg'].to(args.device).permute(0, 2, 1)
            abp = batch['abp'].to(args.device).permute(0, 2, 1)
            ppg_raw = batch['ppg_raw'].to(args.device).permute(0, 2, 1)
            ecg_raw = batch['ecg_raw'].to(args.device).permute(0, 2, 1)
            abp_raw = batch['abp_raw'].to(args.device).permute(0, 2, 1)

            # 模型推理
            abp_pred, recon_ecg, recon_ppg, pred_sbp, pred_dbp = model(torch.concat([ecg, ppg, ecg_raw, ppg_raw], dim=2))

            # 计算损失
            loss = complex_loss(
                ecg.squeeze(2), ppg.squeeze(2), abp.squeeze(2), 
                recon_ecg, recon_ppg, abp_pred, pred_sbp, pred_dbp,
                args
            )
            total_loss += loss.item()

            pred_mmhg = denormalize_abp(abp_pred, config)  # shape: [batch_size, 1250]
            target_mmhg = denormalize_abp(abp.squeeze(), config)  # shape: [batch_size, 1250]
            
            batch_error = pred_mmhg - target_mmhg
            all_errors.append(batch_error.detach().cpu())
            
            pred_mmhg_np = pred_mmhg.detach().cpu().numpy()
            target_mmhg_np = target_mmhg.detach().cpu().numpy()
            ecg_np = ecg.squeeze(2).detach().cpu().numpy()  # [batch_size, time_steps]
            ppg_np = ppg.squeeze(2).detach().cpu().numpy()  # [batch_size, time_steps]

            # ========== 保存目录 ==========
            save_dir = 'vis_results'
            os.makedirs(save_dir, exist_ok=True)

            # ========== 核心修改：3*1子图布局 ==========
            plt.figure(figsize=(12, 9))  # 调整画布尺寸，3行图需要更大的高度

            # 第1个子图：ECG Raw信号
            plt.subplot(3, 1, 1)
            plt.plot(ecg_np[0], label='ECG Raw', color='#1f77b4', linewidth=1.0)
            plt.title('ECG Raw Signal', fontsize=10, fontweight='bold')
            plt.ylabel('Amplitude', fontsize=9)
            plt.legend(loc='upper right', fontsize=8)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tick_params(axis='both', labelsize=8)

            # 第2个子图：PPG Raw信号
            plt.subplot(3, 1, 2)
            plt.plot(ppg_np[0], label='PPG Raw', color='#2ca02c', linewidth=1.0)
            plt.title('PPG Raw Signal', fontsize=10, fontweight='bold')
            plt.ylabel('Amplitude', fontsize=9)
            plt.legend(loc='upper right', fontsize=8)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tick_params(axis='both', labelsize=8)

            # 第3个子图：ABP预测vs真实值（原有逻辑）
            plt.subplot(3, 1, 3)
            plt.plot(target_mmhg_np[0], label='Target (Ground Truth)', color='black', alpha=0.6, linewidth=1.5)
            plt.plot(pred_mmhg_np[0], label='Prediction', color='#d62728', alpha=0.8, linewidth=1.5)
            plt.title('ABP Prediction vs Ground Truth', fontsize=10, fontweight='bold')
            plt.xlabel('Time Points', fontsize=9)
            plt.ylabel('ABP (mmHg)', fontsize=9)
            plt.legend(loc='upper right', fontsize=8)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tick_params(axis='both', labelsize=8)

            # 调整子图间距，避免标题/标签重叠
            plt.subplots_adjust(hspace=0.4, top=0.95, bottom=0.05, left=0.08, right=0.95)

            # 保存图片
            save_path = os.path.join(save_dir, f'step_{steps}_comp.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')  # 提高dpi让图片更清晰

            # 关闭画布释放内存
            plt.close()

            # ====

            for pred_wave, target_wave in zip(pred_mmhg_np, target_mmhg_np):
                pred_sbp, pred_dbp = calculate_sbp_dbp_from_abp_waveform(pred_wave)
                target_sbp, target_dbp = calculate_sbp_dbp_from_abp_waveform(target_wave)
                
                if not np.isnan(pred_sbp) and not np.isnan(target_sbp):
                    sbp_errors.append(pred_sbp - target_sbp)
                if not np.isnan(pred_dbp) and not np.isnan(target_dbp):
                    dbp_errors.append(pred_dbp - target_dbp)
            
            steps += 1

    # 整体MAE/SD
    avg_loss = total_loss / steps if steps > 0 else 0
    final_mae, final_sd = 0.0, 0.0
    if len(all_errors) > 0:
        all_errors_tensor = torch.cat(all_errors).flatten()
        final_mae = torch.mean(torch.abs(all_errors_tensor)).item()
        final_sd = torch.std(all_errors_tensor).item()
    
    # SBP
    sbp_mae, sbp_sd = 0.0, 0.0
    if len(sbp_errors) > 0:
        sbp_errors_np = np.array(sbp_errors)
        sbp_mae = np.mean(np.abs(sbp_errors_np))
        sbp_sd = np.std(sbp_errors_np)
    
    # DBP
    dbp_mae, dbp_sd = 0.0, 0.0
    if len(dbp_errors) > 0:
        dbp_errors_np = np.array(dbp_errors)
        dbp_mae = np.mean(np.abs(dbp_errors_np))
        dbp_sd = np.std(dbp_errors_np)

    return avg_loss, final_mae, final_sd, sbp_mae, sbp_sd, dbp_mae, dbp_sd


def train_mode(args, model, train_loader, val_loader, data_config):
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_dir=args.save_path)

    print("Computing LDS weights...")
    lds_sbp_calculator = LDSWeightCalculator(
        all_train_labels=data_config.all_sbp_labels, 
        bucket_size=2.0, 
        smoothing_sigma=1.0
    ).to(args.device)
    # lds_dbp_calculator = LDSWeightCalculator(
    #     all_train_labels=data_config.all_dbp_labels, 
    #     bucket_size=2.0,    
    #     smoothing_sigma=1.0
    # ).to(args.device)

    print("\n--- Start Training ---")
    for epoch in range(args.epochs):
        model.train()
        train_loss_acc = 0
        steps = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in loop:
            ppg = batch['ppg'].to(args.device).permute(0, 2, 1)
            ecg = batch['ecg'].to(args.device).permute(0, 2, 1)
            abp = batch['abp'].to(args.device).permute(0, 2, 1)
            ppg_raw = batch['ppg_raw'].to(args.device).permute(0, 2, 1)
            ecg_raw = batch['ecg_raw'].to(args.device).permute(0, 2, 1)
            abp_raw = batch['abp_raw'].to(args.device).permute(0, 2, 1)
            
            optimizer.zero_grad()
            
            abp_pred, recon_ecg, recon_ppg, pred_sbp, pred_dbp = model(torch.concat([ecg, ppg, ecg_raw, ppg_raw], dim=2))

            batch_sbp_val = abp_raw.max(dim=1)[0]
            target_weights_sbp = lds_sbp_calculator.get_weights(batch_sbp_val)
            # batch_dbp_val = abp_raw.min(dim=1)[0]
            # target_weights_dbp = lds_dbp_calculator.get_weights(batch_dbp_val)

            loss = complex_loss(
                ecg.squeeze(2), ppg.squeeze(2), abp.squeeze(2), 
                recon_ecg, recon_ppg, abp_pred, pred_sbp, pred_dbp, 
                args, target_weights=target_weights_sbp
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_acc += loss.item()
            steps += 1
            
            loop.set_postfix(
                loss=loss.item()
            )
            
        scheduler.step()
        
        # --- Validation ---
        avg_loss, final_mae, final_sd, sbp_mae, sbp_sd, dbp_mae, dbp_sd = validate(model, val_loader, args, data_config)
        
        print(f"Epoch {epoch+1} Result:")
        print(f"  Train Loss: {train_loss_acc/steps:.4f}")
        print(f"  Val Loss:   {avg_loss:.4f}")
        print(f"  Val MAE:    {final_mae:.2f} mmHg")
        print(f"  Val SD:     {final_sd:.2f} mmHg")
        print(f"  Val SBP MAE:{sbp_mae:.2f} mmHg, SD: {sbp_sd:.2f} mmHg")
        print(f"  Val DBP MAE:{dbp_mae:.2f} mmHg, SD: {dbp_sd:.2f} mmHg")
        
        # --- Early Stopping Check & Saving ---
        early_stopping(final_mae, model, args.model_name, optimizer, epoch, data_config, args)
        
        if early_stopping.early_stop:
            print(f"\n[Early Stopping] Triggered after {epoch+1} epochs.")
            print(f"Best model was saved at: {early_stopping.best_model_path}")
            break

    print("Training Complete.")


def test_mode(args, model, test_loader, data_config):
    """
    封装测试逻辑
    """
    print("\n--- Start Testing ---")
    
    if args.test_checkpoint is None or not os.path.exists(args.test_checkpoint):
        print(f"Error: Checkpoint file not found at {args.test_checkpoint}")
        print("Please provide a valid path using --test_checkpoint")
        sys.exit(1)

    print(f"Loading checkpoint from: {args.test_checkpoint}")
    checkpoint = torch.load(args.test_checkpoint, map_location=args.device)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 运行测试
    avg_loss, final_mae, final_sd, sbp_mae, sbp_sd, dbp_mae, dbp_sd = validate(model, test_loader, args, data_config)
    
    print("\n" + "="*30)
    print("       TEST REPORT       ")
    print("="*30)
    print(f"Model: {args.test_checkpoint}")
    print(f"Test MAE: {final_mae:.4f} mmHg")
    print(f"Test SD:  {final_sd:.4f} mmHg")
    print(f"Test SBP MAE: {sbp_mae:.4f} mmHg, SD: {sbp_sd:.4f} mmHg")
    print(f"Test DBP MAE: {dbp_mae:.4f} mmHg, SD: {dbp_sd:.4f} mmHg")
    print(f"Test Loss: {avg_loss:.4f}")
    print("="*30 + "\n")


def main():
    args = get_args()
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    print(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"Running on device: {args.device}")
    
    data_config = DataConfig(
        data_root=args.data_root,
        original_length=args.original_length,
        window_size=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        is_train=args.is_train
    )
    
    print("--- Initializing Data Pipeline ---")
    train_loader, val_loader, test_loader = DataFactory.get_dataloaders(data_config)
    
    model = TFCPR(args).to(args.device)
    
    if args.is_train:
        train_mode(args, model, train_loader, val_loader, data_config)
    else:
        test_mode(args, model, test_loader, data_config)

if __name__ == '__main__':
    main()