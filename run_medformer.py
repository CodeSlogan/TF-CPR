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


from utils.data_factory import DataFactory, DataConfig, denormalize_abp
from utils.loss import complex_loss
from utils.earlystop import EarlyStopping
from baseline.medformer.Medformer import Medformer


def get_args():
    parser = argparse.ArgumentParser(description='medformer')

    # --- 1. Mode Selection (New) ---
    parser.add_argument('--model_name', type=str, default="medformer", help='Model name for saving checkpoints')
    parser.add_argument('--is_train', type=bool, default=True, help='train or test')
    parser.add_argument('--test_checkpoint', type=str, default=None, help='Path to checkpoint for testing mode')

    # --- 2.1 Dataset Parameters ---
    parser.add_argument('--data_root', type=str, default='./dataset/mimicbp', help='Data root directory')
    parser.add_argument('--original_length', type=int, default=3750, help='Raw NPY file row length')
    parser.add_argument('--seq_len', type=int, default=625, help='Input window size (slicing length)')
    parser.add_argument(
    "--pred_len", type=int, default=625, help="prediction sequence length"
    )
    parser.add_argument('--stride', type=int, default=625, help='Sliding window stride')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Train set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation set ratio')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')

    # --- 2.2 Model Architecture ---
    parser.add_argument(
        "--task_name",
        type=str,
        default="long_term_forecast",
        help="task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]",
    )
    # inputation task
    parser.add_argument("--mask_rate", type=float, default=0.25, help="mask ratio")

    # anomaly detection task
    parser.add_argument(
        "--anomaly_ratio", type=float, default=0.25, help="prior anomaly ratio (%)"
    )

    parser.add_argument("--enc_in", type=int, default=2, help="encoder input size")
    parser.add_argument("--dec_in", type=int, default=2, help="decoder input size")
    parser.add_argument("--c_out", type=int, default=2, help="output size")
    # model define for baselines
    parser.add_argument("--d_model", type=int, default=256, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--e_layers", type=int, default=12, help="num of encoder layers")
    parser.add_argument("--d_layers", type=int, default=6, help="num of decoder layers")
    parser.add_argument("--d_ff", type=int, default=512, help="dimension of fcn")
    parser.add_argument(
        "--moving_avg", type=int, default=25, help="window size of moving average"
    )
    parser.add_argument("--factor", type=int, default=1, help="attn factor")

    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument(
        "--embed",
        type=str,
        default="timeF",
        help="time features encoding, options:[timeF, fixed, learned]",
    )
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument(
        "--output_attention",
        action="store_true",
        help="whether to output attention in encoder",
    )
    parser.add_argument(
        "--no_inter_attn",
        action="store_true",
        help="whether to use inter-attention in encoder, using this argument means not using inter-attention",
        default=False,
    )
    parser.add_argument(
        "--sampling_rate", type=int, default=256, help="frequency sampling rate"
    )
    parser.add_argument(
        "--patch_len_list",
        type=str,
        default="8,8,8,16,16,16,32,32,32,64,64,64",
        help="a list of patch len used in Medformer",
    )
    parser.add_argument(
        "--single_channel",
        action="store_true",
        help="whether to use single channel patching for Medformer",
        default=False,
    )
    parser.add_argument(
        "--augmentations",
        type=str,
        default="flip,shuffle,jitter,mask,drop",
        help="a comma-seperated list of augmentation types (none, jitter or scale). Append numbers to specify the strength of the augmentation, e.g., jitter0.1",
    )
    
    # --- 2.3 Loss Function Weights (Lambdas) ---
    parser.add_argument('--lambda_mse', type=float, default=1.0, help='Weight for MSE Loss')
    parser.add_argument('--lambda_deriv', type=float, default=1.0, help='Weight for Derivative (Shape) Loss')
    parser.add_argument('--lambda_freq', type=float, default=0.1, help='Weight for Frequency Domain Loss')
    parser.add_argument('--lambda_almr', type=float, default=0.1, help='Weight for ALMR Self-Supervised Loss')

    # --- 2.4 Training Hyperparams ---
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (epochs)') # New
    
    # --- 2.5 System ---
    parser.add_argument('--seed', type=int, default=2026, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_path', type=str, default='./checkpoints/', help='Path to save models')

    return parser.parse_args()

def validate(model, dataloader, args, config: DataConfig, prefix="Validating"):
    model.eval()
    total_loss = 0
    steps = 0
    
    all_errors = []  # 整体ABP波形误差
    sbp_errors = []  # SBP单独误差
    dbp_errors = []  # DBP单独误差
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=prefix, leave=False):
            ppg = batch['ppg'].to(args.device).permute(0, 2, 1)
            ecg = batch['ecg'].to(args.device).permute(0, 2, 1)
            abp = batch['abp'].to(args.device).permute(0, 2, 1)

            # 模型推理
            pred_abp_final, pred_shape, pred_stats, recon_ecg, recon_ppg = model(torch.concat([ecg, ppg], dim=2))

            # 计算损失
            loss = nn.MSELoss()(abp_pred, abp.squeeze())
            total_loss += loss.item()

            # 反归一化得到mmHg单位的波形
            pred_mmhg = denormalize_abp(pred_abp_final, config)  # shape: [batch_size, 1250]
            target_mmhg = denormalize_abp(abp.squeeze(), config)  # shape: [batch_size, 1250]
            
            # 1. 计算整体波形误差（原有逻辑保留）
            batch_error = pred_mmhg - target_mmhg
            all_errors.append(batch_error.detach().cpu())
            
            # 2. 计算SBP和DBP单独误差（新增逻辑）
            pred_mmhg_np = pred_mmhg.detach().cpu().numpy()
            target_mmhg_np = target_mmhg.detach().cpu().numpy()
            
            for pred_wave, target_wave in zip(pred_mmhg_np, target_mmhg_np):
                # 计算预测的SBP/DBP
                pred_sbp, pred_dbp = calculate_sbp_dbp_from_abp_waveform(pred_wave)
                # 计算真实的SBP/DBP
                target_sbp, target_dbp = calculate_sbp_dbp_from_abp_waveform(target_wave)
                
                # 收集有效误差（排除nan值）
                if not np.isnan(pred_sbp) and not np.isnan(target_sbp):
                    sbp_errors.append(pred_sbp - target_sbp)
                if not np.isnan(pred_dbp) and not np.isnan(target_dbp):
                    dbp_errors.append(pred_dbp - target_dbp)
            
            steps += 1

    # 计算原有指标（整体MAE/SD）
    avg_loss = total_loss / steps if steps > 0 else 0
    final_mae, final_sd = 0.0, 0.0
    if len(all_errors) > 0:
        all_errors_tensor = torch.cat(all_errors).flatten()
        final_mae = torch.mean(torch.abs(all_errors_tensor)).item()
        final_sd = torch.std(all_errors_tensor).item()
    
    # 计算SBP单独指标（MAE/SD）
    sbp_mae, sbp_sd = 0.0, 0.0
    if len(sbp_errors) > 0:
        sbp_errors_np = np.array(sbp_errors)
        sbp_mae = np.mean(np.abs(sbp_errors_np))
        sbp_sd = np.std(sbp_errors_np)
    
    # 计算DBP单独指标（MAE/SD）
    dbp_mae, dbp_sd = 0.0, 0.0
    if len(dbp_errors) > 0:
        dbp_errors_np = np.array(dbp_errors)
        dbp_mae = np.mean(np.abs(dbp_errors_np))
        dbp_sd = np.std(dbp_errors_np)

    # 返回原有指标 + SBP/DBP单独指标
    return avg_loss, final_mae, final_sd, sbp_mae, sbp_sd, dbp_mae, dbp_sd



def train_mode(args, model, train_loader, val_loader, loss_lambdas, data_config):
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_dir=args.save_path)
    criterion = nn.MSELoss()

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

            # print(ppg.shape, ecg.shape, abp.shape)
            
            optimizer.zero_grad()
            
            abp_pred = model(torch.concat([ecg, ppg], dim=2))
            # print(abp_pred.shape)
            loss = criterion(abp_pred, abp.squeeze())
            
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
        avg_loss, final_mae, final_sd, sbp_mae, sbp_sd, dbp_mae, dbp_sd = validate(model, val_loader, args, data_config, loss_lambdas)
        
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


def test_mode(args, model, test_loader, loss_lambdas, data_config):
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
    test_loss, test_mae, test_sd = validate(model, test_loader, args, data_config, loss_lambdas, prefix="Testing")
    
    print("\n" + "="*30)
    print("       TEST REPORT       ")
    print("="*30)
    print(f"Model: {args.test_checkpoint}")
    print(f"Test MAE: {test_mae:.4f} mmHg")
    print(f"Test SD:  {test_sd:.4f} mmHg")
    print(f"Test Loss: {test_loss:.4f}")
    print("="*30 + "\n")


def main():
    args = get_args()
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"Running on device: {args.device}")

    loss_lambdas = {
        'bp_mse': args.lambda_mse,
        'bp_deriv': args.lambda_deriv,
        'bp_freq': args.lambda_freq,
        'almr': args.lambda_almr
    }
    
    data_config = DataConfig(
        data_root=args.data_root,
        original_length=args.original_length,
        window_size=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    
    print("--- Initializing Data Pipeline ---")
    train_loader, val_loader, test_loader = DataFactory.get_dataloaders(data_config)
    
    model = Medformer(args).to(args.device)
    
    if args.is_train:
        train_mode(args, model, train_loader, val_loader, loss_lambdas, data_config)
    else:
        test_mode(args, model, test_loader, loss_lambdas, data_config)

if __name__ == '__main__':
    main()