import argparse
import torch
from model import Model

def get_args():
    parser = argparse.ArgumentParser(description='CalibrationFreeBPNet Training')

    # --- 1. Model Architecture ---
    parser.add_argument('--d_model', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of Transformer layers')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--patch_size', type=int, default=30, help='Patch size for embedding')
    parser.add_argument('--seq_len', type=int, default=3000, help='Input sequence length of ECG/PPG')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # --- 2. Training Hyperparams ---
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    
    # --- 3. System ---
    parser.add_argument('--seed', type=int, default=2026, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_path', type=str, default='./checkpoints/', help='Path to save models')

    return parser.parse_args()

def main():
    args = get_args()
    
    print("Training Config:", vars(args))
    torch.manual_seed(args.seed)
    
    model = Model(
        seq_len=args.seq_len,
        patch_size=args.patch_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        nhead=args.nhead,
        dropout=args.dropout
    ).to(args.device)
    
    print(f"Model created successfully on {args.device}!")

    dummy_ecg = torch.randn(args.batch_size, args.seq_len, 1).to(args.device)
    dummy_ppg = torch.randn(args.batch_size, args.seq_len, 1).to(args.device)

    # Forward
    bp_pred, ecg_restore, ppg_restore = model(dummy_ecg, dummy_ppg, is_train=True)
    print(f"Output shape: {bp_pred.shape}")

if __name__ == '__main__':
    main()