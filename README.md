# TF-CPR
Time-Frequency Complementarity and Prototype Retrieval for Calibration-free Blood Pressure Estimation Network


```bash

nohup python run.py > error.log 2>&1 &

```

# timeline

1. 25.11.27 create repository


# results

Epoch 23 Result:
  Train Loss: 0.0492
  Val Loss:   0.4611
  Val MAE:    12.43 mmHg
  Val SD:     16.08 mmHg
  Val SBP MAE:15.44 mmHg, SD: 19.56 mmHg
  Val DBP MAE:9.88 mmHg, SD: 12.63 mmHg
Validation metric improved. Saving model...
  [Saved Best Model] -> ./checkpoints/tfcpr_20251203_124556_ep22_mae12.4312.pth (MAE: 12.43)

- Namespace(model_name='tfcpr', is_train=True, test_checkpoint='./checkpoints/tfcpr_20251130_220345_ep2_mae12.6049.pth', data_root='./dataset/mimicbp', original_length=3750, seq_len=625, pred_len=625, stride=625, train_ratio=0.7, val_ratio=0.1, num_workers=4, topk=8, num_slots=2048, cwt_channels=32, d_model=256, n_heads=8, e_layers=4, d_ff=1024, factor=1, num_beats=16, dropout=0.1, activation='gelu', output_attention=False, patch_len_list='16,32,64,128', single_channel=False, augmentations='jitter,mask,drop,scale', lambda_mse=1.2, lambda_almr=0.2, lambda_pcc=1.0, batch_size=512, epochs=100, lr=0.0005, weight_decay=0.0001, patience=15, seed=27, device='cuda:0', save_path='./checkpoints/')
