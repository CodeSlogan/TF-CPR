import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random

# ==========================================
# 1. 导入你的模型 (根据你的文件名修改 import)
# ==========================================
from model.tfcpr.TFCPR import Medformer
# ==========================================
# 2. 极简配置 (专门用于 Debug)
# ==========================================
class DebugConfig:
    # 基础维度
    seq_len = 625
    pred_len = 625
    enc_in = 2       # 假设输入是 PPG + ECG (2通道)
    single_channel = False 
    
    # 核心参数 (缩小规模，方便快速排查)
    d_model = 64     # 故意设小，以此测试梯度流是否通畅
    n_heads = 2
    e_layers = 2
    d_ff = 128
    dropout = 0.0    # 【关键】过拟合测试必须把 Dropout 设为 0
    activation = 'gelu'
    output_attention = False
    
    # Patch 相关
    patch_len_list = "16,32"
    
    # 你的自定义模块参数
    cwt_channels = 16
    num_beats = 8
    num_prototypes = 10
    topK = 3
    is_train = True  # 开启训练模式

# ==========================================
# 3. 准备伪数据 (模拟你的输入形状)
# ==========================================
def get_dummy_batch(batch_size=8, seq_len=625):
    # 模拟输入: (Batch, Seq_Len, Channels)
    # 假设 Ch0=ECG, Ch1=PPG，用正弦波模拟，保证有规律可循
    t = np.linspace(0, 10, seq_len)
    x_data = []
    y_data = []
    
    for _ in range(batch_size):
        # 随机相位和频率，制造一点差异
        freq = random.uniform(0.5, 2.0)
        phase = random.uniform(0, np.pi)
        
        # 输入: 简单的波形
        ecg = np.sin(2 * np.pi * freq * t + phase)
        ppg = np.cos(2 * np.pi * freq * t + phase) * 0.5
        x = np.stack([ecg, ppg], axis=1) # (Seq, 2)
        
        # 目标: 假设我们要预测某种变换后的波形 (例如去除噪声后的 PPG)
        # 这里简单设为 PPG * 2，强迫模型学习线性关系
        target = ppg * 2.0 
        
        x_data.append(x)
        y_data.append(target)
        
    x_tensor = torch.tensor(np.array(x_data), dtype=torch.float32) # (B, 625, 2)
    y_tensor = torch.tensor(np.array(y_data), dtype=torch.float32) # (B, 625)
    
    return x_tensor, y_tensor

# ==========================================
# 4. 测试主循环
# ==========================================
def run_overfit_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device: {device}")
    
    # 初始化配置和模型
    configs = DebugConfig()
    
    # --- 这里请实例化你的模型 ---
    model = Medformer(configs).to(device)

    try:
        model = Medformer(configs).to(device) # 确保 Medformer 类在当前作用域可用
    except NameError:
        print("错误：找不到 Medformer 类。请确保你已经 Import 了你的模型代码。")
        return

    # 定义优化器和 Loss
    # 使用较大的学习率，单 Batch 应该收敛很快
    optimizer = optim.Adam(model.parameters(), lr=1e-3) 
    criterion = nn.MSELoss()
    
    # 获取唯一的 Batch，并在整个循环中重复使用它！
    x_batch, y_batch = get_dummy_batch(batch_size=8)
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    
    print("\nStarting Overfit Test (500 Epochs)...")
    losses = []
    
    model.train() # 确保 Dropout 关掉 (虽然我们在 Config 里设了0)
    
    for epoch in range(500):
        optimizer.zero_grad()
        
        # 前向传播
        # 注意：根据你的 forward 返回值调整这里
        # 你的代码似乎返回: y_pred, recon_ecg, recon_ppg
        outputs = model(x_batch)
        
        if isinstance(outputs, tuple):
            pred = outputs[0] # 取主输出
        else:
            pred = outputs
            
        # 计算 Loss
        loss = criterion(pred, y_batch)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪 (防止 Transformer 梯度爆炸)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

    # ==========================================
    # 5. 结果可视化与诊断
    # ==========================================
    print("\nTest Finished. Plotting results...")
    
    # 【新增】设置后端，防止在无显示器的服务器上报错
    import matplotlib
    matplotlib.use('Agg') 
    
    plt.figure(figsize=(12, 5))
    
    # 图 1: Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("Training Loss (Should go to 0)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale('log') # 对数坐标看细节
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # 图 2: 预测 vs 真值 (取第0个样本)
    plt.subplot(1, 2, 2)
    with torch.no_grad():
        model.eval()
        test_out = model(x_batch)
        if isinstance(test_out, tuple):
            test_pred = test_out[0] # 取主输出
        else:
            test_pred = test_out
            
    pred_np = test_pred[0].cpu().numpy()
    true_np = y_batch[0].cpu().numpy()
    
    # 如果是多通道，只画第一个通道
    if pred_np.ndim > 1:
        pred_np = pred_np[:, 0]
        true_np = true_np[:, 0]
    
    plt.plot(true_np, label='Ground Truth', color='black', alpha=0.7, linewidth=2)
    plt.plot(pred_np, label='Prediction', color='red', linestyle='--', linewidth=2)
    plt.title("Prediction Visualization (Sample 0)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 【新增】保存图片
    save_path = 'debug_result.png'
    plt.savefig(save_path, dpi=150)
    print(f"可视化结果已保存至: {save_path}")
    print("请下载或打开该图片查看拟合情况。")

if __name__ == "__main__":
    run_overfit_test()