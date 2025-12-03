import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path
from tqdm import tqdm  
import torch.nn.functional as F


@dataclass
class DataConfig:
    data_root: str
    sample_rate: int = 125
    original_length: int = 3750
    window_size: int = 625
    stride: int = 625
    
    batch_size: int = 32
    num_workers: int = 4
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    seed: int = 2026
    
    abp_global_median: float = None 
    abp_global_iqr: float = None
    is_train : bool = True
    all_sbp_labels: List[float] = None  # 用于 LDS 计算
    all_dbp_labels: List[float] = None  # 用于 LDS 计算


class MimicBPDataset(Dataset):
    def __init__(self, subject_ids: List[str], config: DataConfig, mode: str = 'train'):
        self.config = config
        self.mode = mode
        self.subject_ids = subject_ids
        self.indices = self._build_indices()
        
        # 检查是否已配置 ABP 统计量
        if self.config.abp_global_median is None or self.config.abp_global_iqr is None:
            raise ValueError("ABP global statistics (median/iqr) are not set in config!")

    def _build_indices(self) -> List[Tuple[str, int, int]]:
        """构建索引 (保持不变)"""
        indices_list = []
        n_segments_per_row = (self.config.original_length - self.config.window_size) // self.config.stride + 1
        rows_per_file = 30 
        
        for pid in self.subject_ids:
            for row in range(rows_per_file):
                for seg in range(n_segments_per_row):
                    start_col = seg * self.config.stride
                    indices_list.append((pid, row, start_col))
        return indices_list

    def _load_npy(self, file_path: Path) -> np.ndarray:
        try:
            return np.load(file_path).astype(np.float32)
        except FileNotFoundError:
            print(f"Warning: File missing {file_path}")
            return np.zeros((30, 3750), dtype=np.float32)

    def _apply_instance_zscore(self, x_slice: np.ndarray, x_full: np.ndarray) -> np.ndarray:
        """
        修改后：使用完整数据 x_full 的统计量(mean, std) 对切片 x_slice 进行标准化
        公式: (x_slice - mu_full) / sigma_full
        """
        mu = np.mean(x_full)
        sigma = np.std(x_full)
        return (x_slice - mu) / (sigma + 1e-6)

    # --- 核心修改 2: Global Robust Normalization ---
    def _apply_global_robust(self, x: np.ndarray) -> np.ndarray:
        """
        使用全局中位数和 IQR 进行标准化。保留物理意义，抵抗离群点。
        """
        median = self.config.abp_global_median
        iqr = self.config.abp_global_iqr

        return (x - median) / (iqr + 1e-6)

    def __len__(self):
        return len(self.indices)
    
    def _augment(self, ppg, ecg):
        """
        专门针对生理信号的增强策略
        """
        # 1. 随机缩放 (Random Scaling): 模拟不同传感器增益/接触紧密度
        # 即使波形变大变小，模型也应该能通过形态判断出血压
        scale_factor = random.uniform(0.8, 1.2)
        ppg = ppg * scale_factor
        # ECG 通常相对稳定，但也稍微给点扰动
        ecg = ecg * random.uniform(0.9, 1.1)

        # 2. 随机直流漂移 (Random Baseline Wander)
        bias = random.uniform(-0.1, 0.1)
        ppg = ppg + bias

        # 4. 随机噪声 (Gaussian Noise)
        noise = np.random.normal(0, 0.01, ppg.shape).astype(np.float32)
        ppg = ppg + noise
        
        return ppg, ecg

    def __getitem__(self, idx):
        subject_id, row_idx, start_col = self.indices[idx]
        end_col = start_col + self.config.window_size
        root = Path(self.config.data_root)
        
        ppg_data = self._load_npy(root / 'ppg' / f"p{subject_id}_ppg.npy")
        ecg_data = self._load_npy(root / 'ecg' / f"p{subject_id}_ecg.npy")
        abp_data = self._load_npy(root / 'abp' / f"p{subject_id}_abp.npy")
        
        ppg_full_row = ppg_data[row_idx]
        ecg_full_row = ecg_data[row_idx]

        ppg_slice = ppg_data[row_idx, start_col:end_col]
        ecg_slice = ecg_data[row_idx, start_col:end_col]
        abp_slice = abp_data[row_idx, start_col:end_col]

        if self.config.is_train:
            ppg_slice, ecg_slice = self._augment(ppg_slice, ecg_slice)
        
        ppg_norm = self._apply_instance_zscore(ppg_slice, ppg_full_row)
        ecg_norm = self._apply_instance_zscore(ecg_slice, ecg_full_row)
        
        abp_norm = self._apply_global_robust(abp_slice)
        
        sample = {
            'ppg_raw': torch.from_numpy(ppg_slice).unsqueeze(0), # (1, L)
            'ecg_raw': torch.from_numpy(ecg_slice).unsqueeze(0), # (1, L)
            'abp_raw': torch.from_numpy(abp_slice).unsqueeze(0), # (1, L)
            'ppg': torch.from_numpy(ppg_norm).unsqueeze(0), # (1, L)
            'ecg': torch.from_numpy(ecg_norm).unsqueeze(0), # (1, L)
            'abp': torch.from_numpy(abp_norm).unsqueeze(0)  # (1, L)
        }
        return sample

class StatisticsCalculator:
    @staticmethod
    def compute_abp_stats(subject_ids: List[str], config: DataConfig) -> Tuple[float, float, List[float]]:
        """
        修改后：同时计算 Median, IQR 以及收集所有的 SBP 标签 (用于 LDS)。
        返回: (median, iqr, all_sbp_labels)
        """
        print(f"Computing Global ABP Statistics & Collecting SBP labels on {len(subject_ids)} subjects...")
        
        abp_dir = Path(config.data_root) / 'abp'
        
        # 1. 用于计算全局归一化 (Median/IQR) 的数据容器
        all_abp_points = [] 
        
        # 2. 用于 LDS 的 SBP 标签容器
        collected_sbp_labels = []
        collected_dbp_labels = []

        for pid in tqdm(subject_ids, desc="Scanning ABP files"):
            file_path = abp_dir / f"p{pid}_abp.npy"
            if file_path.exists():
                try:
                    data = np.load(file_path, mmap_mode='r') 
                    
                    if data.size == 0: continue

                    # --- A. 收集全量点 (用于 Median/IQR) ---
                    flat_data = data.flatten().astype(np.float32)
                    all_abp_points.append(flat_data)
                    
                    # --- B. 收集 SBP 标签 (用于 LDS) ---
                    
                    n_windows = data.shape[1] // config.window_size
                    # 截取能整除的部分
                    valid_len = n_windows * config.window_size
                    truncated_data = data[:, :valid_len] 
                    
                    # Reshape: (30, 6, 625) -> (180, 625)
                    reshaped_data = truncated_data.reshape(-1, config.window_size)
                    
                    # 取每个窗口的最大值作为该样本的 SBP
                    # axis=1 表示在窗口维度取 max
                    window_maxes = np.max(reshaped_data, axis=1) 
                    window_mins = np.min(reshaped_data, axis=1)
                    
                    collected_sbp_labels.extend(window_maxes.tolist())
                    collected_dbp_labels.extend(window_mins.tolist())
                        
                except Exception as e:
                    print(f"Warning: Error reading {pid}: {e}")
                    continue

        if not all_abp_points:
            raise RuntimeError("No ABP data found to compute stats.")
            
        # --- C. 计算全局统计量 ---
        print("Concatenating all points for Stats (this might take RAM)...")
        global_abp = np.concatenate(all_abp_points)
        print(f"Total points: {len(global_abp):,}")
        
        print("Calculating Median and IQR...")
        median = np.median(global_abp)
        q75, q25 = np.percentile(global_abp, [75, 25])
        iqr = q75 - q25
        
        print(f"Global Stats: Median={median:.4f}, IQR={iqr:.4f}")
        print(f"Collected {len(collected_sbp_labels)} SBP & DBP labels for LDS.")
        
        return float(median), float(iqr), collected_sbp_labels, collected_dbp_labels


class DataFactory:
    @staticmethod
    def get_dataloaders(config: DataConfig):
        root = Path(config.data_root)
        
        all_files = glob.glob(str(root / 'ppg' / "*_ppg.npy"))
        if not all_files: raise RuntimeError("No data found")
        subject_ids = sorted(list(set([Path(f).stem.split('_')[0].replace('p', '') for f in all_files])))
        
        # 划分数据集
        train_ids, test_ids = train_test_split(subject_ids, test_size=(1 - config.train_ratio), random_state=config.seed)
        relative_val_ratio = config.val_ratio / (1 - config.train_ratio)
        val_ids, test_ids = train_test_split(test_ids, test_size=(1 - relative_val_ratio), random_state=config.seed)
        
        # --- 核心步骤：同时计算统计量和收集 SBP 列表 ---
        median, iqr, sbp_labels, dbp_labels = StatisticsCalculator.compute_abp_stats(train_ids, config)
        
        config.abp_global_median = median
        config.abp_global_iqr = iqr
        config.all_sbp_labels = sbp_labels  
        config.all_dbp_labels = dbp_labels

        # 创建 Dataset
        train_ds = MimicBPDataset(train_ids, config, mode='train')
        val_ds = MimicBPDataset(val_ids, config, mode='val')
        test_ds = MimicBPDataset(test_ids, config, mode='test')
        
        # 创建 DataLoader
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
        
        return train_loader, val_loader, test_loader
    

def denormalize_abp(tensor: torch.Tensor, config: DataConfig) -> torch.Tensor:
        return tensor * config.abp_global_iqr + config.abp_global_median

if __name__ == "__main__":
    # 配置
    config = DataConfig(data_root="../dataset/mimicbp", batch_size=16)

    # 获取 DataLoader
    train_dl, val_dl, test_dl = DataFactory.get_dataloaders(config)
    
    # 检查一个 Batch
    batch = next(iter(train_dl))
    ecg, ppg, abp = batch['ecg'], batch['ppg'], batch['abp']
    
    print("\n--- Data Inspection ---")
    print(f"ABP Global Median Used: {config.abp_global_median:.2f}")
    print(f"ABP Global IQR Used:    {config.abp_global_iqr:.2f}")
    
    print(f"\nPPG Sample (Instance Z-Score): Mean={ppg[0].mean():.4f}, Std={ppg[0].std():.4f}")
    print("  (Expect Mean ≈ 0, Std ≈ 1)")

    print(f"\nECG Sample (Instance Z-Score): Mean={ecg[0].mean():.4f}, Std={ecg[0].std():.4f}")
    print("  (Expect Mean ≈ 0, Std ≈ 1)")
    
    print(f"\nABP Sample (Global Robust):    Mean={abp[0].mean():.4f}, Min={abp[0].min():.4f}, Max={abp[0].max():.4f}")
    
    reconstructed_abp = denormalize_abp(abp, config)
    print(f"\nReconstructed ABP (mmHg):      Mean={reconstructed_abp[0].mean():.2f}")
