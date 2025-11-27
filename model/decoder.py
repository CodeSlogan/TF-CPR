import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalityFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.proj = nn.Linear(d_model*4, d_model)
        
    def forward(self, ecg, ppg):
        ecg_aligned, _ = self.attn(ecg, ppg, ppg)
        ppg_aligned, _ = self.attn(ppg, ecg, ecg)
        return self.proj(torch.cat([ecg, ppg, ecg_aligned, ppg_aligned], dim=-1))


class AdaIN(nn.Module):
    def __init__(self, c, s):
        super().__init__()
        self.fc = nn.Linear(s, c*2)
        
    def forward(self, x, style):
        g, b = self.fc(style).unsqueeze(-1).chunk(2, 1)
        return ((x - x.mean(2, keepdim=True))/(x.std(2, keepdim=True)+1e-8)) * g + b


class BPDecoder(nn.Module):
    def __init__(self, d_model, fft_dim, cwt_dim, target_len):
        super().__init__()
        self.target_len = target_len
        self.cwt_proj = nn.Conv1d(cwt_dim*2, d_model, 1)
        
        self.up1 = nn.Upsample(scale_factor=5)
        self.conv1 = nn.Conv1d(d_model*2, 64, 3, padding=1)
        self.adain1 = AdaIN(64, fft_dim*2)

        self.up2 = nn.Upsample(scale_factor=6)
        self.conv2 = nn.Conv1d(64, 32, 3, padding=1)
        self.adain2 = AdaIN(32, fft_dim*2)
        self.out = nn.Conv1d(32, 1, 1)
        
    def forward(self, x, fft, cwt):
        x = x.transpose(1, 2) # [B, D, N]
        cwt = self.cwt_proj(F.interpolate(cwt, size=x.shape[2]))
        x = torch.cat([x, cwt], 1)
        
        x = self.adain1(F.gelu(self.conv1(self.up1(x))), fft)
        x = self.adain2(F.gelu(self.conv2(self.up2(x))), fft)
        
        return self.out(F.interpolate(x, size=self.target_len)).transpose(1, 2)