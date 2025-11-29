import torch
import torch.nn as nn

from .feature_extractors import PatchEmbedding, FrequencyModule, ALMR
from .layers import Transformer_BPEncoder
from .decoder import CrossModalityFusion, BPDecoder

class Model(nn.Module):
    def __init__(self, seq_len=3000, patch_size=30, d_model=128, 
                 num_layers=2, nhead=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        
        # 1. patch and frequency extraction
        self.patch_embed = PatchEmbedding(1, d_model, patch_size)
        self.freq_module = FrequencyModule(seq_len, d_model)
        
        # 2. Beat Enhanced (ALMR)
        self.ecg_almr = ALMR(d_model, patch_size=patch_size)
        self.ppg_almr = ALMR(d_model, patch_size=patch_size)
        
        # 3. PPG/ECG Encoder (BP-Book Memory Enhanced)
        self.ecg_encoder = Transformer_BPEncoder(d_model, num_layers=num_layers, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.ppg_encoder = Transformer_BPEncoder(d_model, num_layers=num_layers, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        
        # 4. PPG+ECG and decode
        self.fusion = CrossModalityFusion(d_model)
        self.decoder = BPDecoder(d_model, 64, 32, seq_len)

    def forward(self, ecg, ppg, is_train=False):
        """
        ecg, ppg: [Batch, Seq_Len, 1]
        """
        # [B, L, 1] -> [B, 1, L]
        ecg_in, ppg_in = ecg.transpose(1, 2), ppg.transpose(1, 2)
        
        ecg_fft, ecg_cwt = self.freq_module(ecg_in)
        ppg_fft, ppg_cwt = self.freq_module(ppg_in)
        fft_cat = torch.cat([ecg_fft, ppg_fft], -1)
        cwt_cat = torch.cat([ecg_cwt, ppg_cwt], 1)
        
        h_ecg = self.patch_embed(ecg_in)
        h_ppg = self.patch_embed(ppg_in)
        
        h_ecg, ecg_restore = self.ecg_almr(h_ecg)
        h_ppg, ppg_restore = self.ppg_almr(h_ppg)
        
        h_ecg = self.ecg_encoder(h_ecg)
        h_ppg = self.ppg_encoder(h_ppg)
        
        h_fused = self.fusion(h_ecg, h_ppg)
        bp_pred = self.decoder(h_fused, fft_cat, cwt_cat)
        
        if is_train:
            return bp_pred, ecg_restore, ppg_restore
        return bp_pred, 0.0, 0.0