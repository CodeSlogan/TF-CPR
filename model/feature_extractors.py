import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_c=1, d_model=128, patch_size=30):
        super().__init__()
        self.proj = nn.Conv1d(in_c, d_model, patch_size, stride=patch_size)
        
    def forward(self, x):
        return self.proj(x).transpose(1, 2)


class FrequencyModule(nn.Module):
    def __init__(self, seq_len, d_model=128, fft_dim=64, cwt_dim=32):
        super().__init__()
        
        # 1. FFT Path
        self.fft_mlp = nn.Sequential(nn.Linear(seq_len//2+1, fft_dim), nn.GELU())

        # 2. CWT Path (多尺度卷积模拟)
        self.scale_1 = nn.Conv1d(1, 8, kernel_size=3, padding=1)   # 高频
        self.scale_2 = nn.Conv1d(1, 8, kernel_size=7, padding=3)   # 中高频
        self.scale_3 = nn.Conv1d(1, 8, kernel_size=15, padding=7)  # 中低频
        self.scale_4 = nn.Conv1d(1, 8, kernel_size=31, padding=15) # 低频
        
        self.bn = nn.BatchNorm1d(32) # 8*4=32
        self.act = nn.GELU()
        
        # 融合层
        self.fusion = nn.Conv1d(32, cwt_dim, kernel_size=1)

    def forward(self, x):
        # x: [B, 1, L]
        
        # FFT
        fft = self.fft_mlp(torch.fft.rfft(x.squeeze(1), dim=-1).abs())
        
        # CWT (Multi-scale)
        s1 = self.scale_1(x)
        s2 = self.scale_2(x)
        s3 = self.scale_3(x)
        s4 = self.scale_4(x)
        
        # 拼接不同尺度的特征 -> [B, 32, L]
        cwt_feat = torch.cat([s1, s2, s3, s4], dim=1)
        cwt_feat = self.act(self.bn(cwt_feat))
        
        # 融合 -> [B, cwt_dim, L]
        cwt = self.fusion(cwt_feat)
        
        return fft, cwt


class ALMR(nn.Module):
    def __init__(self, d_model=128, num_beat_queries=16, patch_size=30):
        super().__init__()
        self.patch_size = patch_size
        
        # Beat Queries
        self.beat_queries = nn.Parameter(torch.randn(1, num_beat_queries, d_model))
        
        # Attention Layers
        # Q=Beat, K=Token, V=Token -> Output is Updated Beat
        self.attn_token2beat = nn.MultiheadAttention(d_model, 4, batch_first=True)
        self.attn_beat2token = nn.MultiheadAttention(d_model, 4, batch_first=True)
        
        # ALMR Decoder
        # 输入 Beat Embedding [B, K, D] -> 输出 Patch原型 [B, K, P]
        self.almr_decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, patch_size) 
        )

    def forward(self, x_token):
        """
        x_token: [B, N, D]
        Returns: 
            x_enhanced: [B, N, D]
            rec_signal: [B, N, P] (用于和原始信号 Reshape 后做比较)
        """
        B, N, D = x_token.shape
        P = self.patch_size
        
        # 1. Latent Beat Queries
        q_beat = self.beat_queries.repeat(B, 1, 1) # [B, K, D]
        
        # 2. Token -> Beat (获取 Beat Embedding 和 Attention Map)
        # beats: [B, K, D], attn_map: [B, K, N]
        beats, attn_map = self.attn_token2beat(q_beat, x_token, x_token)
        
        # --- ALMR 计算 (在内部完成) ---
        
        # A. 解码 Beat 原型: [B, K, D] -> [B, K, P]
        beat_prototypes = self.almr_decoder(beats)
        
        # B. 广播与加权 (Reconstruction Logic)
        # 我们需要计算: Sum_over_K (Attn_k * Prototype_k)
        # Attn: [B, K, N] -> [B, K, N, 1]
        attn_expanded = attn_map.unsqueeze(-1)
        
        # Prototype: [B, K, P] -> [B, K, 1, P]
        proto_expanded = beat_prototypes.unsqueeze(2)
        
        # 加权组合: [B, K, N, P]
        weighted_patches = attn_expanded * proto_expanded
        
        # 对 K 维度求和 -> [B, N, P]
        # 这就是重构出的信号，维度是 (Batch, Token数, Patch大小)
        # 这正好对应原始信号被 Reshape 成 [B, N, P] 后的形状
        rec_signal = weighted_patches.sum(dim=1) 
        
        # ----------------------------

        # 3. Beat -> Token (注入回 Token)
        context, _ = self.attn_beat2token(x_token, beats, beats)
        x_enhanced = x_token + context
        
        return x_enhanced, rec_signal