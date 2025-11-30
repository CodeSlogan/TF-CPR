import torch
import torch.nn as nn
import torch.nn.functional as F
from .Medformer_EncDec import EncoderLayer, Encoder
from .SelfAttention_Family import MedformerLayer, FullAttention, AttentionLayer
from .Embed import ListPatchEmbedding


class FreqFeatureExtractor(nn.Module):
    def __init__(self, seq_len, d_model, cwt_channels=32):
        super().__init__()
        self.seq_len = seq_len
        self.fft_mlp = nn.Sequential(
            nn.Linear((seq_len // 2 + 1) * 2, 128),
            nn.ReLU(),
            nn.Linear(128, d_model)
        )
        # CWT 模拟
        self.cwt_conv = nn.Sequential(
            nn.Conv1d(1, cwt_channels, kernel_size=7, padding=3),
            nn.InstanceNorm1d(cwt_channels),
            nn.GELU(),
            nn.Conv1d(cwt_channels, cwt_channels, kernel_size=3, padding=1)
        )
        self.cwt_proj = nn.Linear(cwt_channels, d_model) # Optional projection

    def forward(self, x):
        # x: (B, L)
        x_fft = torch.fft.rfft(x, dim=-1)
        x_fft_feat = torch.cat([x_fft.real, x_fft.imag], dim=-1)
        v_fft = self.fft_mlp(x_fft_feat) 
        
        x_cwt = x.unsqueeze(1) 
        m_cwt = self.cwt_conv(x_cwt) # (B, C_cwt, L)
        return v_fft, m_cwt


class ALMR_Module(nn.Module):
    def __init__(self, d_model, seq_len, num_beats=16, patch_num=None): 
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.q_beat = nn.Parameter(torch.randn(num_beats, d_model))
        
        # Cross Attention: Q=Beat, K=Multi-Scale Context, V=Multi-Scale Context
        self.cross_attn = AttentionLayer(
            FullAttention(False, factor=1, attention_dropout=0.1, output_attention=True),
            d_model, n_heads=1
        )
        
        self.beat_decoder = nn.Sequential(nn.Linear(d_model, seq_len), nn.Tanh())
        
        # Injection Attention: Q=Multi-Scale Context, K=Beat, V=Beat
        self.injection_attn = AttentionLayer(
             FullAttention(False, factor=1, attention_dropout=0.1, output_attention=False),
             d_model, n_heads=4
        )
        self.rhythm_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x_embed_list, is_train=True):
        """
        x_embed_list: List of tensors [(B, N1, D), (B, N2, D), ...]
        """
        # 1. 记录每个尺度的长度，用于后续切分
        lengths = [x.shape[1] for x in x_embed_list]
        
        # 2. 【多粒度聚合】 Concatenate 形成统一的 Context 空间
        # (B, N_total, D) where N_total = sum(lengths)
        x_concat = torch.cat(x_embed_list, dim=1) 
        
        B, N_total, D = x_concat.shape
        
        # 3. Beat Aggregation (在统一空间中检索)
        q_beat_batch = self.q_beat.unsqueeze(0).expand(B, -1, -1) # (B, K, D)
        
        z_beat, attn_map = self.cross_attn(q_beat_batch, x_concat, x_concat, attn_mask=None)
        attn_map = attn_map.squeeze(1) # (B, K, N_total)
        
        x_recon = None
        if is_train:
            s_hat = self.beat_decoder(z_beat) # (B, K, L)
            
            # ALMR 重构逻辑调整：
            # attn_map 是对所有 patch 的权重。我们需要将其插值回原始信号长度 L。
            # 直接 interpolate (B, K, N_total) -> (B, K, L) 是数学上合理的，
            # 意味着模型学会了从混合尺度的权重映射到时间轴。
            attn_up = F.interpolate(attn_map, size=self.seq_len, mode='linear', align_corners=False)
            x_recon = torch.sum(s_hat * attn_up, dim=1) # (B, L)
            
        # 4. Feature Injection (广播回统一空间)
        # 将提取出的纯净 Beat 信息注入回混合序列
        x_injected, _ = self.injection_attn(x_concat, z_beat, z_beat, attn_mask=None)
        
        # Rhythm Injection (Global)
        rhythm = self.rhythm_pool(x_concat.transpose(1, 2)).transpose(1, 2) # (B, 1, D)
        
        # 融合
        x_enhanced_concat = x_concat + x_injected + rhythm
        
        # 5. 【多粒度分发】 Split 回原本的 List 结构
        # torch.split 返回一个 tuple，转回 list
        x_final_list = list(torch.split(x_enhanced_concat, lengths, dim=1))
        
        return x_final_list, x_recon


class AdaIN(nn.Module):
    """
    参考方案中的 AdaIN 实现：高效、简洁
    """
    def __init__(self, c, s):
        super().__init__()
        self.fc = nn.Linear(s, c * 2)
        
    def forward(self, x, style):
        # x: [B, C, L], style: [B, S]
        # chunk(2, 1) 在 dim=1 (channel) 上切分为 gamma 和 beta
        g, b = self.fc(style).unsqueeze(-1).chunk(2, 1)
        # 标准化 + 仿射变换
        mean = x.mean(2, keepdim=True)
        std = x.std(2, keepdim=True) + 1e-8
        return ((x - mean) / std) * g + b

class AdaINResBlock(nn.Module):
    """
    将 AdaIN 有机结合到残差块中，保证深层网络的梯度传播
    """
    def __init__(self, in_channels, out_channels, style_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # 嵌入参考代码风格的 AdaIN
        self.adain = AdaIN(out_channels, style_dim)
        
        self.act = nn.GELU()
        
        # Skip connection
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, style):
        resid = self.skip(x)
        
        x = self.conv1(x)
        x = self.act(x)
        
        # 使用 AdaIN 进行风格注入
        x = self.adain(x, style)
        
        x = self.conv2(x)
        return x + resid
    

class Decoder(nn.Module):
    def __init__(self, d_model, target_len, fft_dim, cwt_dim):
        super().__init__()
        self.target_len = target_len
        
        # 1. CWT 投影层 (参考代码思路)
        # 将双流 CWT (ECG+PPG) 投影到较小的维度，避免 concat 后通道过大
        # 输入维度: cwt_dim (单流通道) * 2
        self.cwt_proj = nn.Conv1d(cwt_dim * 2, d_model // 2, kernel_size=1)
        
        # 2. 初始处理
        # 假设输入是 Transformer 的 sequence output (B, N, D)
        # 我们先把它转为 (B, D, N) 并做一次卷积整理
        self.start_conv = nn.Conv1d(d_model, d_model, 1)

        # 3. 渐进式解码 Block (U-Net Style)
        # Stage 1: Upsample x2
        self.up1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        # Fusion 1: 输入 = 上一层(D) + CWT投影(D/2)
        self.fusion1 = nn.Conv1d(d_model + d_model // 2, d_model, 1)
        self.res1 = AdaINResBlock(d_model, d_model // 2, fft_dim) # 输出通道减半

        # Stage 2: Upsample x2
        self.up2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        # Fusion 2: 输入 = 上一层(D/2) + CWT投影(D/2)
        self.fusion2 = nn.Conv1d(d_model // 2 + d_model // 2, d_model // 2, 1)
        self.res2 = AdaINResBlock(d_model // 2, d_model // 4, fft_dim)

        # Stage 3: Upsample x2
        self.up3 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        # Fusion 3
        self.fusion3 = nn.Conv1d(d_model // 4 + d_model // 2, d_model // 4, 1)
        self.res3 = AdaINResBlock(d_model // 4, 32, fft_dim)

        # 4. 输出层
        self.out_conv = nn.Conv1d(32, 1, kernel_size=1)

    def forward(self, x, fft_style, cwt_ecg, cwt_ppg):
        """
        x: (B, N, D) - Transformer Latent
        fft_style: (B, FFT_Dim) - Global Style
        cwt_ecg, cwt_ppg: (B, C_cwt, L_orig) - Local Details
        """
        # 1. 准备 Latent
        x = x.transpose(1, 2) # (B, N, D) -> (B, D, N)
        x = self.start_conv(x)
        
        # 2. 准备 CWT (Feature Pyramid 思想)
        # 先拼接双流，再投影
        cwt_raw = torch.cat([cwt_ecg, cwt_ppg], dim=1) # (B, 2*C_cwt, L_orig)
        
        # 辅助函数：将 CWT 调整到当前 x 的尺寸并融合
        # 这里结合了参考代码的 "project" 和 "interpolate" 思路
        def fuse_cwt(curr_x, raw_cwt):
            B, C, N_curr = curr_x.shape
            # a. 将高分辨 CWT 下采样/插值到当前层级的分辨率 N_curr
            cwt_resized = F.interpolate(raw_cwt, size=N_curr, mode='linear', align_corners=False)
            # b. 投影减少通道 (Sharing projection weights or independent? 
            # 这里为了简单复用同一个 cwt_proj，注意 cwt_proj 是 1x1 conv，对分辨率不敏感)
            cwt_feat = self.cwt_proj(cwt_resized)
            # c. Concat
            return torch.cat([curr_x, cwt_feat], dim=1)

        # --- Block 1 ---
        x = self.up1(x)       # N -> 2N
        x = fuse_cwt(x, cwt_raw)
        x = self.fusion1(x)   # Reduce channels after concat
        x = self.res1(x, fft_style)

        # --- Block 2 ---
        x = self.up2(x)       # 2N -> 4N
        x = fuse_cwt(x, cwt_raw)
        x = self.fusion2(x)
        x = self.res2(x, fft_style)

        # --- Block 3 ---
        x = self.up3(x)       # 4N -> 8N (此时接近 target_len，如 624)
        x = fuse_cwt(x, cwt_raw)
        x = self.fusion3(x)
        x = self.res3(x, fft_style)

        x_final = F.interpolate(x, size=self.target_len, mode='linear', align_corners=False)
        
        out = self.out_conv(x_final).squeeze(1) # (B, L)
        
        return out


class Medformer(nn.Module):
    def __init__(self, configs):
        super(Medformer, self).__init__()
        self.configs = configs
        self.is_train = configs.is_train
        
        # Embedding & Feature Extraction
        patch_len_list = list(map(int, configs.patch_len_list.split(",")))
        self.ecg_embedding = ListPatchEmbedding(1, configs.d_model, patch_len_list, patch_len_list, configs.dropout, ["none"], True)
        self.ppg_embedding = ListPatchEmbedding(1, configs.d_model, patch_len_list, patch_len_list, configs.dropout, ["none"], True)
        self.freq_extractor = FreqFeatureExtractor(configs.seq_len, configs.d_model, configs.cwt_channels)
        
        # ALMR
        total_patches = sum([int((configs.seq_len - pl) / pl + 2) for pl in patch_len_list])
        self.tbr_ecg = ALMR_Module(configs.d_model, configs.seq_len, configs.num_beats, patch_num=total_patches)
        self.tbr_ppg = ALMR_Module(configs.d_model, configs.seq_len, configs.num_beats, patch_num=total_patches)
        
        self.ecg_encoder = Encoder(
            [EncoderLayer(
                MedformerLayer(len(patch_len_list), configs.d_model, configs.n_heads, configs.dropout),
                configs.d_model, configs.d_ff, dropout=configs.dropout, use_bp_book=True, num_prototypes=configs.num_prototypes, bp_book_topk=configs.topK
            ) for _ in range(configs.e_layers)],
            norm_layer=nn.LayerNorm(configs.d_model)
        )
        self.ppg_encoder = Encoder(
            [EncoderLayer(
                MedformerLayer(len(patch_len_list), configs.d_model, configs.n_heads, configs.dropout),
                configs.d_model, configs.d_ff, dropout=configs.dropout, use_bp_book=True, num_prototypes=configs.num_prototypes, bp_book_topk=configs.topK
            ) for _ in range(configs.e_layers)],
            norm_layer=nn.LayerNorm(configs.d_model)
        )
        
        # Fusion
        self.cross_ecg_to_ppg = AttentionLayer(FullAttention(False), configs.d_model, configs.n_heads)
        self.cross_ppg_to_ecg = AttentionLayer(FullAttention(False), configs.d_model, configs.n_heads)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(configs.d_model * 4, configs.d_model),
            nn.LayerNorm(configs.d_model),
            nn.GELU()
        )
        
        # Decoder 
        self.decoder = Decoder(configs.d_model, configs.seq_len, configs.d_model*2, cwt_dim=configs.cwt_channels)

    def forward(self, x_enc):
        x_ecg = x_enc[:, :, 0] 
        x_ppg = x_enc[:, :, 1]
        
        # 1. Freq Features (FFT for Style, CWT for Details)
        v_fft_ecg, m_cwt_ecg = self.freq_extractor(x_ecg)
        v_fft_ppg, m_cwt_ppg = self.freq_extractor(x_ppg)
        v_style = torch.cat([v_fft_ecg, v_fft_ppg], dim=-1) # (B, 2D)
        
        # 2. Embedding
        h_ecg = self.ecg_embedding(x_ecg.unsqueeze(-1))
        h_ppg = self.ppg_embedding(x_ppg.unsqueeze(-1))
        
        h_ecg, recon_ecg = self.tbr_ecg(h_ecg, self.is_train)
        h_ppg, recon_ppg = self.tbr_ppg(h_ppg, self.is_train)
        
        # 3. Encoder (BP-Book inside)
        h_ecg_enc, _ = self.ecg_encoder(h_ecg)
        h_ppg_enc, _ = self.ppg_encoder(h_ppg)
        
        # 4. Cross Fusion
        h_ecg_to_ppg, _ = self.cross_ecg_to_ppg(h_ecg_enc, h_ppg_enc, h_ppg_enc, attn_mask=None)
        h_ppg_to_ecg, _ = self.cross_ppg_to_ecg(h_ppg_enc, h_ecg_enc, h_ecg_enc, attn_mask=None)
        
        h_fused = torch.cat([h_ecg_enc, h_ppg_enc, h_ecg_to_ppg, h_ppg_to_ecg], dim=-1)
        h_fused = self.fusion_mlp(h_fused) # (B, N, D)
        
        # 5. Decoder (with CWT injection)
        y_pred = self.decoder(h_fused, v_style, m_cwt_ecg, m_cwt_ppg)
        
        if self.is_train:
            return y_pred, recon_ecg, recon_ppg
        else:
            return y_pred