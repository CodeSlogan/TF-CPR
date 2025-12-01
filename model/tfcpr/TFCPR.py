import torch
import torch.nn as nn
import torch.nn.functional as F
from .TFCPR_EncDec import EncoderLayer, Encoder
from .SelfAttention_Family import TFCPRLayer, FullAttention, AttentionLayer
from .Embed import ListPatchEmbedding


# ---------------------------------------------------------
# 1. 改进的特征调制层 (替代 AdaIN)
# ---------------------------------------------------------
class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation: 不破坏原始特征的统计分布(Mean/Std)
    而是基于 Style 进行仿射变换。保留了血压相关的幅度信息。
    """
    def __init__(self, in_channels, style_dim):
        super().__init__()
        self.scale_fc = nn.Linear(style_dim, in_channels)
        self.shift_fc = nn.Linear(style_dim, in_channels)

    def forward(self, x, style):
        # x: (B, C, L)
        # style: (B, S)
        scale = self.scale_fc(style).unsqueeze(-1) # (B, C, 1)
        shift = self.shift_fc(style).unsqueeze(-1)
        return x * (1 + scale) + shift


class ResBlockWithFiLM(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.film = FiLM(out_channels, style_dim) # 替换 AdaIN
        self.act = nn.GELU()
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, style):
        resid = self.skip(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.film(x, style) # 注入频域 Style
        x = self.conv2(x)
        return x + resid

# ---------------------------------------------------------
# 2. 真正的多尺度局部特征提取 (替代伪 CWT)
# ---------------------------------------------------------
class MultiScaleConv(nn.Module):
    """
    模拟 CWT/STFT 的多分辨率特性：使用不同 Kernel Size 并行提取
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 小卷积核捕获高频突变 (QRS)，大卷积核捕获低频趋势 (P/T wave)
        self.branch1 = nn.Conv1d(in_channels, out_channels//2, kernel_size=3, padding=1)
        self.branch2 = nn.Conv1d(in_channels, out_channels//2, kernel_size=11, padding=5)
        self.fusion = nn.Conv1d(out_channels, out_channels, 1)

    def forward(self, x):
        # x: (B, 1, L)
        # 增加 Channel 维度如果输入是 (B, L)
        if x.dim() == 2: x = x.unsqueeze(1)
        
        b1 = F.gelu(self.branch1(x))
        b2 = F.gelu(self.branch2(x))
        out = torch.cat([b1, b2], dim=1)
        return self.fusion(out)

class FreqFeatureExtractor(nn.Module):
    def __init__(self, seq_len, d_model, cwt_channels=32):
        super().__init__()
        self.fft_mlp = nn.Sequential(
            nn.Linear((seq_len // 2 + 1) * 2, 128),
            nn.ReLU(),
            nn.Linear(128, d_model)
        )
        # 替换为多尺度卷积
        self.ms_conv = MultiScaleConv(1, cwt_channels)
        
    def forward(self, x):
        # FFT 部分保持不变
        x_fft = torch.fft.rfft(x, dim=-1)
        x_fft_feat = torch.cat([x_fft.real, x_fft.imag], dim=-1)
        v_fft = self.fft_mlp(x_fft_feat) 
        
        # 多尺度卷积部分
        m_local = self.ms_conv(x) # (B, C_cwt, L)
        return v_fft, m_local

# ---------------------------------------------------------
# 3. 简化的 ALMR (修复重构逻辑)
# ---------------------------------------------------------
class ALMR_Module(nn.Module):
    def __init__(self, d_model, seq_len, num_beats=16):
        super().__init__()
        self.d_model = d_model
        
        self.q_beat = nn.Parameter(torch.randn(num_beats, d_model))
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        
        self.recon_head = nn.Sequential(
            nn.Linear(d_model, seq_len // num_beats * 2), 
            nn.GELU(),
            nn.Linear(seq_len // num_beats * 2, seq_len // num_beats) 
        )
        self.final_recon_proj = nn.Linear(num_beats * (seq_len // num_beats), seq_len)

    def forward(self, x_embed_list, is_train=True):
        # 1. Concat
        x_concat = torch.cat(x_embed_list, dim=1) # (B, N, D)
        B, N, D = x_concat.shape

        # 2. Beat Extraction
        q = self.q_beat.unsqueeze(0).expand(B, -1, -1) # (B, K, D)
        z_beat, _ = self.cross_attn(q, x_concat, x_concat) # (B, K, D)
        
        x_recon = None
        if is_train:
            local_sig = self.recon_head(z_beat) # (B, K, Sub_L)
            flat_sig = local_sig.view(B, -1)
            # 调整长度回 seq_len
            x_recon = self.final_recon_proj(flat_sig) 

        beat_global = z_beat.mean(dim=1, keepdim=True) # (B, 1, D)
        x_enhanced = x_concat + beat_global
        
        # Split back
        lengths = [x.shape[1] for x in x_embed_list]
        x_final_list = list(torch.split(x_enhanced, lengths, dim=1))
        
        return x_final_list, x_recon


class Decoder(nn.Module):
    def __init__(self, d_model, target_len, fft_dim, cwt_dim):
        super().__init__()
        self.target_len = target_len
        self.cwt_proj = nn.Conv1d(cwt_dim * 2, d_model // 2, 1) # ECG+PPG
        self.start_conv = nn.Conv1d(d_model, d_model, 1)

        # 3层 U-Net，通道数递减
        self.up1 = nn.Upsample(scale_factor=2, mode='linear')
        self.block1 = ResBlockWithFiLM(d_model + d_model//2, d_model, fft_dim)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='linear')
        self.block2 = ResBlockWithFiLM(d_model + d_model//2, d_model//2, fft_dim)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='linear')
        self.block3 = ResBlockWithFiLM(d_model//2 + d_model//2, d_model//4, fft_dim)
        
        self.out = nn.Conv1d(d_model//4, 1, 1)

    def forward(self, x, fft_style, cwt_ecg, cwt_ppg):
        # x: (B, N, D)
        x = x.transpose(1, 2)
        x = self.start_conv(x) # (B, D, N)

        # 处理 CWT: (B, 2C, L)
        raw_cwt = torch.cat([cwt_ecg, cwt_ppg], dim=1)
        
        def get_skip(target_len):
            # 将 CWT 缩放到当前层级并投影
            c = F.interpolate(raw_cwt, size=target_len, mode='linear', align_corners=False)
            return self.cwt_proj(c)

        # Stage 1
        x = self.up1(x)
        skip = get_skip(x.shape[-1])
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, fft_style)

        # Stage 2
        x = self.up2(x)
        skip = get_skip(x.shape[-1])
        x = torch.cat([x, skip], dim=1)
        x = self.block2(x, fft_style)
        
        # Stage 3
        x = self.up3(x)
        skip = get_skip(x.shape[-1])
        x = torch.cat([x, skip], dim=1)
        x = self.block3(x, fft_style)
        
        # Final
        x = F.interpolate(x, size=self.target_len, mode='linear')
        return self.out(x).squeeze(1)


class TFCPR(nn.Module):
    def __init__(self, configs):
        super(TFCPR, self).__init__()
        self.configs = configs
        self.is_train = configs.is_train
        
        # Embedding & Feature Extraction
        patch_len_list = list(map(int, configs.patch_len_list.split(",")))
        self.ecg_embedding = ListPatchEmbedding(1, configs.d_model, patch_len_list, patch_len_list, configs.dropout, ["none"], True)
        self.ppg_embedding = ListPatchEmbedding(1, configs.d_model, patch_len_list, patch_len_list, configs.dropout, ["none"], True)
        self.freq_extractor = FreqFeatureExtractor(configs.seq_len, configs.d_model, configs.cwt_channels)
        
        # ALMR
        total_patches = sum([int((configs.seq_len - pl) / pl + 2) for pl in patch_len_list])
        self.tbr_ecg = ALMR_Module(configs.d_model, configs.seq_len, configs.num_beats)
        self.tbr_ppg = ALMR_Module(configs.d_model, configs.seq_len, configs.num_beats)
        
        self.ecg_encoder = Encoder(
            [EncoderLayer(
                TFCPRLayer(len(patch_len_list), configs.d_model, configs.n_heads, configs.dropout),
                configs.d_model, configs.d_ff, dropout=configs.dropout, use_bp_book=False, num_prototypes=configs.num_prototypes, bp_book_topk=configs.topK
            ) for _ in range(configs.e_layers)],
            norm_layer=nn.LayerNorm(configs.d_model)
        )
        self.ppg_encoder = Encoder(
            [EncoderLayer(
                TFCPRLayer(len(patch_len_list), configs.d_model, configs.n_heads, configs.dropout),
                configs.d_model, configs.d_ff, dropout=configs.dropout, use_bp_book=False, num_prototypes=configs.num_prototypes, bp_book_topk=configs.topK
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

        self.scalar_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),      # (B, N, D) -> (B, D, 1) 全局池化
            nn.Flatten(),                 # (B, D)
            nn.Linear(configs.d_model, 128),
            nn.LayerNorm(128),            # 加个 Norm 稳一点
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(128, 2)             # [Mean, Std]
        )

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
        pred_shape = self.decoder(h_fused, v_style, m_cwt_ecg, m_cwt_ppg) 
        
        # ---------------------------------------------------
        # Stage 2: 数值预测 (Scalar Prediction)
        # ---------------------------------------------------
        pred_stats = self.scalar_head(h_fused.transpose(1, 2)) # (B, 2)
        
        pred_mean = pred_stats[:, 0].unsqueeze(1) # (B, 1)
        pred_std  = pred_stats[:, 1].unsqueeze(1) # (B, 1)
        pred_std = F.softplus(pred_std)
        # ---------------------------------------------------
        # Reconstruction: 物理重构
        # ---------------------------------------------------
        # 最终 ABP = Shape * Std + Mean
        # 这一步是可导的，梯度会分别传给 Decoder 和 ScalarHead
        pred_abp_final = pred_shape * pred_std + pred_mean

        if self.is_train:
            # 返回: 最终ABP, 归一化形状, 预测的统计量
            return pred_abp_final, pred_shape, pred_stats, recon_ecg, recon_ppg
        else:
            return pred_abp_final