import torch
import torch.nn as nn
import torch.nn.functional as F
from .TFCPR_EncDec import EncoderLayer, Encoder
from .SelfAttention_Family import TFCPRLayer, FullAttention, AttentionLayer
from .Embed import ListPatchEmbedding
import math

class MultiScaleConv(nn.Module):
    """
    [SOTA] 可学习的 Morlet 小波变换层
    不使用随机初始化的 Conv1d, 而是基于物理公式生成 Morlet 小波核。
    允许模型微调小波的形状，以适应具体的血压任务。
    """
    def __init__(self, in_channels, out_channels, kernel_size=63):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # 1. 初始化可学习的尺度参数 (Scales)
        scales = torch.logspace(0, math.log10(kernel_size // 2), steps=out_channels)
        self.scales = nn.Parameter(scales)
        
        # 2. 1x1 卷积用于通道融合 (代替你的 fusion)
        self.fusion = nn.Conv1d(in_channels * out_channels, out_channels, 1)

    def _generate_morlet_filters(self):
        """
        动态生成 Morlet 小波卷积核
        Formula: Psi(t) = C * exp(-t^2 / 2) * cos(5t)  (简化版实数 Morlet)
        """
        device = self.scales.device
        
        # 创建时间轴 [-1, 1]
        t = torch.linspace(-1, 1, self.kernel_size).to(device)
        t = t.view(1, 1, -1)  # [1, 1, L]
        
        # 扩展 scales: [Out, 1, 1]
        s = self.scales.view(-1, 1, 1)
        
        # 核心：根据当前 Scale 动态计算波形
        # Scale 越小，频率越高，波形越窄
        # 这里的 5.0 是 Morlet 的基准频率，可以设为可学习参数
        envelope = torch.exp(- (t ** 2) / (2 * (s ** 2) + 1e-6))
        oscillation = torch.cos(5.0 * t / (s + 1e-6))
        
        filters = envelope * oscillation  # [Out, 1, Kernel]
        
        # 归一化能量，防止梯度爆炸
        filters = filters / (torch.norm(filters, dim=-1, keepdim=True) + 1e-6)
        
        return filters

    def forward(self, x):
        # x: [B, 1, L] or [B, C, L]
        B, C, L = x.shape
        
        # 1. 动态生成滤波器
        filters = self._generate_morlet_filters() # [Out, 1, K]
        
        # 2. 这里的卷积相当于 CWT
        
        # 为了适配 Conv1d 的权重格式 [Out_channels, In_channels/Groups, K]
        # 假设输入是 [B, 1, L]，输出想要 [B, 32, L]
        filters = filters.view(self.out_channels, 1, self.kernel_size)
        
        # 执行卷积
        # 输出形状: [B, Out, L]
        out = F.conv1d(x, filters, padding=self.padding)
        
        # 3. 激活与融合
        out = F.gelu(out)
        
        # 如果需要更复杂的融合，可以在这里加
        return out


class FreqFeatureExtractor(nn.Module):
    def __init__(self, seq_len, d_model=128, cwt_channels=32, num_freq_patches=16):
        super().__init__()
        
        fft_len = seq_len // 2 + 1
        self.input_dim = fft_len * 2
        
        self.patch_conv = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=16, stride=8), # 模拟 Patching，提取局部频谱特征
            nn.BatchNorm1d(128),
            nn.GELU()
        )
        
        self.attention_mlp = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),     
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 128),          
            nn.Sigmoid()                 
        )
        
        self.out_proj = nn.Linear(128, d_model)
        self.ms_conv = MultiScaleConv(1, cwt_channels)

    def forward(self, x_raw):
        """
        x: [B, 1, L] (归一化后的数据，给 CWT 用)
        x_raw: [B, 1, L] (原始数据，给 FFT 用，保留幅值能量)
        """
        B = x_raw.shape[0]
        
        # --- Part A: FFT Path (Frequency Attention) ---
        x_fft = torch.fft.rfft(x_raw.squeeze(1), dim=-1)
        x_fft_mag = x_fft.abs() # [B, 1, L//2+1]
        x_fft_mag = x_fft_mag.unsqueeze(1)
        
        fft_feats = self.patch_conv(x_fft_mag) 
        attn_weights = self.attention_mlp(fft_feats)
        fft_weighted = fft_feats * attn_weights.unsqueeze(-1)
        
        v_fft = fft_weighted.mean(dim=-1) # [B, 128]
        v_fft = self.out_proj(v_fft)      # [B, d_model]
        
        # --- Part B: CWT Path (保持不变) ---
        m_local = self.ms_conv(x_raw.unsqueeze(1)) # [B, Cwt_Channels, L]
        
        return v_fft, m_local


class ALMR_Module(nn.Module):
    def __init__(self, d_model, raw_seq_len=1250, num_beats=16):
        super().__init__()
        self.d_model = d_model
        self.num_beats = num_beats
        self.raw_seq_len = raw_seq_len
        
        self.q_beat = nn.Parameter(torch.randn(1, num_beats, d_model))
        
        self.attn_token2beat = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.attn_beat2token = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        
        # 定义一组固定的“时间锚点”，代表原始信号的每一个采样点
        # Shape: [1, Raw_Len, D]
        # 使用 Sinusoidal 位置编码初始化，代表绝对时间位置
        self.time_queries = nn.Parameter(self._init_positional_encoding(raw_seq_len, d_model), requires_grad=False)
        
        # Beat -> Signal 的解码注意力
        self.recon_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.out_proj = nn.Linear(d_model, 1)

    def _init_positional_encoding(self, length, d_model):
        """生成标准的时间位置编码作为查询锚点"""
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0) # [1, L, D]

    def forward(self, x_token, is_train=True):
        """
        x_token: [B, N_mixed, D] (由多粒度拼接而成的 Token，N 大小不定)
        x_raw:   [B, L] (原始信号)
        """
        B, N, D = x_token.shape
        
        q = self.q_beat.repeat(B, 1, 1) # [B, K, D]
        z_beat, _ = self.attn_token2beat(q, x_token, x_token)
        
        # --- 2. Feature Injection (回注增强) ---
        context, _ = self.attn_beat2token(x_token, z_beat, z_beat)
        x_enhanced = x_token + context
        
        # --- 3. Coordinate-Based Reconstruction (坐标解码) ---
        
        if is_train :
            # 这里的逻辑是：用"时间坐标"去查询"Beat特征"
            # 只有当 Beat 特征真正包含了完整的波形形态时，它才能正确回答每个时间点的数值
            # Q: [B, Raw_Len, D] (时间坐标)
            time_q = self.time_queries.repeat(B, 1, 1).to(x_token.device)
            
            # K, V: [B, K, D] (Beat 特征)
            # 输出: [B, Raw_Len, D]
            recon_feat, _ = self.recon_attn(time_q, z_beat, z_beat)
            
            # 映射回波形数值: [B, Raw_Len, 1] -> [B, Raw_Len]
            x_recon = self.out_proj(recon_feat).squeeze(-1)
            
        return x_enhanced, x_recon


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


class Decoder(nn.Module):
    def __init__(self, d_model, target_len, fft_dim, cwt_dim):
        super().__init__()
        self.target_len = target_len
        self.cwt_proj = nn.Conv1d(cwt_dim * 2, d_model // 2, 1) # ECG+PPG
        self.start_conv = nn.Conv1d(d_model, d_model, 1)

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

        # (B, 2C, L)
        raw_cwt = torch.cat([cwt_ecg, cwt_ppg], dim=1)
        
        def get_skip(target_len):
            c = F.interpolate(raw_cwt, size=target_len, mode='linear', align_corners=False)
            return self.cwt_proj(c)

        x = self.up1(x)
        skip = get_skip(x.shape[-1])
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, fft_style)

        x = self.up2(x)
        skip = get_skip(x.shape[-1])
        x = torch.cat([x, skip], dim=1)
        x = self.block2(x, fft_style)
        
        x = self.up3(x)
        skip = get_skip(x.shape[-1])
        x = torch.cat([x, skip], dim=1)
        x = self.block3(x, fft_style)
        
        x = F.interpolate(x, size=self.target_len, mode='linear')
        return self.out(x).squeeze(1)


class JointBPBook(nn.Module):
    """
    联合生理状态字典
    存储 (ECG特征 + PPG特征 + PTT关系) 的联合高维原型
    """
    def __init__(self, num_slots=2048, d_model=128, topk=5):
        super().__init__()
        self.d_model = d_model
        self.topk = topk
        
        self.memory = nn.Parameter(torch.randn(num_slots, d_model))
        self.retrieval_scale = nn.Parameter(torch.tensor(1.0))
        
        self.query_proj = nn.Linear(d_model, d_model)

    def forward(self, x_fused):
        """
        x_fused: [B, N, D] or [B, D] (如果是全局检索)
        """
        if x_fused.dim() == 3:
            query = x_fused.mean(dim=1) 
        else:
            query = x_fused
            
        query = self.query_proj(query)
        
        # 1. 归一化 & 相似度
        q_norm = F.normalize(query, dim=-1)
        m_norm = F.normalize(self.memory, dim=-1)
        sim = torch.matmul(q_norm, m_norm.t()) # [B, Slots]
        
        # 2. Top-K 检索
        scores, indices = torch.topk(sim, self.topk, dim=-1)
        weights = F.softmax(scores, dim=-1)
        
        # 3. 聚合原型
        # indices: [B, K] -> [B*K]
        flat_indices = indices.view(-1)
        retrieved = F.embedding(flat_indices, self.memory).view(*indices.shape, -1) # [B, K, D]
        
        # [B, K, 1] * [B, K, D] -> sum -> [B, D]
        prototype = (retrieved * weights.unsqueeze(-1)).sum(dim=1)
        
        # 4. 广播并注入回序列
        # [B, N, D] + [B, 1, D]
        if x_fused.dim() == 3:
            return x_fused + self.retrieval_scale * prototype.unsqueeze(1)
        else:
            return x_fused + self.retrieval_scale * prototype


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
        # self.tbr_ecg = ALMR_Module(configs.d_model, configs.seq_len, configs.num_beats)
        # self.tbr_ppg = ALMR_Module(configs.d_model, configs.seq_len, configs.num_beats)
        
        self.ecg_encoder = Encoder(
            [EncoderLayer(
                TFCPRLayer(len(patch_len_list), configs.d_model, configs.n_heads, configs.dropout),
                configs.d_model, configs.d_ff, dropout=configs.dropout) for _ in range(configs.e_layers)],
            norm_layer=nn.LayerNorm(configs.d_model)
        )
        self.ppg_encoder = Encoder(
            [EncoderLayer(
                TFCPRLayer(len(patch_len_list), configs.d_model, configs.n_heads, configs.dropout),
                configs.d_model, configs.d_ff, dropout=configs.dropout) for _ in range(configs.e_layers)],
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
        self.joint_bp_book = JointBPBook(num_slots=configs.num_slots, d_model=configs.d_model, topk=configs.topk)
        
        self.decoder = Decoder(configs.d_model, configs.seq_len, configs.d_model*2, cwt_dim=configs.cwt_channels)
        self.sbp_head = nn.Linear(configs.d_model, 1)
        self.dbp_head = nn.Linear(configs.d_model, 1)


    def forward(self, x_enc):
        x_ecg = x_enc[:, :, 0] 
        x_ppg = x_enc[:, :, 1]
        x_ecg_raw = x_enc[:, :, 2]
        x_ppg_raw = x_enc[:, :, 3]
        
        # 1. Freq Features (FFT for Style, CWT for Details)
        v_fft_ecg, m_cwt_ecg = self.freq_extractor(x_ecg_raw)
        v_fft_ppg, m_cwt_ppg = self.freq_extractor(x_ppg_raw)
        v_style = torch.cat([v_fft_ecg, v_fft_ppg], dim=-1) # (B, 2D)
        
        # 2. Embedding
        h_ecg = self.ecg_embedding(x_ecg.unsqueeze(-1))
        h_ppg = self.ppg_embedding(x_ppg.unsqueeze(-1))

        # 3. Encoder
        h_ecg_enc, _ = self.ecg_encoder(h_ecg)
        h_ppg_enc, _ = self.ppg_encoder(h_ppg)

        # h_ecg_enc, recon_ecg = self.tbr_ecg(h_ecg_enc, self.is_train)
        # h_ppg_enc, recon_ppg = self.tbr_ppg(h_ppg_enc, self.is_train)
        
        # 4. Cross Fusion
        h_ecg_to_ppg, _ = self.cross_ecg_to_ppg(h_ecg_enc, h_ppg_enc, h_ppg_enc, attn_mask=None)
        h_ppg_to_ecg, _ = self.cross_ppg_to_ecg(h_ppg_enc, h_ecg_enc, h_ecg_enc, attn_mask=None)
        
        h_fused = torch.cat([h_ecg_enc, h_ppg_enc, h_ecg_to_ppg, h_ppg_to_ecg], dim=-1)
        h_fused = self.fusion_mlp(h_fused) # (B, N, D)
        fused_calibrated = self.joint_bp_book(h_fused)
        
        # 5. Decoder 
        abp_pred = self.decoder(fused_calibrated, v_style, m_cwt_ecg, m_cwt_ppg)
        feat_global = fused_calibrated.mean(dim=1)
        pred_sbp = self.sbp_head(feat_global)
        pred_dbp = self.dbp_head(feat_global) 

        if self.is_train:
            # return abp_pred, recon_ecg, recon_ppg, pred_sbp, pred_dbp
            return abp_pred, None, None, pred_sbp, pred_dbp
        else:
            return abp_pred