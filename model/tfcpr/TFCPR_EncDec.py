import torch
import torch.nn as nn
import torch.nn.functional as F

class BPBookLayer(nn.Module):
    """
    阶段 III: BP-Book 检索与校准 (The Interaction Point)
    修正版: 
    1. 增加 Top-K 检索
    2. 使用全局特征 (Global Pooling) 进行匹配
    """
    def __init__(self, d_model, num_prototypes=1024, topk=5, alpha=0.1):
        super(BPBookLayer, self).__init__()
        self.d_model = d_model
        self.topk = topk
        self.alpha = alpha
        
        # Memory Bank: 存储通用 ECG/PPG 模式 (Bases)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, d_model))
        
    def forward(self, x):
        # x: [Batch, Seq_Len, D_model]
        B, L, D = x.shape
        
        query = torch.mean(x, dim=1) # (B, D)
        
        query_norm = F.normalize(query, p=2, dim=-1)   # (B, D)
        proto_norm = F.normalize(self.prototypes, p=2, dim=-1) # (K, D)
        
        scores = torch.matmul(query_norm, proto_norm.t())
        topk_scores, topk_indices = torch.topk(scores, self.topk, dim=-1)
        attn_weights = F.softmax(topk_scores, dim=-1).unsqueeze(-1) # (B, topk, 1)
        
        expanded_protos = self.prototypes.unsqueeze(0).expand(B, -1, -1)
        gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, D)
        
        retrieved_topk_protos = torch.gather(expanded_protos, 1, gather_indices) # (B, topk, D)
        
        proto_agg = torch.sum(attn_weights * retrieved_topk_protos, dim=1)
        out = x + self.alpha * proto_agg.unsqueeze(1)
        
        return out

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff, dropout, activation="relu", use_bp_book=True, num_prototypes=1024, bp_book_topk=5):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        
        # 修改点：集成 BP-Book
        self.use_bp_book = use_bp_book
        if self.use_bp_book:
            self.bp_book = BPBookLayer(d_model, num_prototypes, bp_book_topk)

    def forward(self, x):
        new_x, attn = self.attention(x)
        x = [_x + self.dropout(_nx) for _x, _nx in zip(x, new_x)]

        y = x = [self.norm1(_x) for _x in x]
        
        # 修改点：在 FFN 之前应用 BP-Book (对每个 patch head 的输出做处理，或者先 concat)
        # 注意：TFCPR 的输入 x 是 list [patch_len_1, patch_len_2...]
        # BP-Book 最好作用在融合后的语义上，但为了不破坏 EncoderLayer 结构，我们对每个尺度的 x 独立应用
        if self.use_bp_book:
            y = [self.bp_book(_y) for _y in y]
            
        y = [self.dropout(self.activation(self.conv1(_y.transpose(-1, 1)))) for _y in y]
        y = [self.dropout(self.conv2(_y).transpose(-1, 1)) for _y in y]

        return [self.norm2(_x + _y) for _x, _y in zip(x, y)], attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x)
            attns.append(attn)
        
        # concat logic
        x = torch.cat(x, dim=1) 

        if self.norm is not None:
            x = self.norm(x)

        return x, attns