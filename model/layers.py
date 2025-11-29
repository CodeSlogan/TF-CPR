import torch
import torch.nn as nn
import torch.nn.functional as F

class BPBookMemory(nn.Module):
    def __init__(self, num_slots=1024, d_model=128, topk=5):
        super().__init__()
        self.d_model = d_model
        self.topk = topk
        self.memory = nn.Parameter(torch.randn(num_slots, d_model))
        self.retrieval_scale = nn.Parameter(torch.tensor(0.05))

        # --- 新增：用于全局特征提取的 CNN 模块 ---
        # 目的：在平均前，先用 learnable filters 提取高级时序模式
        self.global_extractor = nn.Sequential(
            # Conv1d 必须以 [B, C, L] 形式输入，所以 C=D_model, L=N
            # 简化起见，我们使用一个 1x1 卷积和非线性层，用于特征压缩和融合
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.GELU(),
            # 可以根据需要添加更多层或使用更大的 kernel size (e.g., 3)
        )

    def forward(self, x):
        """
        x: [Batch, N, D_Model]
        """
        B, N, D = x.shape
        
        x_conv = x.transpose(1, 2) # [B, D, N]
        x_feat = self.global_extractor(x_conv)
        
        # 2. Global Average Pooling (得到全局 Query)
        # [B, D, N] -> [B, D]
        x_global_query = x_feat.mean(dim=2) 
        
        # 3. 归一化并计算相似度 (后续步骤与之前一致)
        x_norm = F.normalize(x_global_query, dim=-1)
        mem_norm = F.normalize(self.memory, dim=-1)
        
        sim_matrix = torch.matmul(x_norm, mem_norm.t()) # [B, Slots]
        
        # 4. 检索、加权聚合
        topk_scores, topk_indices = torch.topk(sim_matrix, self.topk, dim=-1)
        attention_weights = F.softmax(topk_scores, dim=-1) # [B, K]
        
        flat_indices = topk_indices.view(-1)
        retrieved_vectors = F.embedding(flat_indices, self.memory).view(B, self.topk, D)
        
        global_prototype = (retrieved_vectors * attention_weights.unsqueeze(-1)).sum(dim=1) # [B, D]
        
        # 5. 广播残差连接
        return x + self.retrieval_scale * global_prototype.unsqueeze(1)


class TransformerBPBookLayer(nn.Module):
    """
    [自定义 Transformer Layer]
    结构: Self-Attn -> Add&Norm -> [BP-Book] -> FFN -> Add&Norm
    """
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        # 1. Multi-Head Self Attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        # 2. BP-Book (嵌入在 FFN 之前)
        self.bp_book = BPBookMemory(num_slots=1024, d_model=d_model, topk=5)
        
        # 3. Feed Forward Network (FFN)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.GELU()

        # Norms & Dropouts
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # src: [B, Seq, D]
        
        # --- Part 1: Self-Attention (Pre-Norm) ---
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2)
        src = src + self.dropout1(src2) # Residual
        
        # --- Part 2: BP-Book Retrieval ---
        src = self.bp_book(src)
        
        # --- Part 3: FFN ---
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2) # Residual
        
        return src

class Transformer_BPEncoder(nn.Module):
    """
    堆叠多个 TransformerBPBookLayer
    """
    def __init__(self, d_model=128, num_layers=2, nhead=4, dim_feedforward=512, dropout=0.1): 
        super().__init__()
        self.layers = nn.ModuleList([
            # 将参数透传给 Layer
            TransformerBPBookLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout) 
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x