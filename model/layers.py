import torch
import torch.nn as nn
import torch.nn.functional as F

class BPBookMemory(nn.Module):
    """
    (Memory Bank) - Key Component.
    Query -> Cosine Sim -> TopK -> Weighted Sum -> Residual Add
    """
    def __init__(self, num_slots=1024, d_model=128, topk=5):
        super().__init__()
        self.d_model = d_model
        self.topk = topk
        
        # 初始化为正态分布，随网络一起训练更新
        self.memory = nn.Parameter(torch.randn(num_slots, d_model))
        
        # 可选：一个可学习的缩放因子，控制检索信息融入的强度
        self.retrieval_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        """
        x: [Batch, Seq_Len, D_Model]
        """
        B, N, D = x.shape
        
        # 1. 归一化以计算余弦相似度
        x_norm = F.normalize(x, dim=-1)
        mem_norm = F.normalize(self.memory, dim=-1)
        
        # 2. 计算相似度矩阵: [B, N, Slots]
        sim_matrix = torch.matmul(x_norm, mem_norm.t())
        
        # 3. Top-K 检索
        topk_scores, topk_indices = torch.topk(sim_matrix, self.topk, dim=-1)
        
        # 4. 重新归一化 TopK 分数作为权重 (Softmax)
        attention_weights = F.softmax(topk_scores, dim=-1) # [B, N, K]
        
        # 5. 从 Memory 中 gather 向量
        flat_indices = topk_indices.view(-1)
        retrieved_vectors = F.embedding(flat_indices, self.memory)
        retrieved_vectors = retrieved_vectors.view(B, N, self.topk, D)
        
        # 6. 加权聚合
        weighted_retrieval = (retrieved_vectors * attention_weights.unsqueeze(-1)).sum(dim=2)
        
        # 7. 残差连接
        return x + self.retrieval_scale * weighted_retrieval


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