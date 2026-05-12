import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttnBlock(nn.Module):
    """标准 Transformer 子块（Pre-LN）：
    1) 自注意力 + 残差
    2) 前馈网络 + 残差

    该模块用于分别建模 local 序列和 global 序列内部的时序/结构依赖。
    """

    def __init__(self, hidden_dim, nhead=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention residual: x <- x + MHA(LN(x))
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + self.dropout1(attn_out)

        # FFN residual: x <- x + FFN(LN(x))
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x


class BidirectionalCrossAttnBlock(nn.Module):
    """双向 Cross-Attention 融合块（无门控，全部残差）：
    1) local 查询 global（local <- global）
    2) global 查询 local（global <- local）
    3) 两侧各自再接一层 FFN 残差

    这样可以在保持 local/global 各自表达的同时，显式注入跨域交互信息。
    """

    def __init__(self, hidden_dim, nhead=8, dropout=0.1):
        super().__init__()
        self.local_norm_q = nn.LayerNorm(hidden_dim)
        self.global_norm_kv = nn.LayerNorm(hidden_dim)
        self.cross_l2g = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout_l2g = nn.Dropout(dropout)

        self.global_norm_q = nn.LayerNorm(hidden_dim)
        self.local_norm_kv = nn.LayerNorm(hidden_dim)
        self.cross_g2l = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout_g2l = nn.Dropout(dropout)

        self.local_ffn_norm = nn.LayerNorm(hidden_dim)
        self.local_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.local_ffn_dropout = nn.Dropout(dropout)

        self.global_ffn_norm = nn.LayerNorm(hidden_dim)
        self.global_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.global_ffn_dropout = nn.Dropout(dropout)

    def forward(self, local_x, global_x):
        # local <- global
        q_local = self.local_norm_q(local_x)
        kv_global = self.global_norm_kv(global_x)
        local_cross, _ = self.cross_l2g(q_local, kv_global, kv_global, need_weights=False)
        local_x = local_x + self.dropout_l2g(local_cross)

        # global <- local
        q_global = self.global_norm_q(global_x)
        kv_local = self.local_norm_kv(local_x)
        global_cross, _ = self.cross_g2l(q_global, kv_local, kv_local, need_weights=False)
        global_x = global_x + self.dropout_g2l(global_cross)

        # 两侧各自 FFN 残差
        local_x = local_x + self.local_ffn_dropout(self.local_ffn(self.local_ffn_norm(local_x)))
        global_x = global_x + self.global_ffn_dropout(self.global_ffn(self.global_ffn_norm(global_x)))
        return local_x, global_x


class HyperLoRAGenerator(nn.Module):
    """旧路径的 LoRA 参数直生模型（非扩散）：
    - 输入：global_cycles, local_cycles
    - 输出：A_norm, B_norm, log_s

    结构：
    1) global 分支用 attention pooling 提取全局指纹
    2) local 分支用 GRU 抽取局部趋势
    3) 拼接后回归 A/B 方向与 log_s 能量

    该类主要被旧脚本 05.py / 06.py 使用。
    """

    def __init__(self, feature_dim=64, rank=4):
        super().__init__()
        # 1) Global 特征：用可学习 query 做 attention pooling
        self.query = nn.Parameter(torch.randn(1, 1, 32))
        self.global_proj = nn.Linear(feature_dim, 32)
        self.attn = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)

        # 2) Local 特征：GRU 抽取窗口内趋势
        self.local_encoder = nn.GRU(feature_dim, 32, batch_first=True)

        # 3) log_s 分支（独立 MLP）
        self.log_s_mlp = nn.Sequential(
            nn.Linear(32 + 32, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # 4) A/B 回归头
        self.fc_A = nn.Linear(32 + 32, 64 * rank)
        self.fc_B = nn.Linear(32 + 32, rank * 32)

    def forward(self, global_cycles, local_cycles):
        # A) Global attention pooling
        g_feat = self.global_proj(global_cycles)  # [B, k1, 32]
        z_g, _ = self.attn(self.query.repeat(g_feat.size(0), 1, 1), g_feat, g_feat)
        z_g = z_g.squeeze(1)  # [B, 32]

        # B) Local GRU encoding
        _, z_l = self.local_encoder(local_cycles)
        z_l = z_l.squeeze(0)  # [B, 32]

        # C) 拼接后预测 A/B/log_s
        z_combined = torch.cat([z_g, z_l], dim=-1)  # [B, 64]

        A = self.fc_A(z_combined).view(-1, 64, 4)
        B = self.fc_B(z_combined).view(-1, 4, 32)

        # A/B 归一化为方向分量，log_s 负责尺度
        A_norm = A / (torch.norm(A, p="fro", dim=(1, 2), keepdim=True) + 1e-8)
        B_norm = B / (torch.norm(B, p="fro", dim=(1, 2), keepdim=True) + 1e-8)
        log_s = self.log_s_mlp(z_combined)

        return A_norm, B_norm, log_s


class AdvancedHyperGen(nn.Module):
    """另一套超网络变体（非扩散）：
    - global 先过 TransformerEncoderLayer
    - local/global 做 cross-attention
    - 用 FiLM 思想在 base_A/base_B 上生成增量

    该类是实验性增强版本，当前主线通常不直接使用。
    """

    def __init__(self, feature_dim=64, rank=4):
        super().__init__()
        self.global_encoder = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=8, batch_first=True)

        # 可学习 base manifold
        self.base_A = nn.Parameter(torch.randn(64, rank))
        self.base_B = nn.Parameter(torch.randn(rank, 32))

        # FiLM 参数生成器：输出 gamma / beta
        self.film_gen = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Linear(128, (64 * rank + rank * 32) * 2),
        )

        # log_s 头
        self.log_s_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, global_cycles, local_cycles):
        # A) 全局深加工
        g_feat = self.global_encoder(global_cycles)  # [B, K, 64]

        # B) local query global
        z_inter, _ = self.cross_attn(local_cycles.mean(1, keepdim=True), g_feat, g_feat)
        z = z_inter.squeeze(1)  # [B, 64]

        # C) FiLM 参数
        params = self.film_gen(z)
        gamma, beta = params.chunk(2, dim=-1)

        # D) 在 base_A/base_B 上做残差式调制
        A = self.base_A.unsqueeze(0) * (1 + gamma[:, : 64 * 4].view(-1, 64, 4)) + beta[:, : 64 * 4].view(-1, 64, 4)
        B = self.base_B.unsqueeze(0) * (1 + gamma[:, 64 * 4 :].view(-1, 4, 32)) + beta[:, 64 * 4 :].view(-1, 4, 32)

        # E) log_s
        log_s = self.log_s_head(z)
        return A, B, log_s


class WeightDenoiser(nn.Module):
    """当前主线的条件去噪网络（Diff / Flow 共用）：
    输入：
    - x_t: [B, 2048]      当前噪声状态/中间状态
    - t: [B]             时间步（Diff 离散, Flow 连续）
    - global_feats: [B, G, 64]
    - local_feats: [B, L, 64]

    条件编码路径：
    1) local self-attention
    2) global self-attention
    3) local/global 双向 cross-attention
    4) 与 x_t+t_emb query 在 decoder 层做 cross-attention

    注意：同一个 WeightDenoiser 会在每个 step 被重复调用，参数跨 step 共享。
    """

    def __init__(self, weight_dim=2048, hidden_dim=512, nhead=8, dropout=0.1, cond_input_dim=100):
        super().__init__()

        # 1) 时间步嵌入 MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 2) x_t 投影到 hidden_dim
        self.weight_proj = nn.Linear(weight_dim, hidden_dim)

        # 3) 条件编码：local/global 先投影，再分别 self-attn，再双向 cross-attn
        self.local_proj = nn.Linear(cond_input_dim, hidden_dim)
        self.global_proj = nn.Linear(cond_input_dim, hidden_dim)

        # type embedding 用于区分 token 来源域
        self.local_type_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.global_type_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        self.local_self_blocks = nn.ModuleList(
            [SelfAttnBlock(hidden_dim, nhead=nhead, dropout=dropout) for _ in range(3)]
        )
        self.global_self_blocks = nn.ModuleList(
            [SelfAttnBlock(hidden_dim, nhead=nhead, dropout=dropout) for _ in range(3)]
        )
        # 融合方式改为: local query global -> global_context, 再显式 offset 建模
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.offset_norm = nn.LayerNorm(hidden_dim)
        self.offset_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.offset_dropout = nn.Dropout(dropout)
        self.memory_norm = nn.LayerNorm(hidden_dim)

        # 4) 主干 decoder：query 是 x_t 分支，memory 是条件分支
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=dropout,
        )

        # 5) 输出回归到 2048 维 W 参数空间
        self.output_proj = nn.Linear(hidden_dim, weight_dim)

    def forward(self, x_t, t, global_feats, local_feats):
        """单次前向：给定当前状态 x_t 与条件特征，预测下一步所需信号。"""
        # A) 时间嵌入
        t_emb = self._get_timestep_embedding(t, self.weight_proj.out_features)
        t_emb = self.time_mlp(t_emb)

        # B) 构造 query token（x_t 分支）
        h = self.weight_proj(x_t) + t_emb  # [B, hidden_dim]
        h = h.unsqueeze(1)  # [B, 1, hidden_dim]

        # C) 条件三段式编码
        local_x = self.local_proj(local_feats) + self.local_type_embedding
        global_x = self.global_proj(global_feats) + self.global_type_embedding

        local_x = self._add_positional_encoding(local_x)
        global_x = self._add_positional_encoding(global_x)

        # local / global 各自 self-attn
        for block in self.local_self_blocks:
            local_x = block(local_x)
        for block in self.global_self_blocks:
            global_x = block(global_x)

        # 融合: local 做 Query, global 做 Key/Value
        global_context, _ = self.cross_attn(query=local_x, key=global_x, value=global_x, need_weights=False)

        # 显式偏移建模: offset = local - global_context
        offset = local_x - global_context

        # 轻量 FFN + LayerNorm, 不改变维度
        offset = offset + self.offset_dropout(self.offset_ffn(self.offset_norm(offset)))

        # 最终条件: 拼接原始全局序列 + 偏移序列
        memory = torch.cat([global_x, offset], dim=1)
        memory = self.memory_norm(memory)

        # D) query-memory cross-attention
        out = self.decoder_layer(h, memory)

        # E) 输出到权重空间
        return self.output_proj(out.squeeze(1))

    def _add_positional_encoding(self, x):
        """为条件序列添加正弦位置编码，帮助注意力感知序列位置信息。"""
        seq_len = x.size(1)
        dim = x.size(2)
        device = x.device

        half_dim = dim // 2
        if half_dim == 0:
            return x

        # 标准 sinusoidal PE
        freq = torch.exp(torch.arange(half_dim, device=device) * (-math.log(10000.0) / max(half_dim - 1, 1)))
        positions = torch.arange(seq_len, device=device).float().unsqueeze(1)
        sinusoid = positions * freq.unsqueeze(0)
        pos = torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=-1)
        if dim % 2 == 1:
            pos = F.pad(pos, (0, 1))
        return x + pos.unsqueeze(0)

    def _get_timestep_embedding(self, timesteps, embedding_dim):
        """标准正弦时间步编码，用于 Diff/Flow 统一的时间条件注入。"""
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if embedding_dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1))
        return emb


class FlowMatchingDenoiser(WeightDenoiser):
    """Flow 版本去噪器。

    目前不单独改结构，直接复用 WeightDenoiser。
    区别不在网络结构本身，而在训练目标与采样公式：
    - Diff: 学噪声预测
    - Flow: 学速度场/流场预测
    """

    pass
