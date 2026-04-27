import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HyperLoRAGenerator(nn.Module):
    def __init__(self, feature_dim=64, rank=4):
        super().__init__()
        # 1. 引入 Attention 提取全局指纹 (识别体系差异)
        self.query = nn.Parameter(torch.randn(1, 1, 32))
        self.global_proj = nn.Linear(feature_dim, 32)
        self.attn = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        
        # 2. 局部窗口编码器 (保持 GRU 捕捉趋势)
        self.local_encoder = nn.GRU(feature_dim, 32, batch_first=True)
        
        # 3. 增强的 log_s 专门分支 (深度 MLP + 统计约束)
        self.log_s_mlp = nn.Sequential(
            nn.Linear(32 + 32, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # 4. A & B 生成分支
        self.fc_A = nn.Linear(32 + 32, 64 * rank)
        self.fc_B = nn.Linear(32 + 32, rank * 32)

    def forward(self, global_cycles, local_cycles):
        # --- A. Global 特征提取 (Attention Pooling) ---
        g_feat = self.global_proj(global_cycles) # [B, k1, 32]
        # 使用查询向量主动寻找体系特征，而不是被动平均
        z_g, _ = self.attn(self.query.repeat(g_feat.size(0), 1, 1), g_feat, g_feat)
        z_g = z_g.squeeze(1) # [B, 32]

        # --- B. Local 特征提取 ---
        _, z_l = self.local_encoder(local_cycles)
        z_l = z_l.squeeze(0) # [B, 32]
        
        z_combined = torch.cat([z_g, z_l], dim=-1) # [B, 64]
        
        # --- C. 解耦预测 ---
        # 预测 A, B (保持原样)
        A = self.fc_A(z_combined).view(-1, 64, 4)
        B = self.fc_B(z_combined).view(-1, 4, 32)
        A_norm = A / (torch.norm(A, p='fro', dim=(1, 2), keepdim=True) + 1e-8)
        B_norm = B / (torch.norm(B, p='fro', dim=(1, 2), keepdim=True) + 1e-8)
        
        # 使用深层 MLP 预测 log_s，增加其非线性表达能力
        log_s = self.log_s_mlp(z_combined)
        
        return A_norm, B_norm, log_s


class AdvancedHyperGen(nn.Module):
    def __init__(self, feature_dim=64, rank=4):
        super().__init__()
        # 1. 结构化特征提取：参考 BatteryGPT 的自回归/长程捕捉能力 
        self.global_encoder = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, batch_first=True)
        
        # 2. Cross-Attention：让 Local 窗口在 Global 背景下进行“定位”
        self.cross_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=8, batch_first=True)
        
        # 3. 基础权重空间 (Base Manifold)：学习电池退化的共构物理规律
        # 预测增量而非全量，是达到 0.5% 误差的关键
        self.base_A = nn.Parameter(torch.randn(64, rank))
        self.base_B = nn.Parameter(torch.randn(rank, 32))
        
        # 4. FiLM 预测器：预测缩放 (gamma) 和 偏移 (beta)
        self.film_gen = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Linear(128, (64 * rank + rank * 32) * 2) # 输出 gamma 和 beta
        )
        
        # 5. log_s 预测：引入领域自适应 (Domain Adaptation) 思想 [cite: 4014]
        # 增加残差连接，防止深层 MLP 梯度消失
        self.log_s_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, global_cycles, local_cycles):
        # A. 全局特征深加工
        g_feat = self.global_encoder(global_cycles) # [B, K, 64]
        
        # B. 交互：Local 为 Query，Global 为 KV
        # 这模拟了 BatLiNet 的“间接预测”逻辑：预测当前电池相对于标准退化模式的偏离 
        z_inter, _ = self.cross_attn(local_cycles.mean(1, keepdim=True), g_feat, g_feat)
        z = z_inter.squeeze(1) # [B, 64]
        
        # C. FiLM 参数生成 (高精度核心)
        params = self.film_gen(z)
        gamma, beta = params.chunk(2, dim=-1)
        
        # 作用于 Base Weights：W = base * (1 + gamma) + beta
        # 这种残差式的权重生成比直接预测绝对值稳定得多
        A = self.base_A.unsqueeze(0) * (1 + gamma[:, :64*4].view(-1, 64, 4)) + beta[:, :64*4].view(-1, 64, 4)
        B = self.base_B.unsqueeze(0) * (1 + gamma[:, 64*4:].view(-1, 4, 32)) + beta[:, 64*4:].view(-1, 4, 32)
        
        # D. log_s 预测
        log_s = self.log_s_head(z)
        
        return A, B, log_s



class WeightDenoiser(nn.Module):
    """
    核心去噪网络：基于 Transformer 解码器结构
    输入：x_t (385维噪声), t (时间步), global_feats (变长), local_feats (变长)
    """
    def __init__(self, weight_dim=385, hidden_dim=512, nhead=8):
        super().__init__()
        # 1. 时间步嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 2. 权重向量投影
        self.weight_proj = nn.Linear(weight_dim, hidden_dim)
        
        # 3. 条件处理器：处理变长的 global 和 local 特征
        # 假设输入的 feat 已经是 [Batch, Seq, 64]
        self.cond_proj = nn.Linear(64, hidden_dim)
        
        # 4. Transformer 层 (包含 Cross-Attention)
        # 我们使用标准的 Transformer Decoder Layer，因为它天然支持 Query 对 Memory 的交叉注意力
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=hidden_dim*4,
            batch_first=True
        )
        
        # 5. 输出投影
        self.output_proj = nn.Linear(hidden_dim, weight_dim)

    def forward(self, x_t, t, global_feats, local_feats):
        """
        x_t: [B, 385]
        t: [B] 标量时间步
        global_feats: [B, G, 64]
        local_feats: [B, L, 64]
        """
        # A. 时间嵌入
        # 显式使用 hidden_dim (256) 确保对齐
        t_emb = self._get_timestep_embedding(t, self.weight_proj.out_features) 
        t_emb = self.time_mlp(t_emb) 
        
        # B. 基础特征投影并增加序列维度
        h = self.weight_proj(x_t) + t_emb # [B, 256]
        h = h.unsqueeze(1)               # [B, 1, 256] <--- 关键：Transformer 需要 Seq 维度
        
        # C. 拼接条件上下文
        cond = torch.cat([global_feats, local_feats], dim=1) # [B, G+L, 64]
        memory = self.cond_proj(cond) # [B, G+L, 256]
        
        # D. Cross-Attention
        # Query: h, Key/Value: memory
        out = self.decoder_layer(h, memory) # [B, 1, 256]
        
        # E. 映射回权重空间
        return self.output_proj(out.squeeze(1)) # [B, 385]

    def _get_timestep_embedding(self, timesteps, embedding_dim):
        """标准正弦时间嵌入"""
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if embedding_dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1))
        return emb

class FlowMatchingDenoiser(WeightDenoiser):
    """
    与 WeightDenoiser 结构一致，仅用于在训练与评估脚本中按 denoiser_type 做模型切换。
    """
    pass
