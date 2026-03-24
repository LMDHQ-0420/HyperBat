import torch
import torch.nn as nn
import torch.nn.functional as F

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