import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 多尺度卷积：同时提取不同跨度的特征
        self.conv_short = nn.Conv1d(in_channels, out_channels // 2, kernel_size=3, padding=1)
        self.conv_long = nn.Conv1d(in_channels, out_channels // 2, kernel_size=9, padding=4)
        self.bn = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        x = torch.cat([self.conv_short(x), self.conv_long(x)], dim=1)
        return F.leaky_relu(self.bn(x), 0.1)

class Encoder(nn.Module):
    """
    [升级版 Encoder] 
    结合多尺度卷积与自注意力机制，追求极致特征提取。
    """
    def __init__(self, input_len=400, embed_dim=64):
        super().__init__()
        # 初始特征映射
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1)
        )
        
        # 残差卷积块
        self.res_block = MultiScaleConv(32, 64)
        self.downsample = nn.Conv1d(32, 64, kernel_size=1, stride=1)
        
        # 简易自注意力层 (Attention) - 捕捉电压点间的长程依赖
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1) # [B, 1, 400]
            
        x = self.stem(x) # [B, 32, 200]
        
        # 残差连接
        identity = self.downsample(x)
        x = self.res_block(x) + identity # [B, 64, 200]
        
        # Transformer-like Attention
        # x shape: [B, C, L] -> [B, L, C]
        x = x.transpose(1, 2)
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = x.transpose(1, 2) # [B, 64, 200]
        
        x = self.pool(x).flatten(1) # [B, 64]
        return x

class SOHPredictor(nn.Module):
    """
    [升级版 Predictor]
    增加宽度，通过深度非线性逼近 SOH。
    """
    def __init__(self, input_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# GMANetPreTrain 和 GMANet 保持与你逻辑一致
# 仅替换内部的 Encoder 和 Head 实现
# ==========================================

class GMANetPreTrain(nn.Module):
    def __init__(self, input_len=400):
        super().__init__()
        self.encoder = Encoder(input_len, embed_dim=64)
        self.head = SOHPredictor(input_dim=32)
        # 核心对齐层：Phase 1 学习到的这个权重将作为 Phase 2 的“基准锚点”
        self.temp_connector = nn.Linear(64, 32)

    def forward(self, x):
        feat = self.encoder(x)
        aligned = self.temp_connector(feat)
        return self.head(aligned)

class GMANet(nn.Module):
    def __init__(self, input_len=400, rank=4):
        super().__init__()
        self.encoder = Encoder(input_len, embed_dim=64)
        self.head = SOHPredictor(input_dim=32)
        self.rank = rank

    def forward(self, x, param_A, param_B):
        feat = self.encoder(x)
        # W = A @ B [64, 32]
        if param_A.dim() == 2:
            W = param_A @ param_B
            aligned = feat @ W
        else:
            W = torch.bmm(param_A, param_B)
            aligned = torch.bmm(feat.unsqueeze(1), W).squeeze(1)
        return self.head(aligned)