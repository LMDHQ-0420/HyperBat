import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# Part 1: 公共组件 (Shared Components)
# ==========================================

class Encoder(nn.Module):
    """
    [Bottom Bun] 通用特征提取器
    结构: 3层 1D-CNN + Global Pooling
    作用: 将 400维电压/容量曲线 映射为 64维物理特征向量
    """
    def __init__(self, input_len=400, in_channels=1, embed_dim=64, dropout_p=0.1):
        super().__init__()
        
        # Layer 1: 感受野小，提取斜率和噪声
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.drop1 = nn.Dropout1d(p=dropout_p)
        
        # Layer 2: 感受野中，提取峰值形状
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.drop2 = nn.Dropout1d(p=dropout_p)
        
        # Layer 3: 感受野大，提取整体趋势
        self.conv3 = nn.Conv1d(32, embed_dim, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(embed_dim)
        self.drop3 = nn.Dropout1d(p=dropout_p)
        
        # Global Pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # 自动调整维度: [Batch, 400] -> [Batch, 1, 400]
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        x = self.drop2(F.relu(self.bn2(self.conv2(x))))
        x = self.drop3(F.relu(self.bn3(self.conv3(x))))
        
        x = self.pool(x).flatten(1) # Output: [Batch, 64]
        return x


class SOHPredictor(nn.Module):
    """
    [Top Bun] SOH 预测头
    结构: 2层 MLP
    作用: 将对齐后的特征映射到 SOH 值 (0~1)
    """
    def __init__(self, input_dim=32, dropout_p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(16, 1),
            nn.Sigmoid() # 强制输出在 0-1 之间
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# Part 2: 功能模型 (Functional Models)
# ==========================================

class GMANetPreTrain(nn.Module):
    """
    [Phase 1] 预训练模型
    架构: Encoder -> Linear(Temp) -> Head
    用途: 仅用于训练 Encoder 和 Head 的通用参数
    """
    def __init__(self, input_len=400, in_channels=1):
        super().__init__()
        # 实例化公共组件
        self.encoder = Encoder(input_len, in_channels, embed_dim=64)
        self.head = SOHPredictor(input_dim=32)
        
        # 临时的全连接层，用于连接 64维特征 和 32维Head
        # 这个层在 Phase 2 会被扔掉，替换为 Low-Rank Adapter
        self.temp_connector = nn.Linear(64, 32)

    def forward(self, x):
        feat = self.encoder(x)          # [Batch, 64]
        aligned = self.temp_connector(feat) # [Batch, 32] (Linear Transformation)
        soh = self.head(aligned)        # [Batch, 1]
        return soh


class GMANet(nn.Module):
    """
    [Phase 2 & Inference] 生成式流形对齐网络 (三明治架构)
    架构: Frozen Encoder -> Dynamic Low-Rank Adapter -> Frozen Head
    用途: 
       1. Phase 2: 挖掘每颗电池的最佳 A, B 参数
       2. Inference: 接收扩散模型生成的 A, B 进行预测
    """
    def __init__(self, input_len=400, in_channels=1, rank=4):
        super().__init__()
        self.encoder = Encoder(input_len, in_channels, embed_dim=64)
        self.head = SOHPredictor(input_dim=32)
        
        # 定义维度信息，供外部调用
        self.enc_dim = 64
        self.head_in_dim = 32
        self.rank = rank

    def forward(self, x, param_A, param_B):
        """
        x: 输入数据 [Batch, 400]
        param_A: 低秩矩阵 A [Batch, 64, rank] 或 [64, rank] (如果是单颗电池优化)
        param_B: 低秩矩阵 B [Batch, rank, 32] 或 [rank, 32]
        """
        # 1. 提取特征 (通常这部分是 Frozen 的)
        feat = self.encoder(x) # [Batch, 64]
        
        # 2. 动态对齐 (Low-Rank Adapter: W = A @ B)
        # 确保 feat 是 [Batch, 1, 64] 以便进行矩阵乘法
        feat_unsqueezed = feat.unsqueeze(1) 
        
        # 处理 Batch 维度: 如果 A, B 是针对整个 Batch 共享的 (Phase 2 单电池优化时)
        if param_A.dim() == 2: 
            # A: [64, r], B: [r, 32]
            # W: [64, 32]
            generated_weight = param_A @ param_B
            aligned = feat @ generated_weight # [Batch, 64] @ [64, 32] -> [Batch, 32]
            
        else: 
            # 如果 A, B 是扩散模型生成的，每条数据都不一样 (Inference 时)
            # param_A: [Batch, 64, r], param_B: [Batch, r, 32]
            # generated_weight: [Batch, 64, 32]
            generated_weight = torch.bmm(param_A, param_B)
            # [Batch, 1, 64] bmm [Batch, 64, 32] -> [Batch, 1, 32]
            aligned = torch.bmm(feat_unsqueezed, generated_weight).squeeze(1)

        # 3. 预测 (通常这部分是 Frozen 的)
        soh = self.head(aligned)
        return soh