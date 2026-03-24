import torch
import torch.nn as nn

class SOHPredictorCNN(nn.Module):
    def __init__(self):
        super(SOHPredictorCNN, self).__init__()
        
        # 1D-CNN 提取特征，输入长度固定为 400
        self.conv_layers = nn.Sequential(
            # 第一层：卷积核大小为 7，捕捉局部变化，输出 16 通道
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # 长度从 400 降到 200
            
            # 第二层：进一步提取高阶特征
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # 长度从 200 降到 100
        )
        
        # 全连接层进行回归预测；输入固定为 32*100=3200 维
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 100, 64),
            nn.ReLU(),
            nn.Dropout(0.2), # 防止过拟合
            nn.Linear(64, 1)  # 最终输出 1*1 的 SOH 值
        )

    def forward(self, x):
        # x 形状: (Batch, 400)
        # 转换为 Conv1d 需要的 (Batch, Channel, Length) -> (Batch, 1, 400)
        x = x.unsqueeze(1)
        
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # 平铺特征，形状变为 (Batch, 3200)
        x = self.fc_layers(x)
        return x