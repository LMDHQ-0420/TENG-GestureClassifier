"""路线B 模型：2D-CNN

针对小样本设计：4 个卷积块逐步提取时频特征，全局平均池化 + Dropout 强正则。
输入 scalogram 使用全局 log 归一化，保留通道间相对能量。
"""

import torch
import torch.nn as nn


class LightCNN(nn.Module):
    """2D-CNN，输入 [B, 3, 64, 128] scalogram，输出 10 类"""

    def __init__(self, num_classes: int = 10, dropout: float = 0.4, width: int = 32):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(3, width),          # 64x128 -> 32x64
            conv_block(width, width * 2),  # -> 16x32
            conv_block(width * 2, width * 4),  # -> 8x16
            conv_block(width * 4, width * 4),  # -> 4x8
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)
