import torch.nn as nn
import torch

# 服务端模型（仅共享特征提取层）
class ServerSharedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_parameters = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            

            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Flatten(),  # 展平特征图
            nn.Linear(20736, 128),
            nn.ReLU(),
        )

    # 可选：手动初始化参数
    # def reset_parameters(self):
    #     nn.init.xavier_uniform_(self.feature_extractor.weight)
    #     nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return self.shared_parameters(x)


# 定义初始化函数
