import torch.nn as nn
import copy
from typing import Iterator

'''
客户端模型
'''

class ClientMTLModel(nn.Module):
    def __init__(self, server_model):
        super().__init__()
        # 继承服务端的共享特征层
        self.shared_layer = copy.deepcopy(server_model.shared_parameters)
        # 本地任务头
        self.task1_head = nn.Sequential(
            nn.Flatten(),  # 展平特征图
            nn.Linear(64 * 9 * 9, 128),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(128, 10)
        )  # 任务1
        self.task2_head = nn.Sequential(
            nn.Flatten(),  # 展平特征图
            nn.Linear(64 * 9 * 9, 128),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(128, 10)
        )  # 任务2

    def forward(self, x):
        features = self.shared_layer(x)
        return (self.task1_head(features)
                , self.task2_head(features))
