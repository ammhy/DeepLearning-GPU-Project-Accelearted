# model_1.py
import torch
import torch.nn as nn


class ModulationClassifier(nn.Module):
    def __init__(self, num_classes, input_size):
        super(ModulationClassifier, self).__init__()

        # 卷积神经网络部分
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=1024, out_channels=2048, kernel_size=3, padding=1),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=2048, out_channels=4096, kernel_size=3, padding=1),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=4096, out_channels=8192, kernel_size=3, padding=1),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool1d(kernel_size=2)
        )

        # 全连接层部分
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 调整维度以适应 Conv1d 的输入要求
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def get_flatten_size(model, input_dim):
    with torch.no_grad():
        device = next(model.parameters()).device  # 获取模型当前所在设备 (CPU 或 GPU)
        x = torch.zeros(1, *input_dim).to(device)  # 将输入迁移到相同设备
        x = model.cnn(x)
        return x.view(1, -1).size(1)

