import torch
import torch.nn as nn


class AlexNetBlock3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        pool_and_norm: bool,
    ):
        super().__init__()

        self.conv_layer = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.relu = nn.ReLU()
        self.pool_and_norm = pool_and_norm
        if self.pool_and_norm:
            self.norm = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
            self.pool = nn.MaxPool3d(kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.relu(x)
        if self.pool_and_norm:
            x = self.norm(x)
            x = self.pool(x)

        return x


class AlexNet(nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()

        self.block1 = AlexNetBlock3d(in_channels, 96, 11, 4, 0, True)
        self.block2 = AlexNetBlock3d(96, 256, 5, 1, 2, True)
        self.block3 = AlexNetBlock3d(256, 384, 3, 1, 1, False)
        self.block4 = AlexNetBlock3d(384, 384, 3, 1, 1, False)
        self.block5 = AlexNetBlock3d(384, 256, 3, 1, 1, True)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(9216, 4096)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.dropout2 = nn.Dropout(0.5)
        self.classication_layer = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.classication_layer(x)

        return x
