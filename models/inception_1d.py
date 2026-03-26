import torch
import torch.nn as nn


class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        w = self.squeeze(x).view(b, c)
        w = self.excitation(w).view(b, c, 1)
        return x * w


class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        branch_out = out_channels // 4
        # Branch 1: 1x1 convolution
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_out, kernel_size=1, bias=False),
            nn.BatchNorm1d(branch_out),
            nn.GELU()
        )
        # Branch 2: 1x1 -> 3x3
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_out, kernel_size=1, bias=False),
            nn.Conv1d(branch_out, branch_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(branch_out),
            nn.GELU()
        )
        # Branch 3: 1x1 -> 5x5
        self.branch5 = nn.Sequential(
            nn.Conv1d(in_channels, branch_out, kernel_size=1, bias=False),
            nn.Conv1d(branch_out, branch_out, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(branch_out),
            nn.GELU()
        )
        # Branch 4: 1x1 -> 9x9
        self.branch9 = nn.Sequential(
            nn.Conv1d(in_channels, branch_out, kernel_size=1, bias=False),
            nn.Conv1d(branch_out, branch_out, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(branch_out),
            nn.GELU()
        )
        self.se = SEBlock1D(out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        # Residual connection
        self.residual = nn.Sequential()
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b9 = self.branch9(x)

        out = torch.cat([b1, b3, b5, b9], dim=1)
        out = self.se(out)
        out = self.bn(out)
        out = out + self.residual(x)
        return nn.functional.gelu(out)


class InceptionNet1D(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(1)
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU()
        )
        self.block1 = InceptionBlock1D(32, 64)
        self.pool1 = nn.MaxPool1d(2)
        self.block2 = InceptionBlock1D(64, 128)
        self.pool2 = nn.MaxPool1d(2)
        self.block3 = InceptionBlock1D(128, 256)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)    # (batch, 1, seq_len)
        x = self.input_bn(x)
        x = self.stem(x)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
