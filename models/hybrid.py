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


class HybridModel(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=64, nhead=8):
        super(HybridModel, self).__init__()
        # CNN backbone with SE attention
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            # Block 2
            nn.Conv1d(32, d_model, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.MaxPool1d(2),
        )
        self.se = SEBlock1D(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            batch_first=True, dropout=0.2, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=3,
            norm=nn.LayerNorm(d_model)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)       # (batch, 1, seq_len)
        x = self.cnn(x)
        x = self.se(x)
        x = x.transpose(1, 2)       # (batch, seq_len_reduced, d_model)
        x = self.transformer(x)
        x = x.mean(dim=1)           # Mean pooling over sequence

        return self.classifier(x)
