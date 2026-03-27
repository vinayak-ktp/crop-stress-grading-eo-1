import torch.nn as nn


class HybridModel(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=64, nhead=4):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, d_model // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )
        self.skip_proj = nn.Conv1d(1, d_model, kernel_size=1)    # projection for residual
        self.maxpool = nn.MaxPool1d(2)

        reduced_len = input_dim // 2

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            batch_first=True, dropout=0.2
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * reduced_len, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)          # (batch, 1, seq_len)
        residual = self.skip_proj(x)   # project input to d_model channels for skip
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x + residual               # skip connection across both conv blocks
        x = self.maxpool(x)
        x = x.transpose(1, 2)          # (batch, seq_len_reduced, d_model)
        x = self.transformer(x)
        return self.classifier(x)
