import torch.nn as nn


class HybridModel(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=32, nhead=4):
        super(HybridModel, self).__init__()

        self.conv1 = nn.Conv1d(1, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(2)

        reduced_len = input_dim // 2

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=0.3
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * reduced_len, 64),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)    # (batch, 1, seq_len)

        residual = self.conv1(x)
        x = self.relu(residual)
        x = self.conv2(x)
        x = self.relu(x + residual)    # Skip connection
        x = self.maxpool(x)

        x = x.transpose(1, 2)     # (batch, seq_len_reduced, d_model)
        x = self.transformer(x)

        return self.classifier(x)
