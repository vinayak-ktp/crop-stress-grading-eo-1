import torch.nn as nn


class PaperCNN1D(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.3)
        )

        reduced_len = input_dim // 2

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * reduced_len, num_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x
