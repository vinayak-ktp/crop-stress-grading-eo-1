import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class HyperspectralDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.X = torch.tensor(df.drop(columns=['target']).values, dtype=torch.float32)
        self.y = torch.tensor(df['target'].values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        features = self.X[idx].unsqueeze(-1)    # (seq_len, channels=1)
        return features, self.y[idx]


def get_loader(csv_path, batch_size=32, shuffle=False):
    dataset = HyperspectralDataset(csv_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader
