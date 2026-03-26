import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class SpectralAugmentation:
    def __init__(self, noise_std=0.05, band_drop_rate=0.05, scale_range=0.10):
        self.noise_std = noise_std
        self.band_drop_rate = band_drop_rate
        self.scale_range = scale_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Gaussian noise
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        # Random band dropout (zero out random bands)
        if self.band_drop_rate > 0:
            mask = torch.rand(x.shape[0]) > self.band_drop_rate
            x = x * mask.unsqueeze(-1).float()

        # Random scaling (±scale_range)
        if self.scale_range > 0:
            scale = 1.0 + (torch.rand(1).item() * 2 - 1) * self.scale_range
            x = x * scale

        return x


class HyperspectralDataset(Dataset):
    def __init__(self, csv_file, augment=False):
        df = pd.read_csv(csv_file)
        self.X = torch.tensor(df.drop(columns=['target']).values, dtype=torch.float32)
        self.y = torch.tensor(df['target'].values, dtype=torch.long)
        self.augmentation = SpectralAugmentation() if augment else None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        features = self.X[idx].unsqueeze(-1)    # (seq_len, channels=1)
        if self.augmentation is not None:
            features = self.augmentation(features)
        return features, self.y[idx]


def get_class_weights(dataset):
    labels = dataset.y.numpy()
    counts = np.bincount(labels)
    weights = 1.0 / (counts + 1e-8)
    weights = weights / weights.sum() * len(counts)   # Normalize
    return torch.tensor(weights, dtype=torch.float32)


def get_loader(csv_path, batch_size=32, shuffle=False, augment=False, weighted_sampling=False):
    dataset = HyperspectralDataset(csv_path, augment=augment)

    sampler = None
    if weighted_sampling:
        labels = dataset.y.numpy()
        class_counts = np.bincount(labels)
        class_weights = 1.0 / (class_counts + 1e-8)
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(dataset),
            replacement=True
        )
        shuffle = False   # Sampler and shuffle are mutually exclusive

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler)
    return loader
