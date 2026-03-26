import os

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Metadata columns (not spectral features)
META_COLS = [
    'UniqueID',
    'Country',
    'AEZ',
    'Image',
    'Month',
    'Year',
    'jd',
    'long',
    'lat',
    'Crop',
    'Stage'
]


def extract_indices(df):
    cols = df.columns.tolist()

    def get_closest_band(target):
        numeric_cols = []
        for c in cols:
            if c.startswith('X'):
                try:
                    numeric_cols.append((c, int(c[1:])))
                except ValueError:
                    continue
        if not numeric_cols:
            return cols[0]    # Fallback
        closest = min(numeric_cols, key=lambda x: abs(x[1] - target))
        return closest[0]

    nir_col = get_closest_band(854)
    swir1_col = get_closest_band(1649)
    swir2_col = get_closest_band(2133)

    nir = df[nir_col].values.astype(float)
    swir1 = df[swir1_col].values.astype(float)
    swir2 = df[swir2_col].values.astype(float)

    mlvi = (nir - swir1) / (swir2 + 1e-8)
    h_vsi = (nir - swir1) / (nir + swir1 + swir2 + 1e-8)

    return mlvi, h_vsi


def continuum_removal(spectra):
    n_samples, n_bands = spectra.shape
    result = np.zeros_like(spectra)

    for i in range(n_samples):
        spectrum = spectra[i]
        # Build convex hull upper envelope via simple linear interpolation bw local maxima
        hull = np.copy(spectrum)
        x = np.arange(n_bands)

        # Upper envelope: linear interp between first and last point
        envelope = np.interp(x, [0, n_bands - 1], [spectrum[0], spectrum[-1]])
        # Refine: for each segment find local max and rebuild
        for _ in range(3):
            above = spectrum >= envelope
            idx = np.where(above)[0]
            if len(idx) >= 2:
                envelope = np.interp(x, idx, spectrum[idx])

        # Continuum removed = spectrum / envelope
        envelope = np.maximum(envelope, 1e-8)
        result[i] = spectrum / envelope

    return result


def preprocess_hyperspectral_data(input_path, output_dir, target_col='Stage', use_pca=False, pca_variance=0.99):
    # Skip 9 metadata rows
    df = pd.read_csv(input_path, skiprows=9, na_values=['NA', 'na', ''])

    # Target column
    if target_col not in df.columns:
        print(f"'{target_col}' not found")
        target_col = df.columns[-1]
    print(f"Using target column: '{target_col}'")

    y_raw = df[target_col].values

    # String to Integer encoding
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Select spectral band columns (starting with 'X')
    spectral_cols = [c for c in df.columns if c.startswith('X')]
    print(f"Total spectral columns found: {len(spectral_cols)}")

    # Drop columns with >50% missing values
    na_ratio = df[spectral_cols].isna().mean()
    good_cols = na_ratio[na_ratio <= 0.5].index.tolist()
    dropped_cols = set(spectral_cols) - set(good_cols)
    print(f"Dropped {len(dropped_cols)} spectral columns (>50% NAs)")
    print(f"Remaining spectral columns: {len(good_cols)}")

    # Drop rows still having NAs
    X_spectral = df[good_cols]
    has_no_na = X_spectral.notna().all(axis=1)
    n_dropped = len(X_spectral) - has_no_na.sum()
    X_spectral = X_spectral[has_no_na].reset_index(drop=True)
    y = y[has_no_na.values]
    print(f"Dropped {n_dropped} rows with remaining NAs. Keeping {len(X_spectral)} rows.")

    X_raw = X_spectral.astype(float)

    # Savitzky-Golay filter
    n_bands = X_raw.shape[1]
    window_length = min(11, n_bands if n_bands % 2 == 1 else n_bands - 1)
    if window_length < 3:
        window_length = 3

    print(f"Applying Savitzky-Golay filtering with window length {window_length}")
    X_smoothed = savgol_filter(X_raw.values, window_length=window_length, polyorder=2, axis=1)

    print("Computing 1st and 2nd Derivative Spectroscopy")
    X_deriv1 = np.gradient(X_smoothed, axis=1)
    X_deriv2 = np.gradient(X_deriv1, axis=1)

    print("Applying Continuum Removal")
    X_cr = continuum_removal(X_smoothed)

    print("Calculating MLVI and H_VSI")
    mlvi, h_vsi = extract_indices(X_raw)

    # Combined features: Smoothed + 1st Derivative + 2nd Derivative + Continuum Removal + Indices
    X_combined = np.hstack([
        X_smoothed,
        X_deriv1,
        X_deriv2,
        X_cr,
        mlvi.reshape(-1, 1),
        h_vsi.reshape(-1, 1)
    ])

    print(f"Combined feature count: {X_combined.shape[1]}")

    # Optional PCA whitening
    if use_pca:
        print(f"Applying PCA (retaining {pca_variance * 100:.0f}% variance)")
        pca = PCA(n_components=pca_variance, whiten=True, random_state=42)
        X_combined = pca.fit_transform(X_combined)
        print(f"PCA reduced features to: {X_combined.shape[1]}")

    # Normalizing features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    # Processed DataFrame
    feature_cols = [f'F_{i}' for i in range(X_scaled.shape[1])]
    df_processed = pd.DataFrame(X_scaled, columns=feature_cols)
    df_processed['target'] = y

    os.makedirs(output_dir, exist_ok=True)
    processed_path = os.path.join(output_dir, 'processed_data.csv')
    df_processed.to_csv(processed_path, index=False)
    print(f"Saved processed data to {processed_path}")

    train_df, temp_df = train_test_split(df_processed, test_size=0.30, stratify=y, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['target'], random_state=42)

    splits_dir = os.path.join(os.path.dirname(output_dir), 'splits')
    os.makedirs(splits_dir, exist_ok=True)

    train_df.to_csv(os.path.join(splits_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(splits_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(splits_dir, 'test.csv'), index=False)
    print(f"Saved splits to {splits_dir}/")
