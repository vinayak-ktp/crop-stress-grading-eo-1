import os

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy.signal import savgol_filter
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


def get_closest_band(df, target):
    cols = df.columns.tolist()
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


def extract_indices(df):
    nir_col = get_closest_band(df, 854)
    swir1_col = get_closest_band(df, 1649)
    swir2_col = get_closest_band(df, 2133)

    nir = df[nir_col].values.astype(float)
    swir1 = df[swir1_col].values.astype(float)
    swir2 = df[swir2_col].values.astype(float)

    mlvi = (nir - swir1) / (swir2 + 1e-8)
    h_vsi = (nir - swir1) / (nir + swir1 + swir2 + 1e-8)

    return mlvi, h_vsi


def preprocess_hyperspectral_data(input_path, output_dir, target_col='Stage'):
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

    print("Computing Derivative Spectroscopy")
    X_deriv1 = np.gradient(X_smoothed, axis=1)

    print("Calculating MLVI and H_VSI")
    mlvi, h_vsi = extract_indices(X_raw)

    # Combined features: Smoothed + 1st Derivative + Indices
    X_combined = np.hstack([
        X_smoothed,
        X_deriv1,
        mlvi.reshape(-1, 1),
        h_vsi.reshape(-1, 1)
    ])

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

    X_tr = train_df.drop(columns=['target']).values
    y_tr = train_df['target'].values

    # Data augmentation with SMOTE
    sm = SMOTE(random_state=42)
    X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)

    train_df = pd.DataFrame(X_tr_res, columns=train_df.drop(columns=['target']).columns)
    train_df['target'] = y_tr_res
    print("Data augmented with SMOTE")

    train_df.to_csv(os.path.join(splits_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(splits_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(splits_dir, 'test.csv'), index=False)

    print(f"Saved splits to {splits_dir}/")
    