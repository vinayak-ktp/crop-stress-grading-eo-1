import os

import numpy as np
import pandas as pd

SPLITS_DIR = "data/splits"
train = pd.read_csv(os.path.join(SPLITS_DIR, "train.csv"))
val = pd.read_csv(os.path.join(SPLITS_DIR, "val.csv"))
test = pd.read_csv(os.path.join(SPLITS_DIR, "test.csv"))

print("Split sizes")
print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
total = len(train) + len(val) + len(test)
print(f"Total: {total}")
print(f"Train/Val/Test ratio: {len(train)/total:.2%} / {len(val)/total:.2%} / {len(test)/total:.2%}")

feature_cols = [c for c in train.columns if c != 'target']
print("\nFeatures")
print(f"Number of features: {len(feature_cols)}")

print("\nClass Distribution (Train - post SMOTE)")
vc = train['target'].value_counts().sort_index()
for cls, cnt in vc.items():
    print(f"Class {cls}: {cnt} ({cnt/len(train)*100:.1f}%)")

print("\nClass Distribution (Val - original)")
vc = val['target'].value_counts().sort_index()
for cls, cnt in vc.items():
    print(f"Class {cls}: {cnt} ({cnt/len(val)*100:.1f}%)")

print("\nClass Distribution (Test - original)")
vc = test['target'].value_counts().sort_index()
for cls, cnt in vc.items():
    print(f"Class {cls}: {cnt} ({cnt/len(test)*100:.1f}%)")

X = train[feature_cols].values
print("\nFeature Statistics (Train)")
print(f"Mean: {X.mean():.4f}")
print(f"Std: {X.std():.4f}")
print(f"Min: {X.min():.4f}")
print(f"Max: {X.max():.4f}")
print(f"NaN count: {np.isnan(X).sum()}")

combined = pd.concat([train, val, test])
print("\nDuplicate Rows (train vs val+test)")
val_test = pd.concat([val, test])
dupes = pd.merge(train, val_test, how='inner')
print(f"Exact duplicates between train and val/test: {len(dupes)}")
