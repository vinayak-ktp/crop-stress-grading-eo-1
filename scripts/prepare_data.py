import os

from data_pipeline.preprocessing import preprocess_hyperspectral_data

RAW_PATH = os.path.join("data", "raw", "full_data.csv")
PROCESSED_DIR = os.path.join("data", "processed")

preprocess_hyperspectral_data(RAW_PATH, PROCESSED_DIR, target_col='Stage')
