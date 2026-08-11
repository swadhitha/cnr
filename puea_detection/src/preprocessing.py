"""
Data Preprocessing Module for PUEA Detection Framework.
Enforces strict train-only scaler fitting to eliminate data leakage.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils import get_project_paths, FEATURE_COLUMNS, LABEL_COLUMN


def get_data_filepaths():
    """Locates train and test CSV files, looking first in data/ then root/parent."""
    paths = get_project_paths()
    train_path = os.path.join(paths['data'], 'puea_dataset_train.csv')
    test_path = os.path.join(paths['data'], 'puea_dataset_test.csv')

    # Fallback to parent directory if files not in data/
    if not os.path.exists(train_path):
        parent_dir = os.path.dirname(paths['root'])
        train_path = os.path.join(parent_dir, 'puea_dataset_train.csv')
        test_path = os.path.join(parent_dir, 'puea_dataset_test.csv')

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Dataset files missing!\nExpected at:\n - {train_path}\n - {test_path}"
        )

    return train_path, test_path


def load_raw_data():
    """
    Loads train and test CSV files separately without any modification.
    """
    train_path, test_path = get_data_filepaths()
    print(f"[preprocessing] Loading train dataset from: {train_path}")
    print(f"[preprocessing] Loading test dataset from: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train_raw = train_df[FEATURE_COLUMNS].copy()
    y_train = train_df[LABEL_COLUMN].copy()

    X_test_raw = test_df[FEATURE_COLUMNS].copy()
    y_test = test_df[LABEL_COLUMN].copy()

    return X_train_raw, X_test_raw, y_train, y_test


def fit_and_save_scaler(X_train_raw):
    """
    Fits StandardScaler STRICTLY on X_train_raw and saves to models/scaler.pkl.
    """
    paths = get_project_paths()
    scaler_path = os.path.join(paths['models'], 'scaler.pkl')

    scaler = StandardScaler()
    scaler.fit(X_train_raw)

    joblib.dump(scaler, scaler_path)
    print(f"[preprocessing] StandardScaler fitted on train set and saved to {scaler_path}")
    return scaler


def load_scaler():
    """Loads fitted scaler from models/scaler.pkl."""
    paths = get_project_paths()
    scaler_path = os.path.join(paths['models'], 'scaler.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}. Run preprocessing first.")
    scaler = joblib.load(scaler_path)
    return scaler


def load_and_preprocess_data(save_scaler=True):
    """
    Main preprocessor entrypoint.
    Returns:
        X_train_raw, X_test_raw, X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    X_train_raw, X_test_raw, y_train, y_test = load_raw_data()

    if save_scaler:
        scaler = fit_and_save_scaler(X_train_raw)
    else:
        scaler = load_scaler()

    # Transform train and test using scaler fitted on X_train only
    X_train_scaled_arr = scaler.transform(X_train_raw)
    X_test_scaled_arr = scaler.transform(X_test_raw)

    # Return DataFrame format with column names preserved
    X_train_scaled = pd.DataFrame(X_train_scaled_arr, columns=FEATURE_COLUMNS, index=X_train_raw.index)
    X_test_scaled = pd.DataFrame(X_test_scaled_arr, columns=FEATURE_COLUMNS, index=X_test_raw.index)

    return X_train_raw, X_test_raw, X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == '__main__':
    print("=" * 60)
    print("Running Stage 1 Preprocessing Verification")
    print("=" * 60)

    X_train_raw, X_test_raw, X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_and_preprocess_data(save_scaler=True)

    print("\n--- SHAPE VERIFICATION ---")
    print(f"X_train_raw shape   : {X_train_raw.shape}")
    print(f"X_train_scaled shape: {X_train_scaled.shape}")
    print(f"X_test_raw shape    : {X_test_raw.shape}")
    print(f"X_test_scaled shape : {X_test_scaled.shape}")
    print(f"y_train shape       : {y_train.shape}")
    print(f"y_test shape        : {y_test.shape}")

    assert X_train_raw.shape == X_train_scaled.shape == (80000, 11), "Train shapes mismatch!"
    assert X_test_raw.shape == X_test_scaled.shape == (20000, 11), "Test shapes mismatch!"
    print("SUCCESS: Shape verification passed (80,000 train, 20,000 test, 11 features).")

    print("\n--- SCALER VERIFICATION ---")
    paths = get_project_paths()
    scaler_path = os.path.join(paths['models'], 'scaler.pkl')
    print(f"scaler.pkl exists at {scaler_path}: {os.path.exists(scaler_path)}")
    print(f"Scaler feature means (first 3): {scaler.mean_[:3]}")
    print(f"Scaler feature scales (first 3): {scaler.scale_[:3]}")

    print("\n--- FIRST 3 ROWS COMPARISON (RAW vs SCALED) ---")
    print("\nRAW FEATURES (First 3 rows):")
    print(X_train_raw.head(3).to_string())

    print("\nSCALED FEATURES (First 3 rows, centered near 0):")
    print(X_train_scaled.head(3).to_string())

    print("\n" + "=" * 60)
    print("Stage 1 Preprocessing completed and verified successfully!")
    print("=" * 60)
