"""
Utility functions and configuration constants for PUEA Detection Framework.
"""

import os
import joblib
import pandas as pd
import numpy as np

# Feature and Label Definitions
FEATURE_COLUMNS = [
    'RSS_dBm',
    'SNR_dB',
    'Transmission_Power_dBm',
    'X_Coordinate_km',
    'Y_Coordinate_km',
    'Channel_Occupancy_Time_s',
    'Frequency_Hz',
    'RSS_Deviation',
    'SNR_Deviation',
    'Distance_Mismatch_km',
    'SINR_dB'
]

LABEL_COLUMN = 'Label'

FEATURE_DESCRIPTIONS = {
    'RSS_dBm': 'Received Signal Strength (dBm)',
    'SNR_dB': 'Signal-to-Noise Ratio (dB)',
    'Transmission_Power_dBm': 'Transmission Power (dBm)',
    'X_Coordinate_km': 'Transmitter X Position (km)',
    'Y_Coordinate_km': 'Transmitter Y Position (km)',
    'Channel_Occupancy_Time_s': 'Channel Occupancy Time (s)',
    'Frequency_Hz': 'Operating Frequency (Hz)',
    'RSS_Deviation': 'RSS Deviation from PU Profile',
    'SNR_Deviation': 'SNR Deviation from Expected Value',
    'Distance_Mismatch_km': 'Location / Distance Mismatch (km)',
    'SINR_dB': 'Signal-to-Interference-plus-Noise Ratio (dB)'
}


def get_project_paths():
    """
    Returns absolute paths to key project directories.
    Handles execution from both project root and src/ directory.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(current_dir) == 'src':
        root_dir = os.path.dirname(current_dir)
    else:
        root_dir = current_dir

    paths = {
        'root': root_dir,
        'data': os.path.join(root_dir, 'data'),
        'models': os.path.join(root_dir, 'models'),
        'reports': os.path.join(root_dir, 'reports'),
        'figures': os.path.join(root_dir, 'reports', 'figures'),
        'app': os.path.join(root_dir, 'app')
    }

    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    return paths


def save_object(obj, filename):
    """Saves a python object using joblib to the models directory."""
    paths = get_project_paths()
    file_path = os.path.join(paths['models'], filename)
    joblib.dump(obj, file_path)
    print(f"[utils] Object saved to {file_path}")
    return file_path


def load_object(filename):
    """Loads a python object using joblib from the models directory."""
    paths = get_project_paths()
    file_path = os.path.join(paths['models'], filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model or artifact not found at {file_path}")
    obj = joblib.load(file_path)
    print(f"[utils] Object loaded from {file_path}")
    return obj
