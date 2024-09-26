import os
import numpy as np
import plotly.graph_objs as go
import mne
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#################################### Load data ###########################################################

def construct_file_paths():
    """Construct file paths for AD, CN, and FTD EEG data."""
    file_path_AD, file_path_CN, file_path_FTD = [], [], []

    for file_num in range(1, 89):
        sub_id = f'sub-{file_num:03d}'
        file_path = f'../../data/bids/{sub_id}/eeg/{sub_id}_task-eyesclosed_eeg.set'
        
        if file_num < 37:
            file_path_AD.append(file_path)
        elif 37 <= file_num < 67:
            file_path_CN.append(file_path)
        elif 67 <= file_num < 89:
            file_path_FTD.append(file_path)

    return file_path_AD, file_path_CN, file_path_FTD

def load_EEG_data(file_paths):
    """Load EEG data from given file paths using MNE."""
    raw_data = []

    for i, file_path in enumerate(file_paths):
        print(file_path)
        if os.path.exists(file_path):
            try:
                raw = mne.io.read_raw_eeglab(file_path, preload=True)
                raw_data.append(raw)
                logging.info(f"Loaded {file_path} successfully.")
            except Exception as e:
                logging.error(f"Failed to load {file_path}: {str(e)}")
        else:
            logging.warning(f"File not found: {file_path}")

    return raw_data

def load_all_data():
    """Load all AD, CN, and FTD EEG data."""
    file_path_AD, file_path_CN, file_path_FTD = construct_file_paths()

    raw_AD = load_EEG_data(file_path_AD)
    raw_CN = load_EEG_data(file_path_CN)
    raw_FTD = load_EEG_data(file_path_FTD)

    if raw_AD:
        logging.info(f"Loaded {len(raw_AD)} AD datasets.")
    if raw_CN:
        logging.info(f"Loaded {len(raw_CN)} CN datasets.")
    if raw_FTD:
        logging.info(f"Loaded {len(raw_FTD)} FTD datasets.")

    return raw_AD, raw_CN, raw_FTD
   