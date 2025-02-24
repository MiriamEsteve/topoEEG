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
    file_path = []

    for file_num in range(1, 89):
        sub_id = f'sub-{file_num:03d}'
        file_path_original = f'./topoEEG/data/bids/{sub_id}/eeg/{sub_id}_task-eyesclosed_eeg.set'
        
        file_path.append(file_path_original)

    return file_path

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
    file_path = construct_file_paths()

    raw = load_EEG_data(file_path)

    logging.info(f"Loaded {len(raw)} datasets.")
   
    return raw