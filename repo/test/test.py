import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis import tda
import numpy as np
from read_file import load_all_data

def main():
    # Read files
    raw = load_all_data()

    # Step 1: Create object tda
    topoEEG_obj = tda(raw = raw, n_components=19, random_state=97, max_iter=1000, grid_size = 100, fmin=1, fmax=50, tmin=0, tmax=240)

    # Step 2: Compute ICA
    topoEEG_obj.raw_clean = topoEEG_obj.compute_ica()

    # Step 3: Compute PSD band power
    topoEEG_obj.point_cloud = topoEEG_obj.compute_plot_psd_band_power()

    # Step 4: Classification
    topoEEG_obj.classify_landscapes()


if __name__ == "__main__":
    # Execute step by step
    main()
