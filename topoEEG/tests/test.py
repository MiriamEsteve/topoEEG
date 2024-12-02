import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis import tda
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import ripser
from read_file import load_all_data

def main():
    # Read files
    raw = load_all_data()

    # Step 1: Create object tda
    topoEEG_obj = tda(raw = raw, n_components=10, random_state=97, max_iter=100, grid_size = 10000)

    # Step 2: Compute ICA
    topoEEG_obj.raw_clean = topoEEG_obj.compute_ica()

    # Step 3: Compute PSD band power
    topoEEG_obj.point_cloud = []

    # Step 4: Prepare raw_clean with subject IDs
    topoEEG_obj.raw_clean = [(f"subj_{i}", raw) for i, raw in enumerate(topoEEG_obj.raw_clean)]

    # Step 5: Compute PSD band power
    topoEEG_obj.point_cloud = topoEEG_obj.compute_psd_band_power()

    # Print mean values for verification
    for i, point_cloud in enumerate(topoEEG_obj.point_cloud):
        print(f"Subject {i} mean PSD band power: {point_cloud.mean()}")

    for i, raw in enumerate(topoEEG_obj.raw_clean):
        

        # Define grid and compute landscape values
        grid = np.linspace(0, np.max(topoEEG_obj.point_cloud[-1]), topoEEG_obj.grid_size)

        # Compute persistence diagram
        topoEEG_obj.landscapes.append(topoEEG_obj.compute_persistence_diagram(grid))
        
        # Plot the persistence landscape
        topoEEG_obj.plot_persistence_landscape(str(i), grid, topoEEG_obj.landscapes[-1])
    
    # Classification
    topoEEG_obj.classify_landscapes(topoEEG_obj.landscapes)

    # Step All
    #topoEEG_obj.run_analysis()



if __name__ == "__main__":
    # Execute step by step
    main()
