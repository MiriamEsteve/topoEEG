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
    # Step 1: Create object tda
    topoEEG_obj = tda(raw = None, n_components=10, random_state=97, max_iter=100, grid_size = 10000)

    # Step 2: Compute ICA
    topoEEG_obj.compute_ica()

    # Step 3: Compute PSD band power
    fmin, fmax = 10, 20
    topoEEG_obj.point_cloud = []

    for i in range(len(topoEEG_obj.raw)):
        # Compute the point cloud
        topoEEG_obj.point_cloud.append(topoEEG_obj.compute_psd_band_power(str(i), topoEEG_obj.raw[i], fmin, fmax))

        # Step 4: Compute persistence diagram
        diagram = topoEEG_obj.compute_persistence_diagram(topoEEG_obj.point_cloud[i])
        
        # Step 5: Compute landscape values
        grid = np.linspace(0, np.max(topoEEG_obj.point_cloud[i]), topoEEG_obj.grid_size)
        topoEEG_obj.landscapes.append(topoEEG_obj.compute_landscape_values(diagram, grid))

        # Step 6: Plot persistence landscape
        topoEEG_obj.plot_persistence_landscape(topoEEG_obj.landscapes)

    # Step 7: Classify landscapes
    topoEEG_obj.classify_landscapes()

    # Step All
    #topoEEG_obj.run_analysis()



if __name__ == "__main__":
    # Execute step by step
    main()
