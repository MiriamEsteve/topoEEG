import numpy as np
from analysis import tda
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import ripser
from read_file import load_all_data

def main():
    # Step 1: Create object tda
    tda_obj = tda(raw = None, n_components=14, random_state=97, max_iter=100, grid_size = 10000)

    # Step 2: Compute ICA
    tda_obj.compute_ica(tda_obj)

    # Step 3: Compute PSD band power
    tda_obj.compute_psd_band_power(tda_obj)

    # Step 4: Compute persistence diagram
    tda_obj.compute_persistence_diagram(tda_obj)

    # Step 5: Compute landscape values
    tda_obj.compute_landscape_values(tda_obj)

    # Step 6: Plot persistence landscape
    tda_obj.plot_persistence_landscape(tda_obj)

    # Step 7: Classify landscapes
    tda_obj.classify_landscapes(tda_obj)

    # Step All
    #tda_obj.run_analysis()



if __name__ == "__main__":
    # Execute step by step
    main()
