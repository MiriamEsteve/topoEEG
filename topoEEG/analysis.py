import numpy as np
from read_file import (load_all_data, logging)
from plotting import (
    plot_ica, plot_persistence_landscape, compute_psd_band_power
)
from utils import (
    compute_persistence_diagram, compute_landscape_values, 
    classify_landscapes
)

import ripser
import mne
from mne import EpochsArray, concatenate_raws
import pandas as pd

class tda:
    def __init__(self, raw = None, n_components=14, random_state=97, max_iter=100):
        # Now you can work with the loaded data
        if raw is not None :
            self.raw = raw
        else:
            self.raw = load_all_data()
        
        # ICA analysis
        self.n_components= n_components
        self.random_state=random_state
        self.max_iter=max_iter
        self.point_cloud = []
        self.landscapes = []

    def run_analysis(self):
        
        #plot_ica(self, self.raw, self.n_components, self.random_state, self.max_iter)
        
        # PSD band power
        fmin, fmax = 10, 20  # Alpha band 8, 12
        tmin, tmax = 0, 1000  # First 1000 seconds of the recording 0, 1000


        for i in range(len(self.raw)):
            # Compute the point cloud
            self.point_cloud.append(compute_psd_band_power(str(i), self.raw[i], fmin, fmax, tmin=None, tmax=None))
            print(self.point_cloud[-1].mean())

            # Compute persistence diagrams
            diagram = compute_persistence_diagram(self.point_cloud[i])
            # Define grid and compute landscape values
            grid = np.linspace(0, np.max(self.point_cloud[i]), 1000)
            self.landscapes.append(compute_landscape_values(diagram, grid))

            # Plot the persistence landscape
            plot_persistence_landscape(str(i), grid, self.landscapes[-1])
        
        # Classification
        classify_landscapes(self.landscapes)

