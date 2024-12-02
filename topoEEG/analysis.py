import numpy as np
from read_file import (load_all_data, logging)
from plotting import (
    plot_ica, plot_persistence_landscape, plot_psd_band_power
)
from utils import (
    utils_compute_persistence_diagram, compute_landscape_values, 
    classify_landscapes, wassertein_distance
)

import ripser
import mne
from mne import EpochsArray, concatenate_raws
import pandas as pd

class tda:
    def __init__(self, raw = None, n_components=10, random_state=97, max_iter=1000, grid_size = 10000, fmin=1, fmax=30, tmin=0, tmax=60):
        # Now you can work with the loaded data
        if raw is not None :
            self.raw = raw
        else:
            self.raw = load_all_data()
        
        # ICA analysis
        self.n_components= n_components
        self.random_state=random_state
        self.max_iter=max_iter
        self.fmin = 1
        self.fmax = 30
        self.tmin = 0
        self.tmax = 60
        self.grid_size = grid_size
        self.point_cloud = []
        self.landscapes = []

    def compute_ica(self):
        """
        Computes Independent Component Analysis (ICA) for EEG data.

        Returns:
        - Plot ICA object.
        - Cleaned raw data.
        """
       
        raw_clean = plot_ica(self.raw, self.n_components, self.random_state, self.max_iter)
        
        return raw_clean


    def compute_psd_band_power(self, subj, raw):
        """
        Computes mean power spectral density (PSD) band power for each channel in an EEGLAB file.

        Parameters:
        - subj: Subject identifier.
        - raw: raw EEGLAB .set file.
        - fmin, fmax: Frequency band limits for the calculation of mean PSD (in Hz).

        Returns:
        - A vector of mean PSD band power values for each channel.
        """       
        point_cloud = plot_psd_band_power(subj, raw, self.fmin, self.fmax, self.tmin, self.tmax)

        return point_cloud
    
    # Compute persistence diagram
    def compute_persistence_diagram(self, grid):
        """
        Computes the persistence diagram from a point cloud using GUDHI's Rips complex.
        
        Parameters:
        - point_cloud: numpy array representing the point cloud.
        
        Returns:
        - Persistence diagram (list of birth-death pairs).
        """
        # Compute persistence diagrams
        diagram = utils_compute_persistence_diagram(self.point_cloud[-1])

        value = compute_landscape_values(diagram, grid)
        return value * (10**20)  # Correct exponentiation
    
    def plot_persistence_landscape(self, subj, grid, landscape):
        plot_persistence_landscape(subj, grid, landscape)
    
    def classify_landscapes(self):
        """
        Classifies persistence landscapes using a classifier.

        Parameters:
        - landscapes: List of persistence landscapes.

        Returns:
        - Predicted labels.
        """
        classify_landscapes(self.landscapes)
    
    def run_analysis(self):
        
        self.raw_clean = self.compute_ica()
        
        for i, raw in enumerate(self.raw_clean):
            # Compute the point cloud
            self.point_cloud.append(self.compute_psd_band_power(str(i), raw))
            print(self.point_cloud[-1].mean())

            # Define grid and compute landscape values
            grid = np.linspace(0, np.max(self.point_cloud[-1]), self.grid_size)

            # Compute persistence diagram
            self.landscapes.append(self.compute_persistence_diagram(grid))
            
            # Plot the persistence landscape
            self.plot_persistence_landscape(str(i), grid, self.landscapes[-1])
        
        # Classification
        classify_landscapes(self.landscapes)

        # Calculate Wasserstein distance
        #distance_matrix = wassertein_distance(self.point_cloud)
        # If you have labels or identifiers for each diagram, it would be useful to use them as index and column names in the DataFrame
        #labels = [f"Diagram {i}" for i in range(len(distance_matrix))]

        # Convert to DataFrame
        #df = pd.DataFrame(distance_matrix, index=labels, columns=labels)
        #df.to_csv("./Figures/persistence_diagrams_distances.csv", index=True)