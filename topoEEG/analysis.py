import numpy as np
from read_file import (load_all_data, logging)
from plotting import (
    plot_ica, plot_persistence_landscape, compute_psd_band_power
)
from utils import (
    compute_persistence_diagram, compute_landscape_values, 
    classify_landscapes, wassertein_distance
)

import ripser
import mne
from mne import EpochsArray, concatenate_raws
import pandas as pd

class tda:
    def __init__(self, raw = None, n_components=10, random_state=97, max_iter=100, grid_size = 10000):
        # Now you can work with the loaded data
        if raw is not None :
            self.raw = raw
        else:
            self.raw = load_all_data()
        
        # ICA analysis
        self.n_components= n_components
        self.random_state=random_state
        self.max_iter=max_iter
        self.grid_size = grid_size
        self.point_cloud = []
        self.landscapes = []

    def run_analysis(self):
        
        plot_ica(self, self.raw, self.n_components, self.random_state, self.max_iter)
        
        # PSD band power
        fmin, fmax = 10, 20  # Alpha band 8, 12

        for i in range(len(self.raw)):
            # Compute the point cloud
            self.point_cloud.append(compute_psd_band_power(str(i), self.raw[i], fmin, fmax, tmin=None, tmax=None))
            print(self.point_cloud[-1].mean())

            # Compute persistence diagrams
            diagram = compute_persistence_diagram(self.point_cloud[i])
            # Define grid and compute landscape values
            grid = np.linspace(0, np.max(self.point_cloud[i]), self.grid_size)

            value = compute_landscape_values(diagram, grid)
            self.landscapes.append(value * (10**20))  # Correct exponentiation
            
            # Plot the persistence landscape
            plot_persistence_landscape(str(i), grid, self.landscapes[-1])
        
        # Classification
        classify_landscapes(self.landscapes)

        # Calculate Wasserstein distance
        #distance_matrix = wassertein_distance(self.point_cloud)
        # If you have labels or identifiers for each diagram, it would be useful to use them as index and column names in the DataFrame
        #labels = [f"Diagram {i}" for i in range(len(distance_matrix))]

        # Convert to DataFrame
        #df = pd.DataFrame(distance_matrix, index=labels, columns=labels)
        #df.to_csv("./Figures/persistence_diagrams_distances.csv", index=True)

    def compute_ica(self, raw = None, n_components=14, random_state=97, max_iter=100):
        """
        Computes Independent Component Analysis (ICA) for EEG data.

        Parameters:
        - raw: Raw EEG data.
        - n_components: Number of components to extract.
        - random_state: Random seed for ICA.
        - max_iter: Maximum number of iterations.

        Returns:
        - Plot ICA object.
        """
        # Now you can work with the loaded data
        if raw is not None :
            self.raw = raw
        else:
            self.raw = load_all_data()

        plot_ica(self.raw, self.n_components, self.random_state, self.max_iter)


    def compute_psd_band_power(self, subj, raw, fmin=10, fmax=20):
        """
        Computes mean power spectral density (PSD) band power for each channel in an EEGLAB file.

        Parameters:
        - subj: Subject identifier.
        - raw: raw EEGLAB .set file.
        - fmin, fmax: Frequency band limits for the calculation of mean PSD (in Hz).

        Returns:
        - A vector of mean PSD band power values for each channel.
        """
        # Pick types of channels to include in the analysis (e.g., EEG channels)
        picks = self.raw.pick_types(eeg=True, meg=False, stim=False)
        
        # Compute the Power Spectral Density (PSD) for each channel using Welch's method
        psds = self.raw.compute_psd(method="welch", fmin=fmin, fmax=fmax)
        # Plotting
        fig = psds.plot()
        fig.savefig('./Figures/psd/psd_subj' + subj + '.png')

        # Extract the PSD data and frequency values
        psds, freqs = psds.get_data(return_freqs=True)

        point_cloud = psds.T

        print("Point cloud shape:", point_cloud.shape)

        return point_cloud
    
    def compute_persistence_diagram(self, point_cloud):
        """
        Computes the persistence diagram for a given point cloud.

        Parameters:
        - point_cloud: A point cloud representing the data.

        Returns:
        - Persistence diagram.
        """
        rips_complex = ripser.Rips(maxdim=2)
        diagrams = rips_complex.fit_transform(point_cloud)
        return diagrams
    
    
    def classify_landscapes(self, landscapes):
        """
        Classifies persistence landscapes using a classifier.

        Parameters:
        - landscapes: List of persistence landscapes.

        Returns:
        - Predicted labels.
        """
        # Generate labels
        labels = self.generate_labels()

        # Flatten the landscapes
        X = np.array(landscapes).reshape(len(landscapes), -1)

        # Train a classifier
        clf = self.RandomForestClassifier(random_state=42)
        clf.fit(X, labels)

        # Predict labels
        predicted_labels = clf.predict(X)

        # Compute accuracy
        accuracy = self.accuracy_score(labels, predicted_labels)
        print(f"Accuracy: {accuracy}")

        return predicted_labels