import numpy as np
from read_file import (load_all_data, logging)
from plotting import (
    plot_ica, parallel_plot_psd_band_power, plot_persistence_landscape, plot_persistence_image
)
from utils import (
    utils_compute_persistence_diagram, compute_landscape_values, 
    utils_classify_landscapes, utils_classify_persistence_images
)
import concurrent.futures
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
        self.fmin = fmin
        self.fmax = fmax
        self.tmin = tmin
        self.tmax = tmax
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


    def compute_plot_psd_band_power(self):
        """
        Computes and plots the PSD band power for all subjects in raw_clean.
        
        Returns:
        - List of point clouds for each subject.
        """
        # Step 4: Prepare raw_clean with subject IDs
        self.raw_clean = [(f"subj_{i}", raw) for i, raw in enumerate(self.raw_clean)]

        # Ensure raw_clean is prepared as [(subj_id, raw_object), ...]
        if not isinstance(self.raw_clean, list) or not all(isinstance(x, tuple) and len(x) == 2 for x in self.raw_clean):
            raise ValueError("raw_clean must be a list of tuples [(subj_id, raw_object), ...].")

        # Compute PSD band power in parallel
        point_clouds = parallel_plot_psd_band_power(
            self.raw_clean, self.fmin, self.fmax, self.tmin, self.tmax
        )

        return point_clouds


    def generate_labels(self):
        """
        Generate labels for each subject in the EEG data.

        Returns:
        - List of labels (e.g., ["AD", "CN", "FTD", ...]) corresponding to each subject.
        """
        # Define the labels based on subject order (this should match the number of subjects)
        labels = ["AD", "CN", "FTD", "CN", "AD", "CN", "FTD", "CN", "AD", "FTD"]  # Example labels
        # Repeat or truncate the list of labels to match the number of subjects in self.point_cloud
        num_subjects = len(self.point_cloud)  # This should be the number of subjects
        labels = (labels * (num_subjects // len(labels))) + labels[:(num_subjects % len(labels))]
        
        return labels
    
    # Compute persistence diagram
    def compute_persistence_diagram(self, grid):
        """
        Computes the persistence diagram from a point cloud using GUDHI's Rips complex.
        
        Parameters:
        - grid: numpy array representing the grid.
        
        Returns:
        - Persistence diagram (list of birth-death pairs).
        """
        # Generate labels ("AD", "CN", "FTD") for each subject
        y_str = self.generate_labels()  # You can define this function to return your labels.
        
        # Encode labels ("AD", "CN", "FTD") into numerical values (e.g., 0, 1, 2)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_str)
        
        # Verify the distribution of classes
        unique_classes, counts = np.unique(y, return_counts=True)        

        # Step 1: Compute persistence diagrams in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            persistence_diagrams = list(executor.map(utils_compute_persistence_diagram, self.point_cloud))

        # Step 2: Compute landscape values for each persistence diagram
        with concurrent.futures.ThreadPoolExecutor() as executor:
            landscape_values = list(executor.map(lambda diag: compute_landscape_values(diag, grid), persistence_diagrams))

        landscape_values = np.squeeze(landscape_values)  # Removes dimensions of size 1

        return landscape_values


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
        # Define grid and compute landscape values
        grid = np.linspace(0, np.max(self.point_cloud), self.grid_size)

        # Compute persistence diagram
        self.landscapes = self.compute_persistence_diagram(grid)

        for i in range(len(self.landscapes)):
            # Plot the persistence landscape
            self.plot_persistence_landscape(str(i), grid, self.landscapes[i])

        utils_classify_landscapes(self.landscapes)
    
    def plot_persistence_image(self, subj, data, spread=1.0, pixels=(30, 30)):
        plot_persistence_image(subj, data, spread, pixels)

    def classify_persistence_images(self):
        """
        Classifies persistence images using a classifier.

        Parameters:
        - persistence_images: List of persistence images.

        Returns:
        - Predicted labels.
        """
        utils_classify_persistence_images(self.point_cloud)


    def run_analysis(self):
        # Read files
        raw = load_all_data()

        # Step 2: Compute ICA
        self.raw_clean = self.compute_ica()

        # Step 3: Compute PSD band power
        self.point_cloud = self.compute_plot_psd_band_power()

        # Step 4: Classification
        self.classify_landscapes()
        self.classify_persistence_images()

