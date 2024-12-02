import numpy as np
from read_file import (load_all_data, logging)
from plotting import (
    plot_ica, plot_persistence_landscape, parallel_plot_psd_band_power
)
from utils import (
    utils_compute_persistence_diagram, compute_landscape_values, 
    classify_landscapes
)
import concurrent.futures

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


    def compute_plot_psd_band_power(self):
        """
        Computes and plots the PSD band power for all subjects in raw_clean.
        
        Returns:
        - List of point clouds for each subject.
        """
        # Ensure raw_clean is prepared as [(subj_id, raw_object), ...]
        if not isinstance(self.raw_clean, list) or not all(isinstance(x, tuple) and len(x) == 2 for x in self.raw_clean):
            raise ValueError("raw_clean must be a list of tuples [(subj_id, raw_object), ...].")

        # Compute PSD band power in parallel
        point_clouds = parallel_plot_psd_band_power(
            self.raw_clean, self.fmin, self.fmax, self.tmin, self.tmax
        )

        return point_clouds



    # Compute persistence diagram
    def compute_persistence_diagram(self, grid):
        """
        Computes the persistence diagram from a point cloud using GUDHI's Rips complex.
        
        Parameters:
        - point_cloud: numpy array representing the point cloud.
        
        Returns:
        - Persistence diagram (list of birth-death pairs).
        """
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
        classify_landscapes(self.landscapes)
    
    def run_analysis(self):
        
        self.raw_clean = self.compute_ica()
        
        # Step 4: Prepare raw_clean with subject IDs
        self.raw_clean = [(f"subj_{i}", raw) for i, raw in enumerate(self.raw_clean)]

        # Step 5: Compute PSD band power
        self.point_cloud = self.compute_plot_psd_band_power()

        # Print mean values for verification
        for i, point_cloud in enumerate(self.point_cloud):
            print(f"Subject {i} mean PSD band power: {point_cloud.mean()}")

        # Define grid and compute landscape values
        grid = np.linspace(0, np.max(self.point_cloud), self.grid_size)

        # Compute persistence diagram
        self.landscapes = self.compute_persistence_diagram(grid)
            
        for i in range(len(self.landscapes)):
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