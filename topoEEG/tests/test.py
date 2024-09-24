import numpy as np
from src.read_file import load_all_data
from src.plotting import (
    plot_eeg_openneuro, plot_persistence_diagrams, plot_lifetime_diagrams, 
    plot_persistence_images, plot_classification, plot_evaluation_and_refinement, create_flowchart
)
from src.utils import (
    generate_persistence_diagrams, calculate_lifetime_diagrams, 
    compute_gudhi_barycenter, perform_classification
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import ripser

def main():
    # Step 1: Load EEG Data
    raw_AD, raw_CN, raw_FTD = load_all_data()

if __name__ == "__main__":
    # Execute step by step
    main()
