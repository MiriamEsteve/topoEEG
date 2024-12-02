import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from gtda.homology import VietorisRipsPersistence
from persim import PersLandscapeExact
import matplotlib.pyplot as plt
import gudhi as gd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from persim import PersLandscapeExact
from plotting import (
    plot_classification, plot_evaluation_and_refinement
)
import numpy as np

def utils_compute_persistence_diagram(point_cloud):
    """
    Computes the persistence diagram from a point cloud using GUDHI's Rips complex.
    
    Parameters:
    - point_cloud: numpy array representing the point cloud.
    
    Returns:
    - Persistence diagram (list of birth-death pairs).
    """
    rips_complex = gd.RipsComplex(points=point_cloud, max_edge_length=2)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    persistence_diagram = simplex_tree.persistence()

    # Filter out points with infinite death time
    filtered_diagram = [(birth, death) for birth, death in persistence_diagram if death != float('inf')]
    
    return filtered_diagram

def compute_landscape_values(diag, grid):
    """
    Computes persistence landscape values from a persistence diagram.
    
    Parameters:
    - diag: Persistence diagram (list of tuples with (dim, (birth, death))).
    - grid: Grid over which to compute the landscape values.
    
    Returns:
    - Landscape values as a numpy array.
    """
    # Initialize landscape values as zero (same shape as grid)
    landscape_values = np.zeros_like(grid)

    for interval in diag:
        dim, (birth, death) = interval

        # Loop over the grid and update the landscape values
        for i, t in enumerate(grid):
            if birth < t < death:
                landscape_values[i] = max(landscape_values[i], min(t - birth, death - t))
    
    return landscape_values

def generate_labels():
    """
    Generates labels for the dataset based on class counts:
    AD: 36 samples
    CN: 30 samples
    FTD: 22 samples
    """
    labels = ['AD'] * 36 + ['CN'] * 30 + ['FTD'] * 22
    return np.array(labels)

def classify_landscapes(landscapes):
    # Generate labels ("AD", "CN", "FTD")
    y_str = generate_labels()  # You can define this function to return your labels.

    # Encode labels ("AD", "CN", "FTD") into numerical values (e.g., 0, 1, 2)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)

    # Verify the distribution of classes
    unique_classes, counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip(unique_classes, counts))}")

    # Ensure there are at least two distinct classes (should be 3 in this case)
    if len(unique_classes) < 2:
        raise ValueError("The dataset contains fewer than 2 classes.")

    # Prepare feature matrix from landscapes
    X = np.vstack(landscapes)  # Stack landscapes to create the feature matrix

    print(f"Feature matrix shape: {X.shape}")
    print(f"Label vector shape: {y.shape}")
    # Check if X and y have consistent dimensions
    if X.shape[0] != len(y):
        raise ValueError(f"Mismatch between feature matrix samples ({X.shape[0]}) and labels ({len(y)})")

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Support Vector Machine': SVC(probability=True),
        'Random Forest': RandomForestClassifier(),
        'Neural Network': MLPClassifier(max_iter=1000)
    }

    # Train and evaluate each classifier
    for name, clf in classifiers.items():
        # Fit the classifier
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Create a classification report with zero_division parameter
        report = classification_report(y_test, y_pred, target_names=["AD", "CN", "FTD"], output_dict=True, zero_division=0)
        print(f"Classifier: {name}")
        print(report)

        # Calculate the accuracy (optional)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")

        # Optionally, you can also print the precision for each class
        print(f"Precision (AD): {report['AD']['precision']:.4f}")
        print(f"Precision (CN): {report['CN']['precision']:.4f}")
        print(f"Precision (FTD): {report['FTD']['precision']:.4f}")

        # Plot the classification scatter plot
        plot_classification(X_test, y_test, f'./Figures/6_Classification_{name.replace(" ", "_")}.png', name)

        # Now pass this report dictionary to the plotting function
        plot_evaluation_and_refinement(report, f'./Figures/7_Evaluation_and_Refinement_{name.replace(" ", "_")}.png', name)
