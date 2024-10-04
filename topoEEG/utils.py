import numpy as np
import ripser
import gudhi
from gudhi.wasserstein.barycenter import lagrangian_barycenter as bary
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from gtda.homology import VietorisRipsPersistence
from persim import plot_diagrams, PersistenceImager
import matplotlib.pyplot as plt
import gudhi as gd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import mne
from plotting import (
    plot_confusion_matrix, plot_classification, plot_evaluation_and_refinement
)


# Compute persistence diagram
def compute_persistence_diagram(point_cloud):
    """
    Computes the persistence diagram from a point cloud using GUDHI's Rips complex.
    
    Parameters:
    - point_cloud: numpy array representing the point cloud.
    
    Returns:
    - Persistence diagram.
    """
    rips_complex = gd.RipsComplex(points=point_cloud, max_edge_length=2)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    return simplex_tree.persistence()


def compute_landscape_values(diag, grid):
    """
    Computes persistence landscape values from a persistence diagram.
    
    Parameters:
    - diag: Persistence diagram.
    - grid: Grid over which to compute the landscape values.
    
    Returns:
    - Landscape values as a numpy array.
    """
    landscape_values = np.zeros_like(grid)
    for interval in diag:
        if interval[0] != 1:  # Focusing on H1 (loops) for simplicity
            continue
        birth, death = interval[1]
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

# Function to perform classification
def perform_classification(clf, X, y):
    """
    Fits the classifier and returns predictions and evaluation report.
    
    Parameters:
    - clf: The classifier to be used.
    - X: Feature matrix (input data).
    - y: Target labels (output data).
    
    Returns:
    - X: Feature matrix.
    - y: Target labels.
    - report: Classification report.
    """
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=46)

    # Fit the classifier
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Create a classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    return X, y, report

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Assuming these functions already exist:
# def plot_classification(X, y, filename, name)
# def plot_evaluation_and_refinement(report, filename, name)


def classify_landscapes(landscapes):
    print("##################################################")
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
    print(f"Feature train matrix shape: {X_train.shape}")
    print(f"Label train vector shape: {y_train.shape}")
    print(f"Feature test matrix shape: {X_test.shape}")
    print(f"Label test vector shape: {y_test.shape}")
    unique_classes, counts = np.unique(y_train, return_counts=True)
    print(f"Class distribution train: {dict(zip(unique_classes, counts))}")
    unique_classes, counts = np.unique(y_test, return_counts=True)
    print(f"Class distribution test: {dict(zip(unique_classes, counts))}")

    print("##################################################")

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
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0, output_dict=True)
        print(f"Classifier: {name}")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

        # Plot the classification scatter plot
        plot_classification(X_test, y_test, f'./Figures/6_Classification_{name.replace(" ", "_")}.png', name)

        # Plot the evaluation and refinement (classification report)
        report_str = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0)
        plot_evaluation_and_refinement(report_str, f'./Figures/7_Evaluation_and_Refinement_{name.replace(" ", "_")}.png', name)
