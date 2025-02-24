import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import gudhi as gd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from plotting import (
    plot_classification, plot_evaluation_and_refinement, plot_persistence_image, assign_precision
)
from persim import PersImage
from ripser import ripser
from sklearn.preprocessing import StandardScaler


def utils_compute_persistence_diagram(point_cloud):
    """
    Computes the persistence diagram from a point cloud using GUDHI's Rips complex.
    """
    rips_complex = gd.RipsComplex(points=point_cloud, max_edge_length=1)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    persistence_diagram = simplex_tree.persistence()

    # Filter out points with infinite death time
    return [(birth, death) for birth, death in persistence_diagram if death != float('inf')]


def compute_landscape_values(diag, grid):
    """
    Computes persistence landscape values from a persistence diagram.
    """
    landscape_values = np.zeros_like(grid)

    for interval in diag:
        dim, (birth, death) = interval
        for i, t in enumerate(grid):
            if birth < t < death:
                landscape_values[i] = max(landscape_values[i], min(t - birth, death - t))
    
    return landscape_values


def generate_labels():
    """
    Generates labels for the dataset.
    """
    return np.array(['AD'] * 36 + ['CN'] * 30 + ['FTD'] * 22)


def preprocess_and_split_data(features):
    """
    Preprocess data: splits into training and testing sets, and applies resampling.
    """
    # Generate and encode labels
    y_str = generate_labels()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)

    # Verify class distribution
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=42)

    # Resample to balance the dataset (SMOTE and Under-sampling)
    smote = SMOTE(random_state=52)
    undersample = RandomUnderSampler(random_state=52)

    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    X_resampled, y_resampled = undersample.fit_resample(X_resampled, y_resampled)

    print(f"Resampled class distribution: {Counter(y_resampled)}")

    return X_resampled, X_test, y_resampled, y_test


def perform_grid_search(classifiers, param_grids, X_resampled, y_resampled):
    """
    Perform grid search for hyperparameter tuning on classifiers.
    """
    best_classifiers = {}

    for clf_name, clf in classifiers.items():
        grid_search = GridSearchCV(clf, param_grids[clf_name], cv=5)
        grid_search.fit(X_resampled, y_resampled)
        best_classifiers[clf_name] = grid_search.best_estimator_

    return best_classifiers


def evaluate_classifiers(classifiers, X_test, y_test, name_type):
    """
    Evaluate classifiers on the test data and print classification reports.
    """
    for name, clf in classifiers.items():
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=['AD', 'CN', 'FTD'], output_dict=True, zero_division=0)

        name = name_type + " " + name
        print(f"Classifier: {name}")
        assign_precision(report=report, key='AD')
        assign_precision(report=report, key='CN')
        assign_precision(report=report, key='FTD')

        # Print the updated report
        print(f"Precision (AD): {report['AD']['precision']:.4f}")
        print(f"Recall (AD): {report['AD']['recall']:.4f}")
        print(f"F1-Score (AD): {report['AD']['f1-score']:.4f}")

        plot_classification(X_test, y_test, f'./Figures/6_Classification_{name.replace(" ", "_")}.png', name)
        plot_evaluation_and_refinement(report, f'./Figures/7_Evaluation_and_Refinement_{name.replace(" ", "_")}.png', name)



def utils_classify_landscapes(landscapes):
    """
    Classify persistence landscapes using multiple classifiers.
    """
    # Prepare feature matrix from landscapes
    X = np.vstack(landscapes)

    #print(f"Feature matrix shape: {X.shape}")

    # Preprocess data (splitting and resampling)
    X_resampled, X_test, y_resampled, y_test = preprocess_and_split_data(X)

    # Define classifiers and parameter grids
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=5000, class_weight='balanced'),
        'Support Vector Machine': SVC(probability=True, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(class_weight='balanced'),
        'Neural Network': MLPClassifier(max_iter=5000, early_stopping=True, validation_fraction=0.1, learning_rate='adaptive', random_state=42)
    }

    param_grids = {
        'Logistic Regression': {'C': [0.1, 1, 10], 'penalty': ['l2'], 'solver': ['liblinear']},
        'Random Forest': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
        'Support Vector Machine': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
        'Neural Network': {'hidden_layer_sizes': [(50,), (100,), (150,)], 'activation': ['relu', 'tanh'], 'solver': ['adam', 'sgd']}
    }

    # Perform grid search for best models
    best_classifiers = perform_grid_search(classifiers, param_grids, X_resampled, y_resampled)

    # Evaluate classifiers
    evaluate_classifiers(best_classifiers, X_test, y_test, "landscape")


def utils_compute_persistence_image(data, spread=1.0, pixels=(20, 20)):
    """
    Compute the persistence image from the input data.
    
    Parameters:
    - data: Input data (point cloud or feature matrix).
    - spread: Float. The spread parameter for the PersImage class.
    - pixels: Tuple of ints. Resolution of the persistence image.
    
    Returns:
    - Persistence image (numpy array).
    """
    # Compute the persistence diagram using Ripser
    persistence_diagram = ripser(data)['dgms'][0]  # Assuming H0 (connected components)
    
    # Handle empty persistence diagrams
    if persistence_diagram.size == 0:
        print("Empty persistence diagram, returning a zero persistence image.")
        return np.zeros(pixels)

    # Replace inf or NaN values in the persistence diagram
    valid_mask = np.isfinite(persistence_diagram[:, 1])
    if not np.any(valid_mask):
        print("No valid finite values in persistence diagram, returning a zero persistence image.")
        return np.zeros(pixels)

    if np.any(~valid_mask):
        print("Found invalid (inf/NaN) values in persistence diagram, replacing them with a large value.")
        persistence_diagram[~valid_mask, 1] = np.max(persistence_diagram[valid_mask, 1]) + 1

    # Ensure the persistence diagram is in the correct format
    if persistence_diagram.shape[1] != 2:
        raise ValueError("The persistence diagram should have shape (n_features, 2), where each row is (birth, death).")

    # Validate spread and pixels
    if spread <= 0:
        raise ValueError("Spread must be a positive value.")
    if not (isinstance(pixels, tuple) and len(pixels) == 2 and all(p > 0 for p in pixels)):
        raise ValueError("Pixels must be a tuple of two positive integers.")

    # Compute the persistence image using PersImage
    pers_image_transformer = PersImage(spread=spread, pixels=pixels)
    
    try:
        # Transform the persistence diagram to a persistence image
        persistence_image = pers_image_transformer.transform(persistence_diagram)
    except Exception as e:
        print(f"Error during persistence image transformation: {e}")
        return np.zeros(pixels)

    return persistence_image

def utils_classify_persistence_images(data, spread=1.0, pixels=(20, 20)):
    """
    Classify data using persistence images.
    """
    # Compute persistence images for the input data
    persistence_images = []
    for i, sample in enumerate(data):
        img = utils_compute_persistence_image(sample, spread=spread, pixels=pixels)
        persistence_images.append(img.flatten())  # Flatten the image for classification

        plot_persistence_image(
            subj=str(i),
            persistence_image=img
        )

    # Preprocess data (splitting and resampling)
    X_resampled, X_test, y_resampled, y_test = preprocess_and_split_data(np.array(persistence_images))

    # Define classifiers and parameter grids
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=5000, class_weight='balanced'),
        'Support Vector Machine': SVC(probability=True, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(class_weight='balanced'),
        'Neural Network': MLPClassifier(max_iter=5000, early_stopping=True, validation_fraction=0.1, learning_rate='adaptive', random_state=42)
    }

    param_grids = {
        'Logistic Regression': {'C': [0.1, 1, 10], 'penalty': ['l2'], 'solver': ['liblinear']},
        'Random Forest': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
        'Support Vector Machine': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
        'Neural Network': {'hidden_layer_sizes': [(50,), (100,), (150,)], 'activation': ['relu', 'tanh'], 'solver': ['adam', 'sgd']}
    }

    # Perform grid search for best models
    best_classifiers = perform_grid_search(classifiers, param_grids, X_resampled, y_resampled)

    # Evaluate classifiers
    evaluate_classifiers(best_classifiers, X_test, y_test, "persistence image")

