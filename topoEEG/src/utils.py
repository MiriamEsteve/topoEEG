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

def generate_persistence_diagrams(datas):
    VR = VietorisRipsPersistence(homology_dimensions=[0, 1], n_jobs=-1)
    return [VR.fit_transform(data[None, :, :]) for data in datas]

def calculate_lifetime_diagrams(diagrams):
    return [diagram[0][:, 1] - diagram[0][:, 0] for diagram in diagrams]

def proj_on_diag(x):
    return ((x[1] + x[0]) / 2, (x[1] + x[0]) / 2)

def plot_bary(b, diags, G, ax):
    for i in range(len(diags)):
        indices = G[i]
        n_i = len(diags[i])

        for (y_j, x_i_j) in indices:
            y = b[y_j]
            if y[0] != y[1]:
                if x_i_j >= 0:  # not mapped with the diag
                    x = diags[i][x_i_j]
                else:  # y_j is matched to the diagonal
                    x = proj_on_diag(y)
                ax.plot([y[0], x[0]], [y[1], x[1]], c='black',
                        linestyle="dashed")

    ax.scatter(b[:,0], b[:,1], color='purple', marker='d', label="barycenter (estim)")
    ax.legend()

def compute_gudhi_barycenter(diags, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['blue', 'green', 'red']
    labels = ['Trajectory 1', 'Trajectory 2', 'Trajectory 3']
    for diagram, color, label in zip(diags, colors, labels):
        plot_diagrams(diagram, ax=ax)
    handles = [plt.Line2D([], [], color=color, marker='o', linestyle='', markersize=10, label=label) for color, label in zip(colors, labels)]
    ax.legend(handles=handles)

    b, log = bary(diags, 
         init=0,
         verbose=True)  # we initialize our estimation on the first diagram (the red one.)
    G = log["groupings"]
    plot_bary(b, diags, G, ax=ax)
    plt.savefig(filename)
    plt.close()

def perform_classification(clf):
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    report = classification_report(y_test, predictions)
    return X, y, report
