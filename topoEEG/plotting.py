import matplotlib.pyplot as plt
import numpy as np
from persim import plot_diagrams, PersistenceImager
from PIL import Image
import os
import mne
import matplotlib.pyplot as plt
from mne.datasets import fetch_dataset
from sklearn.metrics import confusion_matrix

def plot_ica(self, raw, n_components, random_state, max_iter):
    """
    Function to load and plot EEG data from OpenNeuro.

    Parameters:
    - raw: MNE Raw object
    
    The function will search for EEG files in BIDS format and plot the data.
    """
    if not os.path.exists('./Figures/ica/'):
        os.makedirs('./Figures/ica/')

    #ICA EEG data
    self.ica = []

    for i in range(len(raw)):
        # set up and fit the ICA
        self.ica.append(mne.preprocessing.ICA(n_components=n_components, random_state=random_state, max_iter=max_iter))
        self.ica[-1].fit(raw[i])

        figs = self.ica[-1].plot_properties(raw[i], show=False) #, picks=ica.exclude
        for idx, fig2 in enumerate(figs):
            fname = f'./Figures/ica/ica_subj{i}_component{idx}.png'
            fig2.savefig(fname)

        # fig.grab().save('screenshot_full.png')
        print("-----------------------------------------------> ICA -> saved Fig" + str(i))
        

def compute_psd_band_power(subj, raw, fmin, fmax, tmin=None, tmax=None):
    """
    Computes mean power spectral density (PSD) band power for each channel in an EEGLAB file.

    Parameters:
    - raw: raw EEGLAB .set file.
    - fmin, fmax: Frequency band limits for the calculation of mean PSD (in Hz).
    - tmin, tmax: Time range within the EEG recording to consider (in seconds).

    Returns:
    - A vector of mean PSD band power values for each channel.
    """
    
    if not os.path.exists('./Figures/psd/'):
        os.makedirs('./Figures/psd/')

    # If a time range is specified, crop the data to this range
    if tmin is not None and tmax is not None:
        raw.crop(tmin=tmin, tmax=tmax)
    
    # Pick types of channels to include in the analysis (e.g., EEG channels)
    picks = raw.pick_types(eeg=True, meg=False, stim=False)
    
    # Compute the Power Spectral Density (PSD) for each channel using Welch's method
    psds = raw.compute_psd(method="welch", fmin=fmin, fmax=fmax)
    # Plotting
    fig = psds.plot()
    fig.savefig('./Figures/psd/psd_subj' + subj + '.png')

    # Extract the PSD data and frequency values
    psds, freqs = psds.get_data(return_freqs=True)

    point_cloud = psds.T

    print("Point cloud shape:", point_cloud.shape)

    return point_cloud

    
def plot_persistence_landscape(subj, grid, landscape_values):
    """
    Plots the persistence landscape.
    
    Parameters:
    - grid: Grid over which the landscape is computed.
    - landscape_values: Computed landscape values.
    - title: Title of the plot.
    """
    
    if not os.path.exists('./Figures/landscape/'):
        os.makedirs('./Figures/landscape/')
    
    title="Persistence Landscape subj_" + subj
    plt.figure(figsize=(8, 6))
    plt.plot(grid, landscape_values)
    plt.title(title)
    plt.xlabel("Feature Space")
    plt.ylabel("Persistence")
    #plt.show()
    fname = './Figures/landscape/landscape_subj' + subj +'.png'
    plt.savefig(fname)


def plot_confusion_matrix(y_true, y_pred, classifier_name):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {classifier_name}')
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_true)))
    plt.xticks(tick_marks, np.unique(y_true), rotation=45)
    plt.yticks(tick_marks, np.unique(y_true))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'./Figures/confusion_matrix_{classifier_name.replace(" ", "_")}.png')
    plt.close()


def plot_classification(X, y, filename, name):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
    ax.set_title("6. Classification: " + name)
    plt.savefig(filename)
    plt.close()

def plot_evaluation_and_refinement(report, filename, name):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, report, horizontalalignment='center', verticalalignment='center', fontsize=12)
    ax.set_title("7. Evaluation and Refinement; " + name)
    ax.axis('off')
    plt.savefig(filename)
    plt.close()
