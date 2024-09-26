import matplotlib.pyplot as plt
import numpy as np
from persim import plot_diagrams, PersistenceImager
from PIL import Image

import mne
import matplotlib.pyplot as plt
from mne.datasets import fetch_dataset

def plot_ica(raw, n_components=4, random_state=97, max_iter=800):
    """
    Function to load and plot EEG data from OpenNeuro.

    Parameters:
    - raw: MNE Raw object
    
    The function will search for EEG files in BIDS format and plot the data.
    """

    #ICA EEG data
    ica= []

    for i in range(len(raw)):
        # set up and fit the ICA
        ica.append(mne.preprocessing.ICA(n_components, random_state, max_iter))
        ica[-1].fit(raw[i])

        figs = ica[-1].plot_properties(raw[i], show=False) #, picks=ica.exclude
        for idx, fig2 in enumerate(figs):
            fname = './Figures/ica_subj' + str(idx) +'.png'
            fig2.savefig(fname)

        # fig.grab().save('screenshot_full.png')
        print("ICA -> saved Fig" + str(i))
        

def compute_psd_band_power(case, subj, raw, fmin, fmax, tmin=None, tmax=None):
    """
    Computes mean power spectral density (PSD) band power for each channel in an EEGLAB file.

    Parameters:
    - raw: raw EEGLAB .set file.
    - fmin, fmax: Frequency band limits for the calculation of mean PSD (in Hz).
    - tmin, tmax: Time range within the EEG recording to consider (in seconds).

    Returns:
    - A vector of mean PSD band power values for each channel.
    """
    # If a time range is specified, crop the data to this range
    if tmin is not None and tmax is not None:
        raw.crop(tmin=tmin, tmax=tmax)
    
    # Pick types of channels to include in the analysis (e.g., EEG channels)
    picks = raw.pick_types(eeg=True, meg=False, stim=False)
    
    # Compute the Power Spectral Density (PSD) for each channel using Welch's method
    psds = raw.compute_psd(method="welch", fmin=fmin, fmax=fmax)
    # Plotting
    fig = psds.plot()
    fig.savefig('../Figures/' + case + 'psds_subj' + subj + '.png')

    # Extract the PSD data and frequency values
    psds, freqs = psds.get_data(return_freqs=True)

    point_cloud = psds.T

    print("Point cloud shape:", point_cloud.shape)

    return point_cloud

    

def plot_persistence_diagrams(diagrams, filename):
    """Plot multiple persistence diagrams with a legend for each, for an unspecified number of diagrams."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Generate a color map that has as many colors as diagrams
    colors = plt.cm.jet(np.linspace(0, 1, len(diagrams)))  # Uses a color map, can be changed to any other

    # Loop through each diagram and assign colors and labels dynamically
    for index, (diagram, color) in enumerate(zip(diagrams, colors)):
        label = f'EEG_data {index + 1}'  # Generating labels dynamically
        ax.scatter(diagram[:, 0], diagram[:, 1], color=color, label=label)  # Plot each diagram with a unique color

    ax.legend()  # This will display the legend using the labels defined
    ax.set_title("Persistence Diagrams")
    ax.set_xlabel('Birth')  # Typically the x-axis represents the birth time
    ax.set_ylabel('Death')  # Typically the y-axis represents the death time
    plt.savefig(filename)
    plt.close()

def plot_persistence_diagrams(diagrams, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.jet(np.linspace(0, 1, len(diagrams)))  # Uses a color map, can be changed to any other
    labels=[]
    for index, (diagram, color) in enumerate(zip(diagrams, colors)):
        labels.append(f'EEG_data {index + 1}')  # Generating labels dynamically
        plot_diagrams(diagram[0], ax=ax)
    handles = [plt.Line2D([], [], color=color, marker='o', linestyle='', markersize=10, label=label) for color, label in zip(colors, labels)]
    ax.legend(handles=handles)
    ax.set_title("2. Generate Persistence Diagrams")
    plt.savefig(filename)
    plt.close()


def plot_lifetime_diagrams(lifetimes, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, lifetime in enumerate(lifetimes):
        ax.scatter(np.arange(len(lifetime)), lifetime, s=5, label=f'EEG_data {i+1}')
    ax.set_title("3. Lifetime Diagram")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Lifetime")
    ax.legend()
    plt.savefig(filename)
    plt.close()

def plot_persistence_images(diagrams_h1, filename):
    """Generate and save persistence images from a list of persistence diagrams."""
    pimgr = PersistenceImager(pixel_size=0.1)
    pimgr.fit(diagrams_h1)  # Fit the range of persistence diagrams
    imgs = pimgr.transform(diagrams_h1)  # Transform diagrams to images
    
    # Handle fewer than three diagrams gracefully
    num_imgs = len(imgs)
    fig, axs = plt.subplots(1, max(1, num_imgs), figsize=(6 * num_imgs, 6))  # Adjust subplot size
    
    if num_imgs == 1:
        axs = [axs]  # Ensure axs is iterable when there's only one image

    # Plot each persistence image
    for i in range(num_imgs):
        ax = axs[i]
        img = imgs[i]
        cax = ax.imshow(img, cmap='viridis', origin='lower')
        ax.set_title(f'Persistence Image {i+1}')
        fig.colorbar(cax, ax=ax)  # Optional: add a colorbar for scale reference

    plt.savefig(filename)
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

def create_flowchart():
    image_paths = [
        ('1_Load_EEG_data_Data.png', 
        '1. Load EEG_data Data: Start by loading EEG_data data, which could include spatial coordinates and timestamps from sources like GPS or AIS systems.'),
        ('2_Persistence_Diagrams.png', 
        '2. Generate Persistence Diagrams: Transform the EEG_data data into persistence diagrams, capturing topological features that persist across various scales of the data.'),
        ('3_Lifetime_Diagram.png', 
        '3. Generate Lifetime Diagrams: Calculate and plot the lifetime of each feature in the persistence diagrams to understand their persistence and significance over different scales.'),
        ('4_Persistence_Images.png', 
        '4. Generate Persistence Images: Visualize the persistence diagrams as images, providing a different perspective on the topological features.'),
        ('5_Calculate_Barycenter.png', 
        '5. Calculate Barycenter: Compute the barycenter of the persistence diagrams to find a representative summary of the data sets.'),
        ('6_Classification_Logistic_Regression.png', 
        '6. Classification (Logistic Regression): Apply Logistic Regression to classify the trajectories based on the features derived from the persistence diagrams and their barycenters.'),
        ('6_Classification_Support_Vector_Machine.png', 
        '6. Classification (Support Vector Machine): Apply Support Vector Machine (SVM) to classify the trajectories based on the features derived from the persistence diagrams and their barycenters.'),
        ('6_Classification_Random_Forest.png', 
        '6. Classification (Random Forest): Apply Random Forest to classify the trajectories based on the features derived from the persistence diagrams and their barycenters.'),
        ('6_Classification_Neural_Network.png', 
        '6. Classification (Neural Network): Apply Neural Network to classify the trajectories based on the features derived from the persistence diagrams and their barycenters.'),
        ('7_Evaluation_and_Refinement_Logistic_Regression.png', 
        '7. Evaluation and Refinement (Logistic Regression): Assess the performance of the Logistic Regression model using metrics such as accuracy, precision, and recall. Refine the model based on the outcomes to improve classification results.'),
        ('7_Evaluation_and_Refinement_Support_Vector_Machine.png', 
        '7. Evaluation and Refinement (Support Vector Machine): Assess the performance of the SVM model using metrics such as accuracy, precision, and recall. Refine the model based on the outcomes to improve classification results.'),
        ('7_Evaluation_and_Refinement_Random_Forest.png', 
        '7. Evaluation and Refinement (Random Forest): Assess the performance of the Random Forest model using metrics such as accuracy, precision, and recall. Refine the model based on the outcomes to improve classification results.'),
        ('7_Evaluation_and_Refinement_Neural_Network.png', 
        '7. Evaluation and Refinement (Neural Network): Assess the performance of the Neural Network model using metrics such as accuracy, precision, and recall. Refine the model based on the outcomes to improve classification results.')
    ]

    fig, axs = plt.subplots(len(image_paths), 1, figsize=(10, 80))  # Adjusted the figure size for better fit

    for i, (img_path, description) in enumerate(image_paths):
        axs[i].axis('off')  # Ensure no axis is shown
        axs[i].text(0.5, 0.5, description, horizontalalignment='center', verticalalignment='center', transform=axs[i].transAxes, fontsize=10, wrap=True, family='monospace')

    plt.tight_layout()
    plt.savefig('Flowchart_with_Descriptions.png')
    #plt.show()  # Optional if you only want to save

