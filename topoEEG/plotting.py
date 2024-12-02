import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-GUI backend
import matplotlib.pyplot as plt
from mne.datasets import fetch_dataset
from sklearn.metrics import confusion_matrix
import os
import mne
from mne.preprocessing import ICA
from concurrent.futures import ThreadPoolExecutor
import os

# Set log level to 'warning' to suppress less critical messages
mne.set_log_level('warning')

def apply_ica(raw, n_components, random_state, max_iter, subject_idx, fmin=1.0, fmax=30.0, tmin=0, tmax=60):
    """
    Helper function to apply ICA to a single subject's data.
    """
    # Reduce sampling frequency to 100 Hz
    raw.resample(sfreq=100)
    raw.pick_types(eeg=True)
    
    # Use the first 60 seconds of the data
    raw_filtered = raw.copy().crop(tmin=tmin, tmax=tmax)

    # Apply high-pass filter (recommended for ICA)
    raw_filtered.filter(l_freq=fmin, h_freq=fmax)

    # Ensure n_components doesn't exceed the number of available channels
    n_components = min(n_components, len(raw_filtered.info['ch_names']))

    # Set up and fit ICA
    ica = ICA(n_components=n_components, random_state=random_state, max_iter=max_iter)
    ica.fit(raw_filtered)

    # Exclude artifact components based on custom criteria
    components = ica.get_components()
    for idx, component in enumerate(components):
        peak_to_peak = component.max() - component.min()
        threshold_value = 200e-6  # Example threshold for exclusion
        if peak_to_peak > threshold_value and idx < n_components:
            ica.exclude.append(idx)
    print(f" ----- Excluded indices: {ica.exclude}")
    print(f"------ Number of ICA components: {ica.n_components_}")

    # Apply ICA to clean the data
    raw_cleaned = ica.apply(raw_filtered)

    print(f"Subject {subject_idx+1}: ICA applied successfully.")

    # Plot ICA component properties
    try:
        figs = ica.plot_properties(raw_cleaned, show=False, picks=range(10))
    except IndexError as e:
        print(f"Error plotting ICA properties for Subject {subject_idx+1}: {e}")
        print(f"Excluded components: {ica.exclude}")
        print(f"Number of ICA components: {ica.n_components_}")
        raise e

    # Save plots for ICA components
    for idx, fig in enumerate(figs):
        fname = f'./Figures/ica/ica_subj{subject_idx+1}_component{idx}.png'
        fig.savefig(fname)
        plt.close(fig)  # Close the figure after saving

    # Reset the exclusion list for the next subject
    ica.exclude = []

    print(f"Subject {subject_idx+1}: ICA components saved.")
    return subject_idx, ica, raw_cleaned


def plot_ica(raw_list, n_components, random_state, max_iter):
    """
    Function to apply ICA and plot EEG components for each subject from a list of MNE Raw objects in parallel.
    """
    # Create directory for saving plots if not already present
    if not os.path.exists('./Figures/ica/'):
        os.makedirs('./Figures/ica/')

    # Use ProcessPoolExecutor for ICA computation to take advantage of multiple cores
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(apply_ica, raw, n_components, random_state, max_iter, idx)
            for idx, raw in enumerate(raw_list)
        ]
        results = [future.result() for future in futures]  # Wait for all tasks to finish

    print("ICA processing for all subjects completed.")

    # Return cleaned data for all subjects
    return [result[2] for result in results]


def plot_psd_band_power(subj, raw, fmin=1, fmax=30, tmin=0, tmax=60):
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

    # Compute the Power Spectral Density (PSD) for each channel using Welch's method
    psds = raw.compute_psd(method="welch", fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax,        
                            n_fft=4096,      # Increase FFT points for higher frequency resolution
                            n_overlap=2048,  # Overlap for better averaging
                            window="hamming" # Window function for spectral leakage control
                        )
    # Plotting
    fig = psds.plot()
    fig.savefig('./Figures/psd/psd_subj' + subj + '.png')

    # Extract the PSD data (shape = n_channels, n_frequencies)
    psds_data = psds.get_data()  # Shape: (n_channels, n_frequencies)
    
    # Now, create the point cloud as the PSD data for each channel
    # The point cloud should be the PSD across all frequencies for each channel
    point_cloud = psds_data.T  # Transpose to (n_frequencies, n_channels)

    # Print shape of the point cloud for verification
    print("Point cloud shape:", point_cloud.shape)

    return point_cloud

    
def plot_persistence_landscape(subj, grid, landscape_values):
    """
    Plots the persistence landscape.
    
    Parameters:
    - grid: Grid over which the landscape is computed.
    - landscape_values: Computed landscape values.
    - title: Title of the plot.

    Returns:
    - Plot of the persistence landscape.

    The function will save the plot in the Figures/landscape/ directory.
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

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def plot_classification(X, y, filename, name):
    """
    Plots the classification results for the three classes (AD, CN, FTD).

    Parameters:
    - X: Feature matrix (n_samples, n_features) for test data.
    - y: True labels for the test data.
    - filename: Path to save the plot.
    - name: Name of the classifier for the title.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define class colors and labels
    class_names = ['AD', 'CN', 'FTD']
    colors = ['red', 'green', 'blue']  # You can choose any colors you prefer

    # Create a scatter plot with distinct colors for each class
    for i, class_name in enumerate(class_names):
        # Get indices for each class
        class_indices = np.where(y == i)[0]
        ax.scatter(X[class_indices, 0], X[class_indices, 1], 
                   color=colors[i], label=class_name, alpha=0.7, edgecolor='k')

    # Create legend
    ax.legend(title='Classes')

    # Set title and labels
    ax.set_title(f"6. Classification: {name}")
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

    # Save the figure
    plt.savefig(filename)
    plt.close()

def plot_evaluation_and_refinement(report, filename, name):
    """
    Plots the evaluation metrics from the classification report.

    Parameters:
    - report: Classification report as a dictionary.
    - filename: Path to save the plot.
    - name: Name of the classifier for the title.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare to visualize metrics
    metrics = ['precision', 'recall', 'f1-score', "support"]
    class_names = ["AD", "CN", "FTD"]  # Corresponding to classes 0, 1, 2
    y_pos = 0.8  # Starting Y position for the first metric

    # Display the metrics for each class
    ax.text(0.1, y_pos, "Metrics:", horizontalalignment='left', verticalalignment='center', fontsize=14)
    y_pos -= 0.05  # Shift position slightly down for metric values

    # Loop through classes and metrics
    for class_name in class_names:
        ax.text(0.1, y_pos, f"{class_name}:", horizontalalignment='left', verticalalignment='center', fontsize=12)
        
        for metric in metrics:
            # Accessing report using class names
            metric_value = report.get(class_name, {}).get(metric, "N/A")
            if isinstance(metric_value, float):
                ax.text(0.3, y_pos, f"{metric.capitalize()}: {metric_value:.2f}", 
                        horizontalalignment='left', verticalalignment='center', fontsize=12)
            else:
                ax.text(0.3, y_pos, f"{metric.capitalize()}: N/A", 
                        horizontalalignment='left', verticalalignment='center', fontsize=12)
            
            y_pos -= 0.05  # Move down for each metric

        y_pos -= 0.1  # Add extra space between classes

    # Add the title and remove axis
    ax.set_title(f"7. Evaluation and Refinement: {name}", fontsize=16)
    ax.axis('off')  # Hide the axes

    # Save the figure
    plt.savefig(filename)
    plt.close()

    
def plot_evaluation_and_refinement_(report, filename, name):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, report, horizontalalignment='center', verticalalignment='center', fontsize=12)
    ax.set_title("7. Evaluation and Refinement; " + name)
    ax.axis('off')
    plt.savefig(filename)
    plt.close()
