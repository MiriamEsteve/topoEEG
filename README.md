# topoEEG

topoEEG is a Python-based analytical framework designed to process and analyze EEG data, integrating the MNE library with Topological Deep Learning (TDL) methods. This package is tailored for advanced neuroimaging research, particularly in exploring neurodegenerative conditions like Alzheimer’s disease, by capturing deeper insights into the underlying neural structures that traditional techniques may miss.


## Project Structure
```
├── Figures/
├── topoEEG/
├──├── data/
├──├── test/
├──├──├── test.py
├──├── __init__.py
├──├── read_file.py
├──├── analysis.py
├──├── data.py
├──├── plotting.py
├──└── utils.py
├── setup.py
└── README.md
LICENSE
```

## Installation

### Clone the Repository

```bash
git clone https://github.com/MiriamEsteve/topoEEG.git
cd topoEEG
```

### Install dependencies
Install the required dependencies using pip.

```bash
# pip install numpy matplotlib scipy scikit-learn persim ripser gudhi giotto-tda POT datalad
python ./topoEEG/setup.py install
```

## Project Setup
Ensure your directory structure looks like the above structure.

## Usage
To run the entire analysis workflow, navigate to the project directory and execute:
```bash
python test.py
```

This will execute each step of the analysis independently and save the corresponding images in the project directory.


## Step 1: Initialize the topoEEG Object
```python
    # Initialize the topoEEG object with raw EEG data
    topoEEG_obj = tda(raw = None, n_components=14, random_state=97, max_iter=100, grid_size = 10000)

```
In this step, the raw EEG data is loaded into the topoEEG object. The data format should be compatible with EEG file types like .set.

The topoEEG object is created, where:
- raw: The input EEG data (can be loaded later).
- n_components: Number of components to use in ICA (Independent Component Analysis).
- random_state: Sets the random seed for reproducibility.
- max_iter: Number of iterations for convergence in ICA.
- grid_size: Defines the grid resolution for persistence landscape computation.


## Step 2: Perform ICA (Independent Component Analysis)
```python
    # Perform ICA to remove artifacts like eye blinks and muscle noise
    topoEEG_obj.compute_ica()
```
ICA is used to separate noise sources from the EEG signal, specifically removing artifacts like eye blinks and muscle movements

## Step 3: Compute Power Spectral Density (PSD) Band Power
```python
    fmin, fmax = 10, 20
    topoEEG_obj.point_cloud = []

    for i in range(len(topoEEG_obj.raw)):
        topoEEG_obj.point_cloud.append(
            topoEEG_obj.compute_psd_band_power(str(i), topoEEG_obj.raw[i], fmin, fmax)
        )
```
The PSD (Power Spectral Density) is calculated for each EEG channel.

- fmin and fmax represent the frequency range to focus on (e.g., 10–20 Hz).
- For each EEG channel, the function compute_psd_band_power() computes the PSD values within this frequency range.


## Step 4: Compute Persistence Diagram
```python
diagram = topoEEG_obj.compute_persistence_diagram(topoEEG_obj.point_cloud[i])
```

The persistence diagram is computed from the PSD point cloud, providing a topological representation of the data. This step highlights key features in the data that may not be visible through traditional methods.

## Step 5: Compute Persistence Landscape Values
```python
grid = np.linspace(0, np.max(topoEEG_obj.point_cloud[i]), topoEEG_obj.grid_size)
topoEEG_obj.landscapes = topoEEG_obj.compute_landscape_values(diagram, grid)
```

The persistence landscape is calculated from the persistence diagram. The landscape transforms topological features into a format suitable for machine learning models.

## Step 6: Plot Persistence Landscape
```python
topoEEG_obj.plot_persistence_landscape(topoEEG_obj.landscapes)
```

Visualize the persistence landscape, which provides a clearer view of topological features extracted from the EEG data.

## Step 7: Classify Landscapes
```python
topoEEG_obj.classify_landscapes()
```

Finally, classify the landscapes using machine learning models such as Support Vector Machines (SVM), Random Forest, Logistic Regression, and Neural Networks.


## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- The TDA tutorial for inspiration on using Gudhi for lanscape calculations.
- The Scikit-learn library for providing powerful machine learning tools.
- A. Falco and M. Esteve thank the grant TED2021-129347B-C22 funded by Ministerio de Ciencia e Innovación/ Agencia Estatal de Investigación