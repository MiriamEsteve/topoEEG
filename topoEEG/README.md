# topoEEG

topoEEG is a Python package designed for 

## Project Structure
project/
├── topoEEG/
│ ├── init.py
│ ├── analysis.py
│ ├── data.py
│ ├── plotting.py
│ └── utils.py
├── pyproject.toml
└── main.py

## Installation

### Prerequisites

Ensure you have Python installed. It's recommended to use a virtual environment to manage dependencies.

Install 'poetry'. Then, execute 'poetry add requests' to install dependencies.

### Clone the Repository

```bash
git clone https://github.com/MiriamEsteve/topoEEG.git
cd topoEEG
```

### Install dependencies
Install the required dependencies using pip.

```bash
pip install numpy matplotlib scipy scikit-learn persim ripser gudhi giotto-tda POT datalad
```
## Project Setup
Ensure your directory structure looks like the above structure.

## Usage
To run the entire analysis workflow, navigate to the project directory and execute:
```bash
python test.py
```

This will execute each step of the analysis independently and save the corresponding images in the project directory.

### Functions

#### Step 1: Plot Trajectory Data
```python
    
    # Step 1: Plot Trajectory Data
    data_gen = SimulatedTrajectoryData()
    brownian_data = data_gen.brownian_motion()
    levy_data = data_gen.levy_flight()
    spiral_data = data_gen.spiral_trajectory()
    circular_data = data_gen.circular_trajectory()

    # Plotting example trajectories
    plot_trajectories(brownian_data, 'Trajectory_Brownian_Motion', '1_Load_Trajectory_Brownian_Motion.png')
    plot_trajectories(levy_data, 'Trajectory_Levy_Flight', '1_Load_Trajectory_Levy_Flight.png')
    plot_trajectories(spiral_data, 'Trajectory_Spiral', '1_Load_Trajectory_Spiral.png')
    plot_trajectories(circular_data, 'Trajectory_Circular', '1_Load_Trajectory_Circular.png')
    ```

#### Step 2: Generate Persistence Diagrams
```python
# Initialize the TrajectoryAnalysis class
analysis = TrajectoryAnalysis()
    
diagrams = generate_persistence_diagrams(analysis.datas)
plot_persistence_diagrams(diagrams, '2_Persistence_Diagrams.png')
```

#### Step 3: Calculate and Plot Lifetime Diagrams
```python
lifetimes = calculate_lifetime_diagrams(diagrams)
plot_lifetime_diagrams(lifetimes, '3_Lifetime_Diagram.png')
```

#### Step 4: Generate Persistence Images
```python
rips_instance = ripser.Rips(maxdim=1, coeff=2)
diagrams_h1 = [rips_instance.fit_transform(data)[1] for data in analysis.datas]
plot_persistence_images(diagrams_h1, '4_Persistence_Images.png')
```

#### Step 5: Compute and Plot Barycenter
```python
compute_gudhi_barycenter(diagrams_h1, '5_Calculate_Barycenter.png')
```

#### Step 6: Perform Classification with Different Models
```python
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Support Vector Machine': SVC(),
    'Random Forest': RandomForestClassifier(),
    'Neural Network': MLPClassifier(max_iter=2000)  # Increased max_iter to 2000
}

for name, clf in classifiers.items():
    X, y, report = perform_classification(clf)
    plot_classification(X, y, f'6_Classification_{name.replace(" ", "_")}.png', name)
    plot_evaluation_and_refinement(report, f'7_Evaluation_and_Refinement_{name.replace(" ", "_")}.png', name)

```

#### Step 7: Create Flowchart
```python
create_flowchart()
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- The TDA tutorial for inspiration on using Gudhi for barycenter calculations.
- The Scikit-learn library for providing powerful machine learning tools.
- A. Falco and M. Esteve thank the grant TED2021-129347B-C22 funded by Ministerio de Ciencia e Innovación/ Agencia Estatal de Investigación