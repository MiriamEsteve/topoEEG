import numpy as np
from src.read_file import (load_all_data, logging)
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

class tda:
    def __init__(self, raw_AD=None, raw_CN=None, raw_FTD=None):
        # Now you can work with the loaded data
        if raw_AD is not None and raw_CN is not None and raw_FTD is not None:
            self.raw_AD = raw_AD
            self.raw_CN = raw_CN
            self.raw_FTD = raw_FTD
        else:
            raw_AD, raw_CN, raw_FTD = load_all_data()

    def run_analysis(self):
        sub_id = 'sub-001'
        plot_eeg_openneuro(self.raw_AD, '1_EEG_Data_sub' + str(sub_id) + '.jpg')
        
        diagrams = generate_persistence_diagrams(self.datas)
        plot_persistence_diagrams(diagrams, '2_Persistence_Diagrams.png')
        
        lifetimes = calculate_lifetime_diagrams(diagrams)
        plot_lifetime_diagrams(lifetimes, '3_Lifetime_Diagram.png')
        
        rips = ripser.Rips(maxdim=1, coeff=2)
        diagrams_h1 = [rips.fit_transform(data)[1] for data in self.datas]
        plot_persistence_images(diagrams_h1, '4_Persistence_Images.png')

        compute_gudhi_barycenter(diagrams_h1, '5_Calculate_Barycenter.png')

        classifiers = {
            'Logistic Regression': LogisticRegression(),
            'Support Vector Machine': SVC(),
            'Random Forest': RandomForestClassifier(),
            'Neural Network': MLPClassifier()
        }
        
        for name, clf in classifiers.items():
            X, y, report = perform_classification(clf)
            plot_classification(X, y, f'6_Classification_{name.replace(" ", "_")}.png', name)
            plot_evaluation_and_refinement(report, f'7_Evaluation_and_Refinement_{name.replace(" ", "_")}.png', name)

        create_flowchart()

