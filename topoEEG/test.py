import numpy as np
from analysis import tda
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import ripser
from read_file import load_all_data

def main():
    # Create object tda
    tda_obj = tda()

    # Step 1: Load EEG Data
    tda_obj.run_analysis()



if __name__ == "__main__":
    # Execute step by step
    main()
