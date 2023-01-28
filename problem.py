import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
from typing import Tuple

import Classifier

#Used to normalized data
ALPHA = 0.1 
QUANTILE = 3

#Split ratios
TEST_SIZE = 0.4
TRAIN_SIZE = 0.6



def standardise(x: pd.Series) -> pd.Series:
    """Standardise the row given

    Args:
        x (pd.Series): row in our dataset that we standardize

    Returns:
        values standardized
    """
    return (x-np.mean(x))/np.std(x)

def rescaleFeature(x: pd.Series) -> pd.Series: 
    """Rescale the row given

    Args:
        x (pd.Series): row in our dataset that we rescale

    Returns:
        values rescaled
    """
    return (1-ALPHA/2)*(np.pi/QUANTILE)*standardise(x)

class Problem:
    """Super class that define our all four problems. We have in this class the common method that define a specific problem.
    """
    def __init__(self) -> None:
        pass
    
    def get_dicinv(self):
        """Get the inverted dictionnary from the original dictionnary

        Returns:
            the inverted dictionnary
        """
        dict = self.get_dict()
        dicinv = {dict[k] : k for k in dict} 
        return dicinv
    
    def get_sets(self) -> Tuple:
        """Split the dataset in a train and a test one with the ratio 0.6 / 0.4

        Returns:
            train and test set
        """
        train_set, test_set = train_test_split(self.get_df(), test_size=TEST_SIZE, train_size=TRAIN_SIZE)
        train_set, test_set = pd.DataFrame(train_set), pd.DataFrame(test_set)
        
        return train_set, test_set

    def initialize(self, problem: object, problem_name: str) -> Tuple:
        """Initialize a classifier and the parameters of the problem object given. 

        Args:
            problem (object): the corresponding problem
            problem_name (str): problem name

        Returns:
            A tuple with a new classifier created and the parameters initialized
        """
        classifier = Classifier.Classifier()
        
        df = problem.get_df()
        dict_qbits = problem.get_dict()
        train_set, test_set = problem.get_sets()
        theta_init = problem.theta_init    
        
        parameters = {
            "df" : df, 
            "dict_qbits" : dict_qbits,
            "train_set" : train_set,
            "test_set" : test_set,
            "theta_init" : theta_init,
            problem_name : problem
            }
        
        return classifier, parameters

    def launch_train(self, problem: object, problem_name: str) -> np.ndarray:
        """Launch the training and print the time duration.

        Args:
            problem (object): the corresponding problem object
            problem_name (str): name of the problem

        Returns:
            the optimized parameter
        """
        classifier, parameters = self.initialize(problem, problem_name)
        print("Training the model...")
        start = time.time()
        
        theta_opti = classifier.train(
            parameters["train_set"], 
            parameters["theta_init"], 
            parameters["dict_qbits"], 
            parameters["df"], 
            parameters[problem_name]
        )
        
        end = time.time()
        minutes, seconds = divmod(end-start, 60)
        print("Training duration: {:0>2}min{:05.2f}s".format(int(minutes),seconds))
        return theta_opti

    def get_accuracy(self, problem: object, problem_name: str, theta_opti: np.ndarray, IBMQ: bool) -> None:
        """Print on the terminal the accuracy obtained on the test set.

        Args:
            problem (object): the problem object
            problem_name (str): name of the problem
            theta_opti (np.ndarray): optimized parameter
            IBMQ (bool): boolean value set to True if we want to use online quantum material
        """
        classifier, parameters = self.initialize(problem, problem_name)
        
        if problem_name == "XOR":
            dicinv = parameters[problem_name].get_dicinv_XOR()
        
        else:
            dicinv = parameters[problem_name].get_dicinv()    
        
        classifier.accuracy(parameters[problem_name], theta_opti, parameters["test_set"], dicinv, IBMQ)
        return
        
    