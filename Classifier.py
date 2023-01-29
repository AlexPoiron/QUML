import numpy as np
import pandas as pd
import qiskit 
import scipy as sp
import swifter
from utils import create_logs, create_logs_circuit

NSHOTS = 1500
qs = qiskit.Aer.get_backend('qasm_simulator')

class Classifier:
    """Classifier class. This is in this class that we define methods used for training, final prediciton and accuracy
    """
    def __init__(self):
        pass
    
    #Loss function
    def loss_batch(self, theta: np.ndarray, batch: pd.Series, dict: dict, df: pd.DataFrame, problem: object) -> float:
        """Loss function

        Args:
            theta (np.ndarray): parameter to optimize
            batch (pd.Series): the train set
            dict (dict): the according dict to the problem
            df (pd.Dataframe): the original dataframe
            problem (object): the problem
        """
        def loss(theta, df, attributes, problem):
            omega = df[attributes].values
            label = dict[df["class"]]
            c = problem.prediction_dict(theta, omega)

            if type(label) == list:
                label = label[0]
                
            if label in c:
                e = np.exp(c[label]/NSHOTS)
            else:
                e = 1
        
            s = np.exp(np.array(list(c.values()))/NSHOTS).sum()
            return -np.log(e/s)
        
        attributes = df.columns[:-1]
        
        s = batch.swifter.apply(
        lambda data : loss(theta, data, attributes, problem),
        axis=1
        )
        print("Loss value:", s.mean())
        
        #Save loss value in the logs
        log = "Loss value: " + str(s.mean())
        create_logs(problem.name, True, [log])

        return s.mean()
    
    #Train method
    def train(self, train_set: pd.DataFrame, theta_init: np.ndarray, dict: dict, df: pd.DataFrame, problem: object) -> np.ndarray:
        """Train function used to get the optimized parameter

        Args:
            train_set (pd.DataFrame): the train set
            theta_init (np.ndarray): parameter initialized
            dict (dict): dict of the problem
            df (pd.DataFrame): full dataframe
            problem (object): the according problem objet

        Returns:
            theta: the optimized parameter
        """
        #We used COBYLA method to optimize theta
        opt = sp.optimize.minimize(fun = lambda theta : self.loss_batch(theta, train_set, dict, df, problem), x0=theta_init, method='COBYLA', tol=1e-4)
        print("-"*20)
        print("Optimal parameter Theta:", opt.x)
        
        #Save optimal parameter in the logs
        logs = ["-"*20, "Optimal parameter Theta: " + str(opt.x)]
        create_logs(problem.name, True, logs)
        
        problem.has_trained = True
        
        return opt.x
    

    def prediction(self, theta: np.ndarray, omega: pd.Series, problem: object, IBMQ: bool) -> str:
        """Get the prediction

        Args:
            theta (np.ndarray): optimized parameter
            omega (pd.Series): row in our test set
            problem (object): the problem object
            IBMQ (bool): set to True if we want to use online quantic material

        Returns:
            The key in the dict which is the prediction
        """
        def argmaxDict(c):
            return max(c, key=c.get)
        
        #If we decide to use online quantic material
        if IBMQ:
            return argmaxDict(problem.prediction_dict_IBMQ(theta, omega))
        return argmaxDict(problem.prediction_dict(theta, omega))

    def accuracy(self, problem: object, theta_opti: np.ndarray, test: pd.DataFrame, dicinv: dict, IBMQ: bool) -> None:
        """Get the accuracy on the test set

        Args:
            problem (object): the problem
            theta_opti (np.ndarray): optimized parameter
            test (pd.DataFrame): test set
            dicinv (dict): inverted dictionnary
            IBMQ (bool): set to True if we want to use online quantic material
        """
        print("-"*20)
        print("Compute the accuracy...")
        nb = 10
        accuracies = []
        for _ in range(nb):
            test["predicted"] = test.apply(lambda row : dicinv[self.prediction(theta_opti, row, problem, IBMQ)], axis=1)
            acc = ((sum(np.array(test["class"] == test["predicted"]))/len(test))*100).round(2)
            accuracies.append(acc)
            
        total_accuracy = sum(accuracies) / nb
        
        #Save the accuracy in the logs
        logs = ["-"*20, "Compute the accuracy..."]
        logs.append("Acurracy of the " + problem.name + " circuit: " + str(total_accuracy) + "%")
        create_logs(problem.name, problem.has_trained, logs)
          
        print("Acurracy of the", problem.name, "circuit: ", sum(accuracies) / nb, '%')
        
        #Save the circuit in a log file
        create_logs_circuit(problem)