import pandas as pd
import numpy as np
import qiskit
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import time
import Classifier

ALPHA = 0.1
QUANTILE = 3
TEST_SIZE = 0.4
TRAIN_SIZE = 0.6
NSHOTS = 400
qs = qiskit.Aer.get_backend('qasm_simulator')

class Quaternary:
    def __init__(self):
        self.name = "Quaternary"

    def get_dict(self):
        return {
            "0" : "00",
            "1" : "01",
            "2" : "10",
            "3" : "11"
            }
    
    def get_dicinv(self):
        dict = self.get_dict()
        dicinv = {dict[k] : k for k in dict} 
        return dicinv
    
    def build_circuit(self, theta, omega):
        qc = qiskit.QuantumCircuit(2)
        for i in range(5):
            if i : qc.cz(0, 1)
            qc.rx(np.pi/2, 0)
            qc.rx(np.pi/2, 1)
            qc.rz(omega[(2*i) % 4], 0)
            qc.rz(omega[(2*i+1) % 4], 1)
            qc.rx(np.pi/2, 0)
            qc.rx(np.pi/2, 1)
            qc.cz(0, 1)
            qc.rx(np.pi/2, 0)
            qc.rx(np.pi/2, 1)
            qc.rz(theta[2*i], 0)
            qc.rz(theta[2*i+1], 1)
            qc.rx(np.pi/2, 0)
            qc.rx(np.pi/2, 1)
            if i == 1 or i == 3:
                qc.cz(0, 1)
                qc.rx(np.pi/2, 0)
                qc.rx(np.pi/2, 1)
                qc.rz(theta[2*i+2], 0)
                qc.rz(theta[2*i+3], 1)
                qc.rx(np.pi/2, 0)
                qc.rx(np.pi/2, 1)
                i+=1
        return qc

    def prediction_dict(self, theta, omega):
        qc = qiskit.QuantumCircuit(2, 2)
        qc.append(self.build_circuit(theta, omega), range(2))
        qc.measure(range(2), range(2))
        
        job = qiskit.execute(qc, shots=NSHOTS, backend=qs)
        c = job.result().get_counts()       
        return c

    def get_df(self):
        X,y = make_classification(n_samples=5000, n_features=4)
        df = pd.DataFrame(X)
        df['class'] = y
        df['class'] = df['class'].astype(str)
        return df

    def get_sets(self):
        train_set, test_set = train_test_split(self.get_df(), test_size=TEST_SIZE, train_size=TRAIN_SIZE)
        train_set, test_set = pd.DataFrame(train_set), pd.DataFrame(test_set)
        
        return train_set, test_set

def define_quaternary():
    #Init XOR Class
    quaternary = Quaternary()
    
    #Get useful informations
    df = quaternary.get_df()
    dict_qbits = quaternary.get_dict()
    train_set, test_set = quaternary.get_sets()
    theta_init = np.random.uniform(0, 2*np.pi, 12)
    
    parameters = {
        "df" : df, 
        "dict_qbits" : dict_qbits,
        "train_set" : train_set,
        "test_set" : test_set,
        "theta_init" : theta_init,
        "quaternary" : quaternary
        }
    
    #Init Classifier
    classifier_quaternary = Classifier.Classifier()
    return classifier_quaternary, parameters

def train_quaternary():
    classifier_quaternary, parameters = define_quaternary()
    print("Training the model...")
    start = time.time()
    
    theta_opti = classifier_quaternary.train(
        parameters["train_set"], 
        parameters["theta_init"], 
        parameters["dict_qbits"], 
        parameters["df"], 
        parameters["quaternary"]
    )
    
    end = time.time()
    minutes, seconds = divmod(end-start, 60)
    print("Training duration: {:0>2}min{:05.2f}s".format(int(minutes),seconds))
    return theta_opti

    #save_results("results/iris_result.txt", theta_opti)

def get_quaternary_accuracy(theta_opti):
    classifier_quaternary, parameters = define_quaternary()
    dicinv = parameters["quaternary"].get_dicinv()
    classifier_quaternary.accuracy(parameters["quaternary"], theta_opti, parameters["test_set"], dicinv)