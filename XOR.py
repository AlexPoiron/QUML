import numpy as np
import pandas as pd
import qiskit
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import Classifier

from proglearn.sims import generate_gaussian_parity

ALPHA = 0.1
QUANTILE = 3
TEST_SIZE = 0.4
TRAIN_SIZE = 0.6
NSHOTS = 400
qs = qiskit.Aer.get_backend('qasm_simulator')

def standardise(x):
    return (x-np.mean(x))/np.std(x)

def rescaleFeature(x):
    return (1-ALPHA/2)*(np.pi/QUANTILE)*standardise(x)

class XOR:
    def __init__(self):
        pass
    
    def get_dict(self):
        return {
        "1" : ["10","01"],
        "0" : ["00","11"]
        }
        
    def get_dicinv(self):
        dict = self.get_dict()
        dicinv = {}
        for k in dict:
            for i in dict[k]:
                dicinv.update({i : k})
        return dicinv

    #Generic methods for build a ciruict and measure it
    def build_circuit(self, theta, omega):
        qc = qiskit.QuantumCircuit(2)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        qc.rz(omega[0], 0)
        qc.rz(omega[1], 1)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        qc.cz(0, 1)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        qc.rz(theta[0], 0)
        qc.rz(theta[1], 1)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        qc.cz(0, 1)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        qc.rz(theta[2], 0)
        qc.rz(theta[3], 1)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        return qc
    
    def prediction_dict(self, theta, omega):
        qc = qiskit.QuantumCircuit(2, 2)
        qc.append(self.build_circuit(theta, omega), range(2))
        qc.measure(range(2), range(2))
        
        job = qiskit.execute(qc, shots=NSHOTS, backend=qs)
        c = job.result().get_counts()
        
        return c
    
    def get_df(self):
        X_rxor, y_rxor = generate_gaussian_parity(1000, angle_params=np.pi / 4)
        X_rxor0 = [values[0] for values in X_rxor]
        X_rxor1 = [values[1] for values in X_rxor]
        list_dict = {'Values0' : X_rxor0,
                     'Values1' : X_rxor1,
                     'class' : [str(value) for value in y_rxor]} 
        df = pd.DataFrame(list_dict)
        return df
    
    def get_sets(self):
        train_set, test_set = train_test_split(self.get_df(), test_size=TEST_SIZE, train_size=TRAIN_SIZE)
        train_set, test_set = pd.DataFrame(train_set), pd.DataFrame(test_set)
        return train_set, test_set

def define_XOR():
    #Init XOR Class
    xor = XOR()
    
    #Get useful informations
    df = xor.get_df()
    dict_qbits = xor.get_dict()
    train_set, test_set = xor.get_sets()
    theta_init = np.random.uniform(0, 2*np.pi, 4)
    
    parameters = {
        "df" : df, 
        "dict_qbits" : dict_qbits,
        "train_set" : train_set,
        "test_set" : test_set,
        "theta_init" : theta_init,
        "xor" :xor
        }
    
    #Init Classifier
    classifier_xor = Classifier.Classifier()
    return classifier_xor, parameters

def train_XOR():
    classifier_xor, parameters = define_XOR()
    print("Training the model...")
    start = time.time()
    
    theta_opti = classifier_xor.train(
        parameters["train_set"], 
        parameters["theta_init"], 
        parameters["dict_qbits"], 
        parameters["df"], 
        parameters["xor"]
    )
    
    end = time.time()
    minutes, seconds = divmod(end-start, 60)
    print("Training duration: {:0>2}min{:05.2f}s".format(int(minutes),seconds))
    return theta_opti

    #save_results("results/iris_result.txt", theta_opti)

def get_XOR_accuracy(theta_opti):
    classifier_xor, parameters = define_XOR()
    dicinv = parameters["xor"].get_dicinv()
    classifier_xor.accuracy(parameters["xor"], theta_opti, parameters["test_set"], dicinv)

