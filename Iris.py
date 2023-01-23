import numpy as np
import pandas as pd
import qiskit
from sklearn.model_selection import train_test_split

ALPHA = 0.1
QUANTILE = 3
TEST_SIZE = 0.4
TRAIN_SIZE = 0.6

def standardise(x):
    return (x-np.mean(x))/np.std(x)

def rescaleFeature(x):
    return (1-ALPHA/2)*(np.pi/QUANTILE)*standardise(x)

class Iris:
    def __init__(self, pathname):
        self.path = pathname
    
    def get_pathname(self):
        return self.path
    
    def get_dict(self):
        return {
        "Iris-setosa" : "00",
        "Iris-versicolor" : "01",
        "Iris-virginica" : "10",
        }    
    
    def build_circuit(self, theta, omega):
        qc = qiskit.QuantumCircuit(2)
        for i in range(4):
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
        return qc
    
    def get_df(self):
        path = self.get_pathname()
        df = pd.read_csv(path)
        attributes = df.columns[:-1]
        for x in attributes:
            df[x] = rescaleFeature(df[x])
        
        return df
    
    def get_sets(self):
        train_set, test_set = train_test_split(self.get_df(), test_size=TEST_SIZE, train_size=TRAIN_SIZE)
        train_set, test_set = pd.DataFrame(train_set), pd.DataFrame(test_set)
        
        return train_set, test_set
        
    