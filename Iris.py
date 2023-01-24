import numpy as np
import pandas as pd
import qiskit
from sklearn.model_selection import train_test_split
import time
import Classifier

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
    
    #Inverse the dict
    def get_dicinv(self):
        dict = self.get_dict()
        dicinv = {dict[k] : k for k in dict} 
        return dicinv    
    
    def build_circuit(self, theta, omega):
        qc = qiskit.QuantumCircuit(2,2)
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
    
    def prediction_dict(self, theta, omega):
        qc = self.build_circuit(theta, omega)
        qc.measure(range(2), range(2))
        
        job = qiskit.execute(qc, shots=NSHOTS, backend=qs)
        c = job.result().get_counts()
        c.pop('11', None)
        
        return c
    
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

def define_iris():
    #Init Iris Class
    iris = Iris("data/iris_csv.csv")
    
    #Get useful informations
    df = iris.get_df()
    dict_qbits = iris.get_dict()
    train_set, test_set = iris.get_sets()
    theta_init = np.random.uniform(0, 2*np.pi, 8)
    
    parameters = {
        "df" : df, 
        "dict_qbits" : dict_qbits,
        "train_set" : train_set,
        "test_set" : test_set,
        "theta_init" : theta_init,
        "iris" :iris
        }
    
    #Init Classifier
    classifier_iris = Classifier.Classifier()
    return classifier_iris, parameters

def train_iris():
    classifier_iris, parameters = define_iris()
    print("Training the model...")
    start = time.time()
    
    theta_opti = classifier_iris.train(
        parameters["train_set"], 
        parameters["theta_init"], 
        parameters["dict_qbits"], 
        parameters["df"], 
        parameters["iris"]
    )
    
    end = time.time()
    minutes, seconds = divmod(end-start, 60)
    print("Training duration: {:0>2}min{:05.2f}s".format(int(minutes),seconds))
    return theta_opti

    #save_results("results/iris_result.txt", theta_opti)

def get_iris_accuracy(theta_opti):
    classifier_iris, parameters = define_iris()
    dicinv = classifier_iris.get_dicinv()
    classifier_iris.accuracy(parameters["iris"], theta_opti, parameters["test_set"], dicinv)
    return
        
    