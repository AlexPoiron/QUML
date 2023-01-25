import pandas as pd
import numpy as np
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

class Skin:
    def __init__(self, pathname):
        self.path = pathname
        self.name = "Skin"

    def get_pathname(self):
        return self.path

    def get_dict(self):
        return {
            "1" : "000",
            "2" : "111",
            }
            
    def get_dicinv(self):
        dict = self.get_dict()
        dicinv = {dict[k] : k for k in dict} 
        return dicinv 

    def build_circuit(self, theta, omega):
        qc = qiskit.QuantumCircuit(3)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        qc.rx(np.pi/2, 2)
        qc.rz(omega[0], 0)
        qc.rz(omega[1], 1)
        qc.rz(omega[2], 2)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        qc.rx(np.pi/2, 2)
        #first thetas combination
        qc.cz(0, 2)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 2)
        qc.rz(theta[0], 0)
        qc.rz(theta[1], 2)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 2)
        #second thetas combination
        qc.cz(0, 1)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        qc.rz(theta[2], 0)
        qc.rz(theta[3], 1)
        qc.rx(np.pi/2, 0)
        qc.rx(np.pi/2, 1)
        #third thetas combination
        qc.cz(1, 2)
        qc.rx(np.pi/2, 1)
        qc.rx(np.pi/2, 2)
        qc.rz(theta[4], 1)
        qc.rz(theta[5], 2)
        qc.rx(np.pi/2, 1)
        qc.rx(np.pi/2, 2)
        return qc
    
    def prediction_dict(self, theta, omega):
        qc = qiskit.QuantumCircuit(3, 3)
        qc.append(self.build_circuit(theta, omega), range(3))
        qc.measure(range(3), range(3))
        
        job = qiskit.execute(qc, shots=NSHOTS, backend=qs)
        c = job.result().get_counts()
        rem_keys = ["001", "010","011", "100", "101", "110"]
        for key in rem_keys:
            if key in c:
                c.pop(key)

        if(c == {}):
            c.update({"000": 0, "111" : 0})
        
        return c

    def get_df(self):
        path = self.get_pathname()
        colnames = ['B', 'G', 'R', 'class']
        df = pd.read_csv(path, sep="\t",names=colnames, header=None)
        df['class'] = df['class'].apply(str)
        
        df_random = df.sample(n=1000)
        attributes = df_random.columns[:-1]
        for x in attributes:
            df_random[x] = rescaleFeature(df[x])
        return df_random

    def get_sets(self):
        train_set, test_set = train_test_split(self.get_df(), test_size=TEST_SIZE, train_size=TRAIN_SIZE)
        train_set, test_set = pd.DataFrame(train_set), pd.DataFrame(test_set)
        
        return train_set, test_set

def define_skin():
    #Init Skin Class
    skin = Skin("data/Skin_NonSkin.txt")
    
    #Get useful informations
    df = skin.get_df()
    dict_qbits = skin.get_dict()
    train_set, test_set = skin.get_sets()
    theta_init = np.random.uniform(0, 2*np.pi, 6)
    
    parameters = {
        "df" : df, 
        "dict_qbits" : dict_qbits,
        "train_set" : train_set,
        "test_set" : test_set,
        "theta_init" : theta_init,
        "skin" : skin
        }
    
    #Init Classifier
    classifier_skin = Classifier.Classifier()
    return classifier_skin, parameters

def train_skin():
    classifier_skin, parameters = define_skin()
    print("Training the model...")
    start = time.time()
    
    theta_opti = classifier_skin.train(
        parameters["train_set"], 
        parameters["theta_init"], 
        parameters["dict_qbits"], 
        parameters["df"], 
        parameters["skin"]
    )
    
    end = time.time()
    minutes, seconds = divmod(end-start, 60)
    print("Training duration: {:0>2}min{:05.2f}s".format(int(minutes),seconds))
    return theta_opti

def get_skin_accuracy(theta_opti):
    classifier_skin, parameters = define_skin()
    dicinv = parameters["skin"].get_dicinv()
    classifier_skin.accuracy(parameters["skin"], theta_opti, parameters["test_set"],dicinv)
    return