import numpy as np
import qiskit 
import pandas as pd
import scipy as sp
import Iris

NSHOTS = 400
qs = qiskit.Aer.get_backend('qasm_simulator')

class Classifier:
    def __init__(self):
        pass
    
    #Getters
    def get_dicinv(self, dict):
        dicinv = {dict[k] : k for k in dict} 
        return dicinv
    
    
    #Loss function
    def loss_batch(self, theta, batch, dict, df, problem):
        def loss(theta, df, attributes, problem):
            omega = df[attributes].values
            print(df)
            label = dict[df["class"]]
            
            c = self.predictionDict(theta, omega, problem)
            c.pop('11', None)
            
            if label in c:
                e = np.exp(c[label]/NSHOTS)
            else :
                e = 1
            s = np.exp(np.array(list(c.values()))/NSHOTS).sum()
            #print("c =", c, "|| s =", s)
            return -np.log(e/s)
        
        attributes = df.columns[:-1]
        s = batch.apply(
        lambda data : loss(theta, data, attributes, problem),
        axis=1
        )
        return s.mean()
    
    #Train method
    def train(self, train_set, theta_init, dict, df, problem):
        opt = sp.optimize.minimize(fun = lambda theta : self.loss_batch(theta, train_set, dict, df, problem), x0=theta_init, method='COBYLA', tol=1e-3)
        return opt.x
    
    #Prediction methods
    def predictionDict(self, theta, omega, problem):
        qc = problem.build_circuit(theta, omega)
        #qc.append(problem.build_circuit(theta, omega), range(2))
        qc.measure(range(2), range(2))
        
        job = qiskit.execute(qc, shots=NSHOTS, backend=qs)
        c = job.result().get_counts()
        c.pop('11', None)
        #print("c =", c)
        return c

    def prediction(self, theta, omega, problem):
        def argmaxDict(c):
            v = None
            for key in c :
                if v is None or c[key] >= v:
                    k = key
                    v = c[key]
            return k
        
        return argmaxDict(self.predictionDict(theta, omega, problem))

    #def predictedClass(self, data):
        omega = data[attributes].values.T
        theta_opti = self.train()
        
        ypred = self.prediction(theta_opti, omega)
        dicinv = self.get_dicinv()
        
        return dicinv[ypred]
    
    #Accuracy method
    def accuracy(self, problem, theta_opti, test, dict):
        dicinv = self.get_dicinv(dict)
        #print(dicinv)
        #print(self.prediction(theta_opti, test.iloc[0], problem))
        test["predicted"] = test.apply(lambda row : dicinv[self.prediction(theta_opti, row, problem)], axis=1)
        
        print("Acurracy of Iris Circuit: ", ((sum(np.array(test["class"] == test["predicted"]))/len(test))*100).round(2), '%')
        #print(test)