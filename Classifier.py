import numpy as np
import qiskit 
import scipy as sp


NSHOTS = 400
qs = qiskit.Aer.get_backend('qasm_simulator')

class Classifier:
    """Classifier class. This is in this class that we define methods used for training, final prediciton and accuracy
    """
    def __init__(self):
        pass
    
    #Loss function
    def loss_batch(self, theta, batch, dict, df, problem):
        def loss(theta, df, attributes, problem):
            omega = df[attributes].values
            label = dict[df["class"]]
            
            c = problem.prediction_dict(theta, omega)

            if type(label) == list:
                label = label[0]
                
            if label in c:
                e = np.exp(c[label]/NSHOTS)
            else :
                e = 1
            s = np.exp(np.array(list(c.values()))/NSHOTS).sum()
            return -np.log(e/s)
        
        attributes = df.columns[:-1]
        s = batch.apply(
        lambda data : loss(theta, data, attributes, problem),
        axis=1
        )
        #print(s.mean())
        return s.mean()
    
    #Train method
    def train(self, train_set, theta_init, dict, df, problem):
        opt = sp.optimize.minimize(fun = lambda theta : self.loss_batch(theta, train_set, dict, df, problem), x0=theta_init, method='COBYLA', tol=1e-3)
        print("Optimal parameter Theta:", opt.x)
        return opt.x
    

    def prediction(self, theta, omega, problem, IBMQ):
        def argmaxDict(c):
            return max(c, key=c.get)
        
        #If we decide to use online quantic material
        if IBMQ:
            return argmaxDict(problem.prediction_dict_IBMQ(theta, omega))
        
        return argmaxDict(problem.prediction_dict(theta, omega))


    #Accuracy method
    def accuracy(self, problem, theta_opti, test, dicinv, IBMQ):
        print("Compute the accuracy...")
        test["predicted"] = test.apply(lambda row : dicinv[self.prediction(theta_opti, row, problem, IBMQ)], axis=1)
        
        print("Acurracy of the", problem.name, "circuit: ", ((sum(np.array(test["class"] == test["predicted"]))/len(test))*100).round(2), '%')