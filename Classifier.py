import numpy as np
import qiskit 
import scipy as sp
import swifter
from utils import create_logs

NSHOTS = 1500
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
            #print("c = ", c)
            if type(label) == list:
                label = label[0]
                
            if label in c:
                e = np.exp(c[label]/NSHOTS)
            else:
                e = 1
            #print("e = ", e)
            
            s = np.exp(np.array(list(c.values()))/NSHOTS).sum()
            #print("s =", s)
            return -np.log(e/s)
        
        attributes = df.columns[:-1]
        s = batch.swifter.apply(
        lambda data : loss(theta, data, attributes, problem),
        axis=1
        )
        #print("mean s: ", s.mean())
        print("Loss value:", s.mean())
        
        #Save loss value in the logs
        log = "Loss value: " + str(s.mean())
        create_logs(problem.name, True, [log])

        return s.mean()
    
    #Train method
    def train(self, train_set, theta_init, dict, df, problem):
        opt = sp.optimize.minimize(fun = lambda theta : self.loss_batch(theta, train_set, dict, df, problem), x0=theta_init, method='COBYLA', tol=1e-4)
        print("-"*20)
        print("Optimal parameter Theta:", opt.x)
        
        #Save optimal parameter in the logs
        logs = ["-"*20, "Optimal parameter Theta: " + str(opt.x)]
        create_logs(problem.name, True, logs)
        
        problem.has_trained = True
        
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