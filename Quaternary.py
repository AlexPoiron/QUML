import numpy as np
import pandas as pd
import qiskit
from problem import Problem
from sklearn.datasets import make_classification

NSHOTS = 400
qs = qiskit.Aer.get_backend('qasm_simulator')

class Quaternary(Problem):
    """Quaternary class corresponding to the 4th problem in the paper.

    Args:
        Problem (class): The super class
    """
    def __init__(self):
        super().__init__()
        self.name = "Quaternary"
        self.theta_init = np.random.uniform(0, 2*np.pi, 12)

    def get_dict(self):
        """Get the dictionnary corresponding to the problem
        
        """
        return {
            "0" : "00",
            "1" : "01",
            "2" : "10",
            "3" : "11"
            }
       
    def build_circuit(self, theta, omega):
        """Build the quantum circuit corresponding to the problem

        Args:
            theta (np.ndarray): the optimized parameter found in the training
            omega (pd.Series): row on the test_set

        Returns:
            Return the qunatum circuit built.
        """
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
        """Get the measurement of our quantum circuit. This measurement gives a count on each possible output possible

        Args:
            theta (np.ndarray): the optimized parameter obtained with the training
            omega (pd.Series): row on the test set

        Returns:
            A qiskit object that is auite similar to a dictionnary with counts on each output qbits.
        """
        qc = qiskit.QuantumCircuit(2, 2)
        qc.append(self.build_circuit(theta, omega), range(2))
        qc.measure(range(2), range(2))
        
        job = qiskit.execute(qc, shots=NSHOTS, backend=qs)
        c = job.result().get_counts()       
        return c

    def get_df(self):
        """Create a Pandas Dataframe

        Returns:
            the Dataframe
        """
        X,y = make_classification(n_samples=5000, n_features=4)
        df = pd.DataFrame(X)
        df['class'] = y
        df['class'] = df['class'].astype(str)
        return df