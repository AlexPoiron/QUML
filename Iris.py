import numpy as np
import pandas as pd
import qiskit
import collections
from problem import Problem, rescaleFeature

#Values used to execute the quantum circuit
NSHOTS = 1500
qs = qiskit.Aer.get_backend('qasm_simulator')

#Token used for the IBMQ circuits
TOKEN = "73547946bd0f7f1e1b48368ac35872c76b8bd0100e1e84ea0411076c44208af1127b3b69f345e138c07b03c36809afba05d2e5d9aa1eac3e4d352be42575af06"


class Iris(Problem):
    """Iris class corresponding to the 1st problem in the paper.

    Args:
        Problem (class): The super class
    """
    def __init__(self):
        super().__init__()
        self.path_data = "data/iris_csv.csv"
        self.name = "Iris"
        self.has_trained = False
        self.theta_init = np.random.uniform(0, 2*np.pi, 8)
  
    def get_dict(self):
        """Get the dictionnary corresponding to the problem
        
        """
        return {
        "Iris-setosa" : "00",
        "Iris-versicolor" : "01",
        "Iris-virginica" : "10",
        }   
    
    def build_circuit(self, theta: np.ndarray, omega: pd.Series) -> qiskit.QuantumCircuit:
        """Build the quantum circuit corresponding to the problem

        Args:
            theta (np.ndarray): the optimized parameter found in the training
            omega (pd.Series): row on the test_set

        Returns:
            Return the qunatum circuit built.
        """
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
    
    def prediction_dict(self, theta: np.ndarray, omega: pd.Series) -> qiskit.result.counts.Counts:
        """Get the measurement of our quantum circuit. This measurement gives a count on each possible output possible

        Args:
            theta (np.ndarray): the optimized parameter obtained with the training
            omega (pd.Series): row on the test set

        Returns:
            A qiskit object that is auite similar to a dictionnary with counts on each output qbits.
        """
        qc = self.build_circuit(theta, omega)
        qc.measure(range(2), range(2))
        job = qiskit.execute(qc, shots=NSHOTS, backend=qs)
        res = {'00':0, '01':0,'10':0}
        c = job.result().get_counts()
        for key in c:
            res[key] = c[key]
        res.pop('11', None)
        return res
    
    def prediction_dict_IBMQ(self, theta: np.ndarray, omega: pd.Series) -> qiskit.result.counts.Counts:
        """Get the measurement of our quantum circuit. This measurement gives a count on each possible output possible. This, time
           we use online quantum material.

        Args:
            theta (np.ndarray): the optimized parameter obtained with the training
            omega (pd.Series): row on the test set

        Returns:
            A qiskit object that is auite similar to a dictionnary with counts on each output qbits.
        """
        qiskit.IBMQ.save_account(TOKEN, overwrite=True) 
        provider = qiskit.IBMQ.load_account()
        backend = qiskit.providers.ibmq.least_busy(provider.backends())

        qc = self.build_circuit(theta, omega)
        qc.measure(range(2), range(2))

        mapped_circuit = qiskit.transpile(qc, backend=backend)
        qobj = qiskit.assemble(mapped_circuit, backend=backend, shots=NSHOTS)

        job = backend.run(qobj)
        print(job.status())
        res = job.result().get_counts()
        res.pop("11", None)

        return res
    
    def get_df(self):
        """Create a Pandas Dataframe

        Returns:
            the Dataframe
        """
        path = self.path_data
        df = pd.read_csv(path)
        attributes = df.columns[:-1]
        for x in attributes:
            df[x] = rescaleFeature(df[x])
        return df   