import numpy as np
import pandas as pd
import qiskit
from problem import Problem, rescaleFeature

NSHOTS = 10000
qs = qiskit.Aer.get_backend('qasm_simulator')
#Token used for the IBMQ circuits
TOKEN = "73547946bd0f7f1e1b48368ac35872c76b8bd0100e1e84ea0411076c44208af1127b3b69f345e138c07b03c36809afba05d2e5d9aa1eac3e4d352be42575af06"

class Skin(Problem):
    """Skin class corresponding to the 3rd problem in the paper.

    Args:
        Problem (class): The super class
    """
    def __init__(self):
        super().__init__()
        self.path = "Skin_NonSkin.txt"
        self.name = "Skin"
        self.has_trained = False
        self.theta_init = np.random.uniform(0, 2*np.pi, 6)

    def get_dict(self):
        """Get the dictionnary corresponding to the problem
        
        """
        return {
            "1" : "000",
            "2" : "111",
            }
            
    def build_circuit(self, theta, omega):
        """Build the quantum circuit corresponding to the problem

        Args:
            theta (np.ndarray): the optimized parameter found in the training
            omega (pd.Series): row on the test_set

        Returns:
            Return the qunatum circuit built.
        """
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
        """Get the measurement of our quantum circuit. This measurement gives a count on each possible output possible

        Args:
            theta (np.ndarray): the optimized parameter obtained with the training
            omega (pd.Series): row on the test set

        Returns:
            A qiskit object that is auite similar to a dictionnary with counts on each output qbits.
        """
        qc = qiskit.QuantumCircuit(3, 3)
        qc.append(self.build_circuit(theta, omega), range(3))
        qc.measure(range(3), range(3))
        
        job = qiskit.execute(qc, shots=NSHOTS, backend=qs)
        res = job.result().get_counts()
        rem_keys = ["001", "010","011", "100", "101", "110"]
        for key in rem_keys:
            if key in res:
                res.pop(key)

        if(res == {}):
            res.update({"000": 0, "111" : 0})
        #res ={'000':0,'111':0}
        #for key in c:
        #    res[key] = c[key]
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

        qc = qiskit.QuantumCircuit(3, 3)
        qc.append(self.build_circuit(theta, omega), range(3))
        qc.measure(range(3), range(3))

        mapped_circuit = qiskit.transpile(qc, backend=backend)
        qobj = qiskit.assemble(mapped_circuit, backend=backend, shots=NSHOTS)

        job = backend.run(qobj)
        print(job.status())
        res = job.result().get_counts()
        rem_keys = ["001", "010","011", "100", "101", "110"]
        for key in rem_keys:
            if key in res:
                res.pop(key)

        if(res == {}):
            res.update({"000": 0, "111" : 0})

        return res

    def get_df(self):
        """Create a Pandas Dataframe

        Returns:
            the Dataframe
        """
        path = "data/" + self.path
        colnames = ['B', 'G', 'R', 'class']
        df = pd.read_csv(path, sep="\t",names=colnames, header=None)
        df['class'] = df['class'].apply(str)
        
        df_random = df.sample(n=1000)
        attributes = df_random.columns[:-1]
        for x in attributes:
            df_random[x] = rescaleFeature(df[x])
        return df_random