import numpy as np

def save_results(pathname: str, theta_opti: np.ndarray) -> None:
    """Save theta_opti in a result file. In this way, we can dierctly compute the accuracy with the 
    optimized parameter without execute the train method.

    Args:
        pathname (str): the pathname where we want to save our optimized parameter
        theta_opti (np.ndarray): the optimized parameter
    """
    f = open(pathname, "w")
    for i in range(theta_opti.size-1):
        f.write(str(theta_opti[i])+' ')
    
    f.write(str(theta_opti[theta_opti.size-1]))
    f.close()
    
def get_result(pathname: str) -> np.ndarray:
    """Get the optimized parameter from the corresponding problem file

    Args:
        pathname (str): The file pathname

    Returns:
        the optimized parameter called theta_opti
    """
    f = open(pathname, "r")
    theta_opti = np.fromstring(f.readline(), dtype=float, sep=' ')
    f.close
    return theta_opti


def create_logs(problem_name: str, trained: bool, logs: list) -> None:
    """Write in the corresponding file in the logs folder the results obtained

    Args:
        problem_name (str): name of the problem used to redirect the logs in the correct file
        trained (bool): boolean value set to True if we want to train the model
        logs (list): list of string containing all the logs
    """
    if trained:
        pathname = "logs/" + problem_name + "_accuracy_train.txt"
        f = open(pathname, "a")
        
    else:
        pathname = "logs/" + problem_name + "_accuracy.txt"
        f = open(pathname, "a")
        
    for log in logs:
        f.write(log + "\n")
    
    f.close
    