import Iris
import XOR
import Skin
import Quaternary

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


def get_correct_problem(problem_name: str) -> object:
    """Get the correct problem object following the name given

    Args:
        problem_name (str): The name of the problem 

    Returns:
        The correct problem object created
    """
    problem = None
    
    if problem_name == "Iris":
        problem = Iris.Iris()
    
    elif problem_name == "XOR":
        problem = XOR.XOR()
    
    elif problem_name == "Skin":
        problem = Skin.Skin()
    
    else:
        problem = Quaternary.Quaternary()
    
    return problem

def execute_problem(problem: object, problem_name: str,  trained: bool, IBQM: bool) -> None:
    """Execute one of the 4 problems. Possibility to train the model or use an online quantic material with
       IBMQ. This function return always the accuracy on the test set.

    Args:
        problem (object): The problem we want to execute. 
        id_problem (int): Corresponding value to the problem in the dictionnary
        trained (bool): Boolean value set to True if we want to train the model before gets the accuracy.
        IBQM (bool): Boolean value set to True if we want to use online quantic material.
    """
    problem = get_correct_problem(problem_name)
    path = "results/" + problem_name + "_result.txt"
    
    if trained:
        theta_opti = problem.launch_train(problem, problem_name)
        save_results(path, theta_opti)
    else:
        theta_opti = get_result(path)
    
    problem.get_accuracy(problem, problem_name, theta_opti, IBQM)
        
        
# Reference list
PROBLEMS = ["Iris", "XOR", "Skin","Quaternary"]

def main():
    
    problem = get_correct_problem("Quaternary")
    execute_problem(problem, "Quatenary", trained=True, IBQM=False)
    return
   
if __name__ == "__main__":
    main()