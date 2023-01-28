import Iris
import XOR
import Skin
import Quaternary
from utils import save_results, get_result


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

def execute_problem(problem: object, trained: bool, IBQM: bool) -> None:
    """Execute one of the 4 problems. Possibility to train the model or use an online quantic material with
       IBMQ. This function return always the accuracy on the test set.

    Args:
        problem (object): The problem we want to execute. 
        id_problem (int): Corresponding value to the problem in the dictionnary
        trained (bool): Boolean value set to True if we want to train the model before gets the accuracy.
        IBQM (bool): Boolean value set to True if we want to use online quantic material.
    """
    problem = get_correct_problem(problem.name)
    path = "results/" + problem.name + "_result.txt"
    
    if trained:
        theta_opti = problem.launch_train(problem)
        save_results(path, theta_opti)
        problem.get_accuracy(problem, theta_opti, IBQM)

    else:
        theta_opti = get_result(path)
        problem.get_accuracy(problem, theta_opti, IBQM)

            
        
# Reference list
PROBLEMS = ["Iris", "XOR", "Skin","Quaternary"]

def main():
    
    problem = get_correct_problem("XOR")
    execute_problem(problem, trained=False, IBQM=False)
    return
   
if __name__ == "__main__":
    main()