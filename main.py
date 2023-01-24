import Iris
import XOR
import Skin
import Quaternary
import numpy as np

#Save theta_opti in a result file. In this way, we can dierctly compute the accuravy with the optimized parameter without execute
# the train method.
def save_results(pathname, theta_opti):
    f = open(pathname, "w")
    for i in range(theta_opti.size-1):
        f.write(str(theta_opti[i])+' ')
    
    f.write(str(theta_opti[theta_opti.size-1]))
    f.close()
    
def get_result(pathname):
    f = open(pathname, "r")
    theta_opti = np.fromstring(f.readline(), dtype=float, sep=' ')
    f.close
    return theta_opti  


def main_iris(trained):
    if trained:
        theta_opti = Iris.train_iris()
        save_results("results/iris_result.txt", theta_opti)
    else:
        theta_opti = get_result("results/iris_result.txt")
    
    Iris.get_iris_accuracy(theta_opti)

def main_XOR(trained):
    if trained:
        theta_opti = XOR.train_XOR()
        save_results("results/XOR_result.txt", theta_opti)
    else:
        theta_opti = get_result("results/XOR_result.txt")
    
    XOR.get_XOR_accuracy(theta_opti)

def main_skin(trained):
    if trained:
        theta_opti = Skin.train_skin()
        save_results("results/Skin_result.txt", theta_opti)
    else:
        theta_opti = get_result("results/Skin_result.txt")
    
    Skin.get_skin_accuracy(theta_opti)

def main_quaternary(trained):
    if trained:
        theta_opti = Quaternary.train_quaternary()
        save_results("results/Quaternary_result.txt", theta_opti)
    else:
        theta_opti = get_result("results/Quaternary_result.txt")
    
    Quaternary.get_quaternary_accuracy(theta_opti)

def main():
    
    
    #main_iris(False)
    #main_XOR(False)
    main_skin(True)
    #main_quaternary(True)
    
    return
   
    
    

if __name__ == "__main__":
    main()