import Classifier
import Iris
import numpy as np
import time

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

def main():
    
    #Init Iris Class
    iris = Iris.Iris("data/iris_csv.csv")
    
    #Get useful informations
    df = iris.get_df()
    dict_qbits = iris.get_dict()
    train_set, test_set = iris.get_sets()
    theta_init = np.random.uniform(0, 2*np.pi, 8)
    
    #Init Classifier
    classifier_iris = Classifier.Classifier()
    #Train
    print("Training the model...")
    start = time.time()
    
    #theta_opti = classifier_iris.train(train_set, theta_init, dict_qbits, df, iris)
    #save_results("results/iris_result.txt", theta_opti)
    
    #end = time.time()
    #minutes, seconds = divmod(end-start, 60)
    #print("Training duration: {:0>2}min{:05.2f}s".format(int(minutes),seconds))
    
    
    #We recup in the result file the optimized parameter
    
    #Accuracy
    theta_opti = get_result("results/iris_result.txt")
    classifier_iris.accuracy(iris, theta_opti, test_set, dict_qbits)
    return

if __name__ == "__main__":
    main()