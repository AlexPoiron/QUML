# QUML Project
We have in this repository our project for the QUML course. In this README, we will explain our architecture and also how to launch and test the project. Through this project, we had to build 4 models that solve four problems a research paper called **Polyadic Quantum Classifier**. In this paper these problems are : 
- Iris :resolve a ternary classification  
- XOR : double classification
- Skin : ternary classification
- Synthetic : quaternary classfication

We decided to implement this project through different **Python script**, instead of a **Python Notebook**. 

## Architecture
At the root we have **3 folders** :
- data : Datasets used for the different problems
- logs : Folder that contains logs when we try to solve a problem with inside loss values through iterations, values of the optimized parameter obtained after the training and the accuracy on the test set. We also have for each problem a file where its quantum circuit is drawed.
- results : Values of the optimized parameter used when we want to get the accuracy without launch a training session

With these 3 folders, we have **Python files** where we have:
- **Iris/XOR/Skin/Quaternary** : that represant our 4 problems. 
- **problem** which is the super class over the four just cited.
- **Classifier** which is where we compute the loss and train our models
- **utils**: file with useful functions, including these to create logs for example

## How to use the project and run an example
To launch the projet, you can follow the example commented  in the main function or follow these steps:
- create a problem with the function **get_correct_problem()** with the corresponding name : Iris, XOR, Skin or Quaternary
- use the function called **execute_problem()** to execute the problem. You give the parameter problem that you just created, a boolean if you want to do a training and another boolean if you want to use quantum material online with IBMQ.
