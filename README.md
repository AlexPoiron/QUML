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
- logs : Folder that contains logs when we try to solve a problem with inside loss values through iterations, values of the optimized parameter obtained after the training and the accuracy on the test set.
- results : Values of the optimized parameter used when we want to get the accuracy without launch a training session

With these 3 folders, we have **Python files** where we have firstly *Iris/XOR/Skin/Quaternary.py*
