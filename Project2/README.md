# Repository for delivery of project 2 FYS-STK4155
### Authors: Jose Luis Barreiro Tome - Romain Corseri


The repository is composed of:

## Gradient descent algorithm

### Generic_code_solver/
The fold contains the codes for gradient descents and their various improvements. The naming scheme of the python script reflects the kind of gradient descent it contains. For example:

#### SGD_Adagrad_momentum.py
The codes runs a stochastic gradient descent using Adagrad to tune the learning rate and momentum. The script also contains the results of the optimization using scikit-learn for benchmarking purposes. The code produces the convergence curve (MSE vs epochs) located in the Results/Solver/ folder.

#### Minibatch.py
The code contain the create_mini_batch() function that generate a given number of random data batches of a given size for stochastic gradient descent. 

## Franke Function: Linear regression using gradient descent

The code run a linear and Ridge regression using gradient descent as iterative solver and scan through learning rates, epochs, batch size and regularization as hyperparameters. For example:

#### FF_OLS_hyperparam_eta_batch_size_SGD_RMSprop_momentum.py
#### FF_Ridge_gridsearch_SGD_ADAGRAD_momentum.py 

The scripts produces a grid search plot located in Results/OLS (for linear regression) or Results/Ridge (for Ridge regression)

## Neural Network and activation functions

### Class definition:

This is the central algorithm of project 2 where Neural Networks with flexible number of layers are implemented (including feed-forward pass, back propagation algorithm etc.):

#### NeuralNetwork_regression_leakyrelu.py
Neural Network class for regresssion task with leaky ReLU as activation functions
#### NeuralNetwork_classification_sigmoid_sigmoid.py 
Neural Network class for classification task with sigmoid as activation function for hidden and output layers)


Training the FFNN and solving  a regression problem (Franke Function):

The codes trains the FFNN and run a grid search through learning rate and regularization as hyperparameters. The codes produce a gridsearch plot located in Results/NN/ 

#### FF_NN_gridsearch_leakyReLU.py
The code train the FFNN with leakyReLU as activation function.


Training the FFNN and solving a classification task (Wisconsin Breast Cancer data):

The codes trains the FFNN and run a grid search through learning rate and regularization as hyperparameters. The codes produce a gridsearch plot located in Results/NN/ 

#### BreastCancerData_NN_classification_leakyReLU_tanh.py
The code train the FFNN with leakyReLU as hidden layer activation function and tanh as activation layer of the output layer. The codes priduce a gridsearch plot located in Results/NN


## Logistic regression

Class definition:

#### LogisticReg.py

Training and grid search for Wisconsin Breast Cancer data. The code run through learning rate and regularization as hyperparameters and produce a grid search plot for the accuracy score located in Results/Logistic

#### BreastCancerData_LogisticReg.py

## Report

In the Report/ folder:

#### Project2_FYS_STK4155_Luis_Romain.pdf
The report contains the analysis of the experiments and conclusion on the advantages of deep learning versus linear regression techniques.

 




