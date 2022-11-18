#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 14:42:21 2022

@author: rpcorser
"""
import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.colors import LogNorm
import seaborn as sns
from sklearn.model_selection import train_test_split as splitter
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
import pickle
import os 
from Functions import Beta_std, FrankeFunction, R2, MSE, DesignMatrix, LinReg
from NeuralNetwork_classification_leakyrelu_sigmoid import NeuralNetwork, leakyrelu, leakyrelu_grad, sigmoid,  accuracy_score_numpy
from sklearn.neural_network import MLPRegressor, MLPClassifier


np.random.seed(0)        #create same seed for random number every time

cancer=load_breast_cancer()      #Download breast cancer dataset

inputs=cancer.data                     #Feature matrix of 569 rows (samples) and 30 columns (parameters)
outputs=cancer.target                  #Label array of 569 rows (0 for benign and 1 for malignant)
labels=cancer.feature_names[0:30]

print('The content of the breast cancer dataset is:')      #Print information about the datasets
print(labels)
print('-------------------------')
print("inputs =  " + str(inputs.shape))
print("outputs =  " + str(outputs.shape))
print("labels =  "+ str(labels.shape))

x=inputs      #Reassign the Feature and Label matrices to other variables
y=outputs



# Visualisation of dataset (for correlation analysis)

plt.figure()
plt.scatter(x[:,0],x[:,2],s=40,c=y,cmap=plt.cm.Spectral)
plt.xlabel('Mean radius',fontweight='bold')
plt.ylabel('Mean perimeter',fontweight='bold')
plt.show()

plt.figure()
plt.scatter(x[:,5],x[:,6],s=40,c=y, cmap=plt.cm.Spectral)
plt.xlabel('Mean compactness',fontweight='bold')
plt.ylabel('Mean concavity',fontweight='bold')
plt.show()


plt.figure()
plt.scatter(x[:,0],x[:,1],s=40,c=y,cmap=plt.cm.Spectral)
plt.xlabel('Mean radius',fontweight='bold')
plt.ylabel('Mean texture',fontweight='bold')
plt.show()

plt.figure()
plt.scatter(x[:,2],x[:,1],s=40,c=y,cmap=plt.cm.Spectral)
plt.xlabel('Mean perimeter',fontweight='bold')
plt.ylabel('Mean compactness',fontweight='bold')
plt.show()


# Generate training and testing datasets

#Select features relevant to classification (texture,perimeter,compactness and symmetery) 
#and add to input matrix

#Reshape and scale input
temp1=np.reshape((x[:,1]-np.mean(x[:,1]))/np.std(x[:,1]),(len(x[:,1]),1))
temp2=np.reshape((x[:,2]-np.mean(x[:,2]))/np.std(x[:,2]),(len(x[:,2]),1))

X=np.hstack((temp1,temp2))      
temp=np.reshape((x[:,5]-np.mean(x[:,5]))/np.std(x[:,5]),(len(x[:,5]),1))

X=np.hstack((X,temp))       
temp=np.reshape((x[:,8]-np.mean(x[:,8]))/np.std(x[:,8]),(len(x[:,8]),1))
X=np.hstack((X,temp))       

x_train,x_test,z_train,z_test=splitter(X,y,test_size=0.8)   #Split datasets into training and testing


del temp1,temp2,temp
print(x_train.shape[0])
M = 10   #size of each minibatch
m = int(z_train.shape[0]/M) #number of minibatches
epochs = 1000 #number of epochs 

etas = np.logspace(-4, -1, 4)
lambdas = np.logspace(-5,2, 8)
n_hidden_neurons = [4,4,4]
n_categories = 1
n_features = x_train.shape[1]
n_inputs = x_train.shape[0]

#Initialize error to store
NN_err_train=np.zeros(shape=(lambdas.shape[0],etas.shape[0]))
NN_err_test=np.zeros(shape=(lambdas.shape[0],etas.shape[0]))
DNN_scikit_train = np.zeros(shape=(lambdas.shape[0],etas.shape[0]))
DNN_scikit_test = np.zeros(shape=(lambdas.shape[0],etas.shape[0]))
#
z_train = np.reshape(z_train, (z_train.shape[0],1))
z_train_ravel = np.ravel(z_train)
z_test_ravel = np.ravel(z_test)


#Looping through regularization and learning rates
i=0
for lmbd in lambdas:
    j=0  
    for eta in etas: 
        
        #####own NN implementation######
        dnn = NeuralNetwork(x_train, z_train, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=M, n_hidden_neurons=n_hidden_neurons, n_categories=n_categories)
        dnn.train()
        
        z_fit = dnn.predict2(x_train)
        z_pred = dnn.predict2(x_test)
        
        
        NN_err_train[i,j] = accuracy_score(z_train,z_fit)
        NN_err_test[i,j] = accuracy_score(z_test,z_pred)
       
        #####scikit NN implementation#####
        dnn2 = MLPClassifier(hidden_layer_sizes=n_hidden_neurons, activation='relu', solver ='lbfgs',
                            alpha=lmbd, batch_size = M, learning_rate_init=eta, max_iter=epochs)
        dnn2.fit(x_train, z_train_ravel)
        z_fit2 = dnn2.predict(x_train)
        z_pred2 = dnn2.predict(x_test)
        
        DNN_scikit_train[i,j] = accuracy_score_numpy(z_train_ravel,z_fit2)
        DNN_scikit_test[i,j] = accuracy_score_numpy(z_test_ravel,z_pred2)
                
        j+=1
    i+=1
    


#Heatmaps for gridsearch on lambda and ridge
x_axis = etas # labels for x-axis
y_axis = lambdas # labels for y-axis 



heat1 = sns.heatmap(NN_err_train,vmin=0.009,vmax=1,annot=True, xticklabels=x_axis, yticklabels=y_axis, cmap="viridis",linewidths =0.5)
heat1.set(xlabel='learning rate', ylabel ='regularization', title = f"Accuracy score training set (Act:leakyReLU-sigmoid)")
plt.savefig(f"Results/NN/BreastCancer_ReLUsig_accuracy_score_train.png", dpi=150)
plt.show()


heat2 = sns.heatmap(NN_err_test,vmin=0.009,vmax=1,annot=True, xticklabels=x_axis, yticklabels=y_axis, cmap="viridis",linewidths =0.5)
heat2.set(xlabel='learning rate', ylabel ='regularization', title = f"Accuracy score test set (Act:leakyReLU-sigmoid)")
plt.savefig(f"Results/NN/BreastCancer_ReLUsig_accuracy_score_test.png", dpi=150)
plt.show()


heat3 = sns.heatmap(DNN_scikit_train,vmin=0.009,vmax=1,annot=True, xticklabels=x_axis, yticklabels=y_axis, cmap="viridis",linewidths =0.5) 
heat3.set(xlabel='learning rate', ylabel ='regularization', title = f"Accuracy score training set Scikit (Act:ReLU)")
plt.savefig(f"Results/NN/BreastCancer_ReLU_accuracy_score_train_scikit.png", dpi=150)
plt.show()


heat4 = sns.heatmap(DNN_scikit_test,vmin=0.009,vmax=1,annot=True, xticklabels=x_axis, yticklabels=y_axis, cmap="viridis",linewidths =0.5)
heat4.set(xlabel='learning rate', ylabel ='regularization', title = f"Accuracy score training set Scikit (Act:ReLU) ")
plt.savefig(f"Results/NN/BreastCancer_ReLU_accuracy_score_test_scikit.png", dpi=150)
plt.show()
