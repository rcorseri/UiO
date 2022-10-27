#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 12:02:03 2022

@author: rpcorser
"""
import numpy as np

def create_mini_batches(X, y, batch_size):
    mini_batches = []
    data = np.hstack((X, y))
    np.random.shuffle(data)
    n_minibatches = data.shape[0] // batch_size
    i = 0
 
    for i in range(n_minibatches + 1):
        mini_batch = data[i * batch_size:(i + 1)*batch_size, :]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))
    if data.shape[0] % batch_size != 0:
        mini_batch = data[i * batch_size:data.shape[0]]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))
    return mini_batches


def relu(X):
    return X * (X > 0)

def relu_grad(X):
    return 1. * (X > 0)
    

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)

def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

class NeuralNetwork:
    def __init__(
            self,
            X_data,
            Y_data,
            n_hidden_neurons=[50],
            n_categories=10,
            epochs=10,
            batch_size=100,
            eta=0.1,
            lmbd=0.0):

        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories = n_categories

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd

        self.create_biases_and_weights()

    def create_biases_and_weights(self):
        
        self.hidden_weights = []
        self.hidden_bias = []
        self.output_weights = []
        self.output_bias = []
        self.z_h = []

        self.hidden_weights.append(np.random.randn(self.n_features,self.n_hidden_neurons[0]))
        self.hidden_bias.append(np.zeros(self.n_hidden_neurons[0])+0.01)
        self.z_h.append(np.zeros(self.n_hidden_neurons[0]))
        j=1
        for i in range(len(self.n_hidden_neurons)-1):
            self.hidden_weights.append(np.random.randn(self.n_hidden_neurons[i],self.n_hidden_neurons[j]))    
            self.hidden_bias.append(np.zeros(self.n_hidden_neurons[j]) + 0.01)
            self.z_h.append(np.zeros(self.n_hidden_neurons[j]))
            j+=1
    
            if j > len(self.n_hidden_neurons):
                break
    
        self.output_weights.append(np.random.randn(self.n_hidden_neurons[-1],self.n_categories))
        self.output_bias.append(np.zeros(self.n_categories)+0.01)
    
    def feed_forward_out(self,X):
        # feed-forward for training
        self.a_h=[0 for _ in range(len(self.n_hidden_neurons))] 
        self.X_curr = X

        for i in range(len(self.n_hidden_neurons)):
            self.X_prev = self.X_curr
            self.z_h[i] = np.matmul(self.X_prev, self.hidden_weights[i]) + self.hidden_bias[i]
            self.a_h[i] = relu(self.z_h[i]) 
            self.X_curr = self.a_h[i]
        
        self.z_o = np.matmul(self.a_h[-1], self.output_weights[0]) + self.output_bias[0]
        #self.a_o = sigmoid(self.z_o)
        self.a_o = self.z_o #For regression problem, the output activation function is the identity function
        return self.a_o

    def feed_forward(self):
        # feed-forward for output
        self.a_h=[0 for _ in range(len(self.n_hidden_neurons))]
        self.X_curr = self.X_data

        for i in range(len(self.n_hidden_neurons)):
          
            self.X_prev = self.X_curr
            self.z_h[i] = np.matmul(self.X_prev, self.hidden_weights[i]) + self.hidden_bias[i]
            self.a_h[i] = relu(self.z_h[i]) 
            self.X_curr = self.a_h[i]
        
        self.z_o = np.matmul(self.a_h[-1], self.output_weights[0]) + self.output_bias[0]
        #a_o = sigmoid(z_o)
        self.a_o = self.z_o 
              

    def backpropagation(self): 
        
            #initialize
            self.hidden_weights_gradient = [0 for _ in range(len(self.n_hidden_neurons))]
            self.hidden_bias_gradient = [0 for _ in range(len(self.n_hidden_neurons))]
            self.output_weights_gradient = []
            self.output_bias_gradient = []
            
            #update output layer first
            self.error_output = self.a_o - self.Y_data
            self.output_weights_gradient = np.matmul(self.a_h[-1].T, self.error_output)
            if self.lmbd > 0.0: 
                self.output_weights_gradient += self.lmbd * self.output_weights[0]
            
            self.output_bias_gradient = np.sum(self.error_output, axis=0) 
            self.output_weights[0] -= self.eta * self.output_weights_gradient[0] 
            self.output_bias[0] -= self.eta * self.output_bias_gradient[0]  
            
            for k in reversed(range(len(self.n_hidden_neurons))):
                
                #if the NN contains only one hidden layer
                if len(self.n_hidden_neurons)==1:
                    self.error_hidden = np.matmul(self.error_output, self.output_weights[0].T) * relu_grad(self.z_h[0]) 
                    self.hidden_weights_gradient[0] = np.matmul(self.X_data.T, self.error_hidden)
                    ####
                    if self.lmbd > 0.0: 
                        self.hidden_weights_gradient[0] += self.lmbd * self.hidden_weights[0]
                    ####
                    self.hidden_bias_gradient[0] = np.sum(self.error_hidden, axis=0)
                    self.hidden_weights[0] -= self.eta * self.hidden_weights_gradient[0]
                    self.hidden_bias[0] -= self.eta * self.hidden_bias_gradient[0]
                    break
                
                #if L>1
                if k==len(self.n_hidden_neurons)-1:        
                    self.error_hidden = np.matmul(self.error_output, self.output_weights[-1].T) * relu_grad(self.z_h[k]) #OBS! derivative of the sigmoid function
                    self.hidden_weights_gradient[k] = np.matmul(self.a_h[k-1].T,self.error_hidden)
                          
                if k!=0 and k!= len(self.n_hidden_neurons)-1:       
                    self.error_hidden = np.matmul(self.error_output, self.hidden_weights[k+1].T) * relu_grad(self.z_h[k])
                    self.hidden_weights_gradient[k] = np.matmul(self.a_h[k-1].T,self.error_hidden)
                     
                if k==0:     
                    self.error_hidden = np.matmul(self.error_output, self.hidden_weights[k+1].T) * relu_grad(self.z_h[k])
                    self.hidden_weights_gradient[k] = np.matmul(self.X_data.T, self.error_hidden)
                ####
                if self.lmbd > 0.0: 
                    self.hidden_weights_gradient[k] += self.lmbd * self.hidden_weights[k] 
                ##### 
                   
                self.hidden_bias_gradient[k] = np.sum(self.error_hidden, axis=0)
                self.hidden_weights[k] -= self.eta * self.hidden_weights_gradient[k]
                self.hidden_bias[k] -= self.eta * self.hidden_bias_gradient[k]
                self.error_output = self.error_hidden
                         


    def predict(self,X):
        output = self.feed_forward_out(X)
        return output
    

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities
    
    

    def train(self):
        data_indices = np.arange(self.n_inputs)
        self.err = []
        self.score = []
        for i in range(self.epochs):      
            for j in range(self.iterations):
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()
            

            
                  
        

