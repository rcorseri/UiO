#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 12:02:03 2022

@author: rpcorser
"""
import numpy as np




def sigmoid(X):
    if X.all()>=0:
        z = np.exp(-X)
        return 1. / (1. +z)
    else:
        z = np.exp(X)
        return z / (1. + z)


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
        self.output_weights = []
        self.output_bias = []
        self.output_weights.append(np.random.randn(self.n_features,self.n_categories))
        self.output_bias.append(np.zeros(self.n_categories)+0.01)
    
    def feed_forward_out(self,X):        
        self.X_curr = X
        self.z_o = np.matmul(X, self.output_weights[0]) + self.output_bias[0]   
        self.a_o = sigmoid(self.z_o)   
        return self.a_o

    def feed_forward(self):     
        self.X_curr = self.X_data
        self.z_o = np.matmul(self.X_data, self.output_weights[0]) + self.output_bias[0]
        self.a_o = sigmoid(self.z_o)
          
    def backpropagation(self): 
            #initialize
            self.output_weights_gradient = []
            self.output_bias_gradient = []   
            #update output layer first
            self.error_output = self.a_o - self.Y_data
            self.output_weights_gradient = np.matmul(self.X_data.T, self.error_output)
            if self.lmbd > 0.0: 
                self.output_weights_gradient += self.lmbd * self.output_weights[0]     
            self.output_bias_gradient = np.sum(self.error_output, axis=0) 
            self.output_weights[0] = self.output_weights[0] - self.eta * self.output_weights_gradient[0] 
            self.output_bias[0] -= self.eta * self.output_bias_gradient[0]  
 
    def predict(self, X):
        probabilities = self.feed_forward_out(X)        
        return np.argmax(probabilities, axis=1)
    
    def predict2(self, X):
        probabilities = self.feed_forward_out(X)        
        return np.where(probabilities>0.5, 1,0)

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
            

            
                  
        

