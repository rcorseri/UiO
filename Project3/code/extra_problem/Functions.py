#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

def FrankeFunction(x,y):
    '''#Definition of the Franke Function'''
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)

def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def ScaleData(x_train, x_test, y_train, y_test):
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    scaler.fit(y_test)
    y_test_scaled = scaler.transform(y_test)
    scaler.fit(y_train)
    y_train_scaled = scaler.transform(y_train)
    
    return x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled

def DesignMatrix(x, y, n ):
    '''This function returns the design matrix of a bi-variate polynomial function'''
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2)		# Number of elements in beta
    X = np.ones((N,l))

    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)

    return X
        
def LinReg(X_train, X_test, y_train):
    OLSbeta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
    ytildeTrain = X_train @ OLSbeta
    ytildeTest = X_test @ OLSbeta
    return ytildeTrain, ytildeTest, OLSbeta

def RidgeReg(X_train, X_test, y_train, y_test,lmb):
    Ridgebeta = np.linalg.pinv(X_train.T @ X_train + lmb*np.identity(X_train.shape[1])) @ X_train.T @ y_train
    ytildeTrain = X_train @ Ridgebeta
    ytildeTest = X_test @ Ridgebeta
    return ytildeTrain, ytildeTest, Ridgebeta

def LassoReg(X_train, X_test, y_train, y_test,lmb):
    modelLasso = Lasso(lmb,fit_intercept=False)
    modelLasso.fit(X_train,y_train)
    ytildeTrain = modelLasso.predict(X_train)
    ytildeTest = modelLasso.predict(X_test)
    return ytildeTrain, ytildeTest

def Beta_std(var,X_train,Beta,p):
    Beta_var = var*np.linalg.pinv(X_train.T @ X_train)
    err = []
    for p_ in range(p):
        err = np.append(err,Beta_var[p_,p_] ** 0.5)
    return err

