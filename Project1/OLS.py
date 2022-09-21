import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

from DesignMatrix import DesignMatrix
from functions import  ScaleData, LinReg
import FrankeFunction as FF
import Calculate_MSE_R2 as error

#Complexity of the problem up to 10
maxdegree= 1

# Make data set.
n = 10


x1 = np.random.uniform(0,1,n)
x2 = np.random.uniform(0,1,n)

#x1 = np.linspace(0,1,n)
#x2 = np.linspace(0,1,n)



y = FF.FrankeFunction(x1,x2)#+np.random.normal(0,1,n)
print(y)


x1 = np.array(x1).reshape(n,1)
x2 = np.array(x2).reshape(n,1)

x = np.hstack((x1,x2)).reshape(n,2)

#Design matrix
X = DesignMatrix(x1,x2,maxdegree)

#Split train (80%) and test(20%) data before looping on polynomial degree
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



#Scaling
#x_train, x_test, y_train, y_test = ScaleData(x_train, x_test, y_train, y_test)
y_train_mean = np.mean(y_train)
x_train_mean = np.mean(x_train,axis=0)
x_train = x_train - x_train_mean
y_train = y_train - y_train_mean
x_test = x_test - x_train_mean
y_test = y_test - y_train_mean


#x_train, x_test, y_train, y_test = ScaleData(x_train, x_test, y_train.reshape(1,-1), y_test.reshape(1,-1))

#OLS
y_fit, y_pred, Beta = LinReg(x_train, x_test, y_train, y_test)


print("\nOptimal estimator Beta")
print(Beta.shape)
#print(Beta)

print("\n\nTraining error")
print("MSE =",error.MSE(y_train,y_fit ))
print("R2 =",error.R2(y_train,y_fit))

print("\nTesting error")
print("MSE =",error.MSE(y_test ,y_pred ))
print("R2  =",error.R2(y_test,y_pred))













