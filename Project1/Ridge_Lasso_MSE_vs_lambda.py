import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from functions import RidgeReg, LassoReg
from DesignMatrix import DesignMatrix
import FrankeFunction as FF


def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n


#Model complexity (polynomial degree up to 10)
maxdegree= 5

#For Ridge and Lasso regression, set up the hyper-parameters to investigate
nlambdas = 100
lambdas = np.logspace(-4, 4, nlambdas)

# Make data set.
n = 100
x1 = np.random.uniform(0,1,n)
x2 = np.random.uniform(0,1,n)
y = FF.FrankeFunction(x1,x2)

#Add normally distributed noise
#y = y + np.random.normal(0,0.1,y.shape)

x1 = np.array(x1).reshape(n,1)
x2 = np.array(x2).reshape(n,1)
x = np.hstack((x1,x2)).reshape(n,2)


#Split train (80%) and test(20%) data before looping on polynomial degree
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

X_train = DesignMatrix(x_train[:,0],x_train[:,1],maxdegree)
X_test = DesignMatrix(x_test[:,0],x_test[:,1],maxdegree)


MSEPredict = np.zeros(nlambdas)
MSETrain = np.zeros(nlambdas)
MSELassoPredict = np.zeros(nlambdas)
MSELassoTrain = np.zeros(nlambdas)


for l in range(nlambdas): 
    ytildeRidge, ypredictRidge, BetaRidge = RidgeReg(X_train, X_test, y_train, y_test,lambdas[l])
    ytildeLasso, ypredictLasso = LassoReg(X_train, X_test, y_train, y_test,lambdas[l])

    MSEPredict[l] = MSE(y_test,ypredictRidge)
    MSETrain[l] = MSE(y_train,ytildeRidge)
    MSELassoPredict[l] = MSE(y_test,ypredictLasso)
    MSELassoTrain[l] = MSE(y_train,ytildeLasso)


# Now plot the results
plt.figure()
plt.plot(np.log10(lambdas), MSETrain,'r-' ,label = 'MSE Ridge train')
plt.plot(np.log10(lambdas), MSEPredict, 'r--', label = 'MSE Ridge Test')
plt.plot(np.log10(lambdas), MSELassoTrain,'b-', label = 'MSE Lasso train')
plt.plot(np.log10(lambdas), MSELassoPredict, 'b--', label = 'MSE Lasso Test')

plt.xlabel('log10(lambda)')
plt.ylabel('MSE')
plt.legend()
plt.savefig("Results/Ridge/Ridge_Lasso_MSE_vs_Lambda.png",dpi=150)
plt.show()