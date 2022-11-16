

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
from Functions import Beta_std, FrankeFunction, R2, MSE, DesignMatrix, LinReg
from sklearn.metrics import mean_squared_error, r2_score
from Minibatch import create_mini_batches
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
#from tf.keras.optimizers import Adagrad


#Create data
np.random.seed(2003)
n = 100
maxdegree = 2

x = np.random.uniform(0,1,n)
y = np.random.uniform(0,1,n)
#z = FrankeFunction(x, y)
z = 1 + x + y + x*y + x**2 + y**2

# Add random distributed noise
#var = 0.1
#z = z + np.random.normal(0,var,z.shape)


x = np.array(x).reshape(n,1)
y = np.array(y).reshape(n,1)

x1 = np.hstack((x,y)).reshape(n,2)
z = np.reshape(z,(z.shape[0],1))
X = DesignMatrix(x1[:,0],x1[:,1],maxdegree)

##


#OLS With gradient descent
M = 10   #size of each minibatch
m = int(z.shape[0]/M) #number of minibatches
n_epochs = 1000 #number of epochs


beta = np.random.randn(X.shape[1],1)
eta = 0.01
delta = 10**-8
j = 0
err=[]

for epoch in range(1,n_epochs+1):
    mini_batches = create_mini_batches(X,z,M) 
    Giter = np.zeros(shape=(X.shape[1],X.shape[1]))
    for minibatch in mini_batches:
        X_mini, z_mini = minibatch
        gradients = (2.0/M)*X_mini.T @ (X_mini @ beta - z_mini)
      
        	# Calculate the outer product of the gradients
        Giter +=gradients @ gradients.T 
        #Simpler algorithm with only diagonal elements
        Ginverse = np.c_[eta/(delta+np.sqrt(np.diagonal(Giter)))]
        # compute update
        update = np.multiply(Ginverse,gradients)
        beta -= update
    err=np.append(err,MSE(z,X @ beta))
        
    
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Mean squared error')
plt.ylim((10**-7,10**1))
plt.plot(err,"b-")
plt.savefig("../Results/Solver/SGD_ADAGRAD.png",dpi=150)
plt.show()
  
print("Beta with adagrad SGD")
print(beta.T)
print("Training error")
print("MSE =",MSE(z,X @ beta))
print("R2 =",R2(z,X @ beta))
    

#OLS with scikit

model = Pipeline([('poly', PolynomialFeatures(degree=maxdegree)),('linear',\
              LinearRegression(fit_intercept=False))])
model = model.fit(x1,z) 
Beta = model.named_steps['linear'].coef_


z_fit = model.predict(x1)


print("\nBeta with Scikit")
print(Beta)
print("Training error")
print("MSE =",MSE(z,z_fit))
print("R2 =",R2(z,z_fit))

