

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
from Functions import Beta_std, FrankeFunction, R2, MSE, DesignMatrix, LinReg
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

#Create data
#np.random.seed(2003)
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


# Split the data in test (80%) and training dataset (20%) 
x_train, x_test, z_train, z_test = train_test_split(x1, z, test_size=0.2)



#Plot the resulting scores (MSE and R2) as functions of the polynomial degree (here up to polymial degree five). 

#Initialize before looping:
TestError = np.zeros(maxdegree)
TrainError = np.zeros(maxdegree)
TestR2 = np.zeros(maxdegree)
TrainR2 = np.zeros(maxdegree)
polydegree = np.zeros(maxdegree)
predictor = []
predictor_std = []


    
X_train = DesignMatrix(x_train[:,0],x_train[:,1],maxdegree)
X_test = DesignMatrix(x_test[:,0],x_test[:,1],maxdegree)

#OLS with matrix inversion

z_fit, z_pred, betas = LinReg(X_train, X_test, z_train)

print("Beta from matrice inversion")
print(betas)
print("Training error")
print("MSE =",MSE(z_train,z_fit))
print("R2 =",R2(z_train,z_fit))  
print("Testing error")
print("MSE =",MSE(z_test,z_pred))
print("R2  =",R2(z_test,z_pred))


#OLS with scikit

model = Pipeline([('poly', PolynomialFeatures(degree=maxdegree)),('linear',\
              LinearRegression(fit_intercept=False))])
model = model.fit(x_train,z_train) 
Beta = model.named_steps['linear'].coef_


z_fit = model.predict(x_train)
z_pred = model.predict(x_test) 


print("\nBeta with Scikit")
print(Beta)
print("Training error")
print("MSE =",MSE(z_train,z_fit))
print("R2 =",R2(z_train,z_fit)) 
print("Testing error")
print("MSE =",MSE(z_test,z_pred))
print("R2  =",R2(z_test,z_pred))



#OLS With adagra gradient descent

beta = np.random.randn(X_train.shape[1],1)
eta = 0.01
eps = [1]
err = []
i=0


delta = 10**-8
j = 0
err=[]
Giter = np.zeros(shape=(X_train.shape[1],X_train.shape[1]))

# improve with momentum gradient descent
change = 0.0
delta_momentum = 0.3

while(eps[-1] >= 10**(-2)) :
    d = z_train.shape[0] 
    z_train = np.reshape(z_train,(d,1))
    gradients = (2.0/d)*X_train.T @ (X_train @ beta - z_train)
    
  	# Calculate the outer product of the gradients
    Giter +=gradients @ gradients.T 
    #Simpler algorithm with only diagonal elements
    Ginverse = np.c_[eta/(delta+np.sqrt(np.diagonal(Giter)))]
    
    # compute update
    new_change = np.multiply(Ginverse,gradients) + delta_momentum*change        
    beta -= new_change
    change = new_change
        
    eps = np.append(eps,np.linalg.norm(gradients))
    err = np.append(err,MSE(z_train,X_train @ beta))
    
    i+=1
   
    
plt.yscale('log')
plt.xlabel('iterations')
plt.ylabel('Mean squared error')
plt.ylim((10**-7,10**1))
plt.plot(err,"b-")
plt.savefig("../Results/Solver/GD_ADAGRAD_momentum.png",dpi=150)
plt.show()

beta = np.reshape(beta,(beta.shape[0],))
z_train = np.reshape(z_train,(d,))

z_pred = X_test @ beta
z_fit = X_train @ beta



print("\nBeta with adagra Gradient Descent")
print(beta)
print("Training error")
print("MSE =",MSE(z_train,z_fit))
print("R2 =",R2(z_train,z_fit))
    
print("Testing error")
  
print("MSE =",MSE(z_test,z_pred))
print("R2  =",R2(z_test,z_pred))