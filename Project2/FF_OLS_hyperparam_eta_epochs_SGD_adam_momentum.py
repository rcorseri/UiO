

import matplotlib.pyplot as plt
from  matplotlib.colors import LogNorm
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
import seaborn as sns
#from tf.keras.optimizers import Adagrad
 

#Create data
np.random.seed(2003)
n = 100
maxdegree = 4

x = np.random.uniform(0,1,n)
y = np.random.uniform(0,1,n)
z = FrankeFunction(x, y)
#z = 1 + x + y + x*y + x**2 + y**2

# Add random distributed noise
var = 0.1
z = z + np.random.normal(0,var,z.shape)



x = np.array(x).reshape(n,1)
y = np.array(y).reshape(n,1)
x1 = np.hstack((x,y)).reshape(n,2)
z = np.reshape(z,(z.shape[0],1))


# Split the data in test (80%) and training dataset (20%) 
x_train, x_test, z_train, z_test = train_test_split(x1, z, test_size=0.2)

z = np.reshape(z_train,(z_train.shape[0],1))
z_test = np.reshape(z_test,(z_test.shape[0],1))


M = 20   #size of each minibatch
m = int(z.shape[0]/M) #number of minibatches
n_epochs = 1500 #number of epochs

#Set up RMSprop

#eta = 0.001
etas = np.logspace(-4,1,6)
# Value for parameter rho
rho = 0.99
rho2 = 0.9
delta = 10**-7
j = 0

# improve with momentum gradient descent
change = 0.0
delta_momentum = 0.3



#Initialize error to store
n_epochs = np.logspace(1,4,4)
n_epochs = n_epochs.astype(int)
#degrees = degrees.astype(int)

err_train=np.zeros(shape=(n_epochs.shape[0],etas.shape[0]))
err_test=np.zeros(shape=(n_epochs.shape[0],etas.shape[0]))

i=0
for i in range(len(n_epochs)):
    j=0
    X = DesignMatrix(x_train[:,0],x_train[:,1], maxdegree)
    X_test = DesignMatrix(x_test[:,0],x_test[:,1], maxdegree)
    beta = np.random.randn(X.shape[1],1)
    change = 0.0
    for eta in etas:
        
        for epoch in range(1,n_epochs[i]+1):
            mini_batches = create_mini_batches(X,z,M) 
            Giter = np.zeros(shape=(X.shape[1],X.shape[1]))
            gradients = np.zeros(shape=(X.shape[1],1))
    
            for minibatch in mini_batches:
                X_mini, z_mini = minibatch
                
                Previous = Giter
                Previous2 = gradients
                
                gradients = (2.0/M)*X_mini.T @ (X_mini @ beta - z_mini)
                
        
                # Calculate the outer product of the gradients
                Giter +=gradients @ gradients.T 
                
                #scaling with rho the new and the previous results
                Gnew = (rho*Previous+(1-rho)*Giter)
                Gnew2 = (rho2*Previous2+(1-rho2)*gradients)
                
                Gnew = (1/(1-rho))*Gnew
                Gnew2 = (1/(1-rho2))*Gnew2
                
                #Simpler algorithm with only diagonal elements
                Ginverse = np.c_[eta/(delta+np.sqrt(np.diagonal(Gnew)))]
                
                # compute update
                new_change = np.multiply(Ginverse,Gnew2) + delta_momentum*change        
                beta -= new_change
                change = new_change
                                  
        err_train[i,j] = MSE(z,X @ beta)
        err_test[i,j] = MSE(z_test,X_test @ beta)
        j+=1
    i+=1
   
        
         

#Heatmaps for gridsearch on lambda and ridge

x_axis = etas # labels for x-axis
y_axis = n_epochs # labels for y-axis 


#heat1 = sns.heatmap(err_train,vmin=0.002,vmax=10,annot=True, fmt=".1e",linewidths =0.5, norm = LogNorm() )
heat1 = sns.heatmap(err_train,vmin=0.009,vmax=1,annot=True, xticklabels=x_axis, yticklabels=y_axis, cmap="viridis",linewidths =0.5, norm = LogNorm() )
heat1.set(xlabel='learning rate', ylabel ='epochs', title = f"Training error \nSGD/ADAM/momentum")
plt.savefig(f"Results/OLS/ADAM/OLS_hyper_epochs_ADAM_train_error.png", dpi=150)
plt.show()

#heat2 = sns.heatmap(err_test,vmin=0.002,vmax=10,annot=True, fmt=".1e",linewidths =0.5, norm = LogNorm() )
heat2 = sns.heatmap(err_test,vmin=0.009,vmax=1,annot=True, xticklabels=x_axis, yticklabels=y_axis, cmap="viridis",linewidths =0.5, norm = LogNorm() )
heat2.set(xlabel='learning rate', ylabel ='epochs', title = f"Test error \nSGD/ADAM/momentum")
plt.savefig(f"Results/OLS/ADAM/OLS_hyper_epochs_ADAM_test_error.png", dpi=150)
plt.show()



#Ridge with scikit

model = Pipeline([('poly', PolynomialFeatures(degree=maxdegree)),('linear',\
              LinearRegression( fit_intercept=False))])
model = model.fit(x_train,z_train) 
Beta = model.named_steps['linear'].coef_


z_fit = model.predict(x_train)
z_pred = model.predict(x_test)

print("\nBeta with Scikit")
print(Beta)
print("Training error")
print("MSE =",MSE(z_train,z_fit))
print("R2 =",R2(z_train,z_fit))
print("Test error")
print("MSE =",MSE(z_test,z_pred))
print("R2 =",R2(z_test,z_pred))
