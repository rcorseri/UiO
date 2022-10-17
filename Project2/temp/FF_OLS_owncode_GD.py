
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
from Functions import Beta_std, FrankeFunction, R2, MSE, DesignMatrix, LinReg

#Create data
#np.random.seed(2003)
n = 100
maxdegree = 5

x = np.random.uniform(0,1,n)
y = np.random.uniform(0,1,n)
z = FrankeFunction(x, y)

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


for degree in range(maxdegree):

#    z_fit, z_pred, betas = LinReg(X_train, X_test, z_train)
#    predictor=np.append(predictor,betas)
    
    X_train = DesignMatrix(x_train[:,0],x_train[:,1],degree+1)
    X_test = DesignMatrix(x_test[:,0],x_test[:,1],degree+1)
    beta = np.random.randn(X_train.shape[1],1)
    
 
    beta = np.random.randn(X_train.shape[1],1)
    eta = 0.001
    eps = 1
    i=0

    while(eps >= 10**(-1)) :
        d = z_train.shape[0] 
        z_train = np.reshape(z_train,(d,1))
        gradient = (2.0/d)*X_train.T @ (X_train @ beta - z_train)
        eps = np.linalg.norm(gradient)
        beta -= eta*gradient
        i+=1
        
    print(i)

    beta = np.reshape(beta,(beta.shape[0],))
    z_train = np.reshape(z_train,(d,))
    
    z_pred = X_test @ beta
    z_fit = X_train @ beta

        
    predictor=np.append(predictor,beta)
    #Accumulate standard deviation values for each Beta
    Beta_err = Beta_std(var, X_train, beta, beta.shape[0])
    predictor_std = np.append(predictor_std,Beta_err)

    polydegree[degree] = degree+1    
    TestError[degree] = MSE(z_test, z_pred)
    TrainError[degree] = MSE(z_train, z_fit)
    TestR2[degree] = R2(z_test,z_pred)
    TrainR2[degree] = R2(z_train,z_fit)
    
    #Display regression results for each polynomial degree
    print("\nModel complexity:")
    print(degree+1)
    print("\nOptimal estimator Beta")
    print(beta.shape)
    print("\nTraining error")
    print("MSE =",MSE(z_train,z_fit))
    print("R2 =",R2(z_train,z_fit))
    print("\nTesting error")
    print("MSE =",MSE(z_test,z_pred))
    print("R2  =",R2(z_test,z_pred))

#Plots 

#MSE   
plt.plot(polydegree, TestError, label='Test sample')
plt.plot(polydegree, TrainError, label='Train sample')
plt.xlabel('Model complexity (degree)')
plt.ylabel('Mean Square Error')
plt.xticks(np.arange(1, maxdegree+1, step=1))  # Set label locations.
plt.legend()
plt.savefig("Results/OLS/MSE_vs_complexity_withGD.png", dpi=150)
plt.show()

#R2 score
plt.plot(polydegree, TestR2, label='Test sample')
plt.plot(polydegree, TrainR2, label='Train sample')
plt.xlabel('Model complexity')
plt.ylabel('R2 score')
plt.xticks(np.arange(1, maxdegree+1, step=1))  # Set label locations.
plt.legend()
plt.savefig("Results/OLS/R2_vs_complexity_withGD.png", dpi=150)
plt.show()


