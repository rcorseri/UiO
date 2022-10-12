
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


#Plot also the parameters 𝛽 as you increase the order of the polynomial.
#Beta coefficients (up to 21)

#print(predictor.shape)

#plt.plot(predictor[0:55])

#plt.plot(predictor[0:3],'md-' , label='degree=1')
#plt.plot(predictor[3:9],'r-*' , label='degree=2')
#plt.plot(predictor[9:19],'b-*' , label='degree=3')
#plt.plot(predictor[19:34],'g*-' , label='degree=4')
#plt.plot(predictor[34:55],'y*-' , label='degree=5')
#
#locs, labels = plt.xticks()  # Get the current locations and labels.
#plt.xticks(np.arange(0, 1, step=1))  # Set label locations.
#plt.xticks(np.arange(21), [r'$\beta_0$', r'$\beta_1$', r'$\beta_2$', \
#           r'$\beta_3$', r'$\beta_4$', r'$\beta_5$', \
#           r'$\beta_6$', r'$\beta_7$', r'$\beta_8$', \
#           r'$\beta_9$', r'$\beta_{10}$', r'$\beta_{11}$', \
#           r'$\beta_{12}$', r'$\beta_{13}$', r'$\beta_{14}$', \
#           r'$\beta_{15}$', r'$\beta_{16}$', r'$\beta_{17}$', \
#           r'$\beta_{18}$', r'$\beta_{19}$', r'$\beta_{20}$'\
#           ], rotation=45)  # Set text labels.
#
#plt.ylabel("Optimal Beta - predictor value")
#plt.legend(loc='lower right',prop={'size': 8})
#plt.show()
#plt.savefig("Results/OLS/OLS_Beta_Optimal_degree5.png",dpi=150)



##Beta coefficients with error bars for degree 1, 3 and 5 
#plt.errorbar(np.array(range(0,21)),predictor[34:55], yerr=predictor_std[34:55],uplims=True, lolims=True, fmt='o', markersize=4, capsize=1,label='degree=5')
#plt.errorbar(np.array(range(0,10)),predictor[9:19], yerr=predictor_std[9:19],uplims=True, lolims=True, fmt='o', markersize=4, capsize=1,label='degree=3')
#plt.errorbar(np.array(range(0,3)),predictor[0:3], yerr=predictor_std[0:3],uplims=True, lolims=True,fmt='o', markersize=4, capsize=1, label='degree=1')
#
#locs, labels = plt.xticks()  # Get the current locations and labels.
#plt.xticks(np.arange(0, 1, step=1))  # Set label locations.
#plt.xticks(np.arange(21), [r'$\beta_0$', r'$\beta_1$', r'$\beta_2$', \
#           r'$\beta_3$', r'$\beta_4$', r'$\beta_5$', \
#           r'$\beta_6$', r'$\beta_7$', r'$\beta_8$', \
#           r'$\beta_9$', r'$\beta_{10}$', r'$\beta_{11}$', \
#           r'$\beta_{12}$', r'$\beta_{13}$', r'$\beta_{14}$', \
#           r'$\beta_{15}$', r'$\beta_{16}$', r'$\beta_{17}$', \
#           r'$\beta_{18}$', r'$\beta_{19}$', r'$\beta_{20}$',r'$\beta_{21}$'\
#           ], rotation=45)  # Set text labels.
#
#
#plt.ylabel("Optimal Beta - predictor value")
#plt.legend()
#plt.show()
#plt.savefig("Results/OLS/OLS_Beta_Optimal_degree5_std.png",dpi=150)
