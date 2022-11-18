import matplotlib.pyplot as plt
from  matplotlib.colors import LogNorm
import numpy as np
from sklearn.model_selection import train_test_split
from Functions import Beta_std, FrankeFunction, R2, MSE, DesignMatrix, LinReg
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
import seaborn as sns
from NeuralNetwork_regression_leakyrelu import NeuralNetwork, leakyrelu, leakyrelu_grad, create_mini_batches
from sklearn.neural_network import MLPRegressor

 
#Create data
np.random.seed(2003)
n = 100

x = np.random.uniform(0,1,n)
y = np.random.uniform(0,1,n)
z = FrankeFunction(x, y)

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
epochs = 10000 #number of epochs 

etas = np.logspace(-5, -1, 5)
lambdas = np.logspace(-5, 2, 8)
n_hidden_neurons = [4]
n_categories = 1
n_features = x_train.shape[1]
n_inputs = x_train.shape[0]

#Initialize error to store
NN_err_train=np.zeros(shape=(lambdas.shape[0],etas.shape[0]))
NN_err_test=np.zeros(shape=(lambdas.shape[0],etas.shape[0]))
DNN_scikit_train = np.zeros(shape=(lambdas.shape[0],etas.shape[0]))
DNN_scikit_test = np.zeros(shape=(lambdas.shape[0],etas.shape[0]))

z_train_ravel = np.ravel(z_train)
z_test_ravel = np.ravel(z_test)


#looping through regularization and learning rates
i=0
for lmbd in lambdas:
    j=0  
    for eta in etas: 
        
        #####own NN implementation######
        dnn = NeuralNetwork(x_train, z_train, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=M, n_hidden_neurons=n_hidden_neurons, n_categories=n_categories)
        dnn.train()
        z_fit = dnn.predict(x_train)
        z_pred = dnn.predict(x_test)
          
        NN_err_train[i,j] = MSE(z_train,z_fit)
        NN_err_test[i,j] = MSE(z_test,z_pred)
        
        #####scikit NN implementation#####
        dnn2 = MLPRegressor(hidden_layer_sizes=n_hidden_neurons, solver = 'lbfgs', activation='relu',
                            alpha=lmbd, batch_size = M, learning_rate_init=eta, max_iter=epochs)
        dnn2.fit(x_train, z_train_ravel)
        z_fit2 = dnn2.predict(x_train)
        z_pred2 = dnn2.predict(x_test)
        
        DNN_scikit_train[i,j] = MSE(z_train_ravel,z_fit2)
        DNN_scikit_test[i,j] = MSE(z_test_ravel,z_pred2)
        
        #####Keras NN implementation#####
        
                
        j+=1
    i+=1
    


#Heatmaps for gridsearch on lambda and ridge
x_axis = etas # labels for x-axis
y_axis = lambdas # labels for y-axis 



heat1 = sns.heatmap(NN_err_train,vmin=0.009,vmax=10,annot=True, xticklabels=x_axis, yticklabels=y_axis, cmap="viridis",linewidths =0.5, norm = LogNorm() )
heat1.set(xlabel='learning rate', ylabel ='regularization', title = f"Training error ")
plt.savefig(f"Results/NN/FF_training.png", dpi=150)
plt.show()


heat2 = sns.heatmap(NN_err_test,vmin=0.009,vmax=10,annot=True, xticklabels=x_axis, yticklabels=y_axis, cmap="viridis",linewidths =0.5, norm = LogNorm() )
heat2.set(xlabel='learning rate', ylabel ='regularization', title = f"Test error ")
plt.savefig(f"Results/NN/FF_test_error.png", dpi=150)
plt.show()


heat3 = sns.heatmap(DNN_scikit_train,vmin=0.009,vmax=10,annot=True, xticklabels=x_axis, yticklabels=y_axis, cmap="viridis",linewidths =0.5, norm = LogNorm() )
heat3.set(xlabel='learning rate', ylabel ='regularization', title = f"Training error Scikit ")
plt.savefig(f"Results/NN/FF_training_scikit.png", dpi=150)
plt.show()


heat4 = sns.heatmap(DNN_scikit_test,vmin=0.009,vmax=10,annot=True, xticklabels=x_axis, yticklabels=y_axis, cmap="viridis",linewidths =0.5, norm = LogNorm() )
heat4.set(xlabel='learning rate', ylabel ='regularization', title = f"Test error Scikit ")
plt.savefig(f"Results/NN/FF_test_error_scikit.png", dpi=150)
plt.show()



