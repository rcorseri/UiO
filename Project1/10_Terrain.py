'''This program performs Ordinary least square, Ridge and Lasso regression on a terrain dataset
and cross-validation as resampling technique to evaluate which model fits the data best.
Author: R Corseri & L Barreiro'''

#%%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from Functions import MSE, DesignMatrix, LinReg, RidgeReg, LassoReg
from imageio import imread
from matplotlib import cm
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso
import seaborn as sb
from matplotlib.ticker import LinearLocator, FormatStrFormatter


#%%
# Load the terrain
terrain = imread('data/SRTM_data_Norway_1.tif')

# Show the terrain
plt.figure()
plt.title('Terrain over Norway')
plt.imshow(terrain, cmap='viridis')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar()
plt.savefig("Results/Terrain/Map_v01.png",dpi=150)
plt.show()
#%%
n = 1000
maxdegree = 5 # polynomial order
terrain = terrain[:n,:n]
# Creates mesh of image pixels
x = np.linspace(0,1, np.shape(terrain)[0])
y = np.linspace(0,1, np.shape(terrain)[1])
x_mesh, y_mesh = np.meshgrid(x,y)

x_mesh = np.ravel(x_mesh)
y_mesh = np.ravel(y_mesh)

z = terrain.ravel()

# Show the terrain
plt.figure()
plt.title('Terrain over Norway')
plt.imshow(terrain, cmap='viridis')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar()
plt.savefig("Results/Terrain/Map_v02.png",dpi=150)
plt.show()

#%%
#Plot 3D
#x = np.linspace(0,1, np.shape(terrain)[0])
#y = np.linspace(0,1, np.shape(terrain)[1])
#z=terrain
#
#fig = plt.figure()
#ax = fig.gca(projection='3d')

#x, y = np.meshgrid(x,y)
##z=np.reshape(terrain,x)
#
## Plot the surface without noise
#surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)
## Customize the z axis.
#ax.set_zlim(-0.10, 1.40)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#ax.set_zticks([0, 500, 1000, 1500])
#ax.axes.xaxis.set_ticklabels([])
#ax.axes.yaxis.set_ticklabels([])
#ax.axes.zaxis.set_ticklabels([])
#plt.title('Original terrain')
## Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)
##plt.savefig("plots/Terrain/Map_3D.png", dpi=150)
#plt.show()

#%% Cross validation in OLS

#Scale the data
x = np.linspace(0,1, np.shape(terrain)[0])
y = np.linspace(0,1, np.shape(terrain)[1])
x = x.reshape(n,1)
y = y.reshape(n,1)

scaler = StandardScaler()

# scaler.fit(x)
# x_scaled = scaler.transform(x)

# scaler.fit(y)
# y_scaled = scaler.transform(y)

z=terrain
scaler.fit(z)
z_scaled = scaler.transform(z)



MSE_test = np.zeros(maxdegree)
MSE_train = np.zeros(maxdegree)
k=10

# Decide degree on polynomial to fit
poly = PolynomialFeatures(degree = 6)

# Initialize a KFold instance
k = 10
kfold = KFold(n_splits = k)

# Perform the cross-validation to estimate MSE
scores_KFold_Train = np.zeros((maxdegree, k))
scores_KFold_Test = np.zeros((maxdegree, k))

#
polydegree = np.zeros(maxdegree)

i = 0
for degree in range(maxdegree):
    polydegree[degree] = degree+1
    X = DesignMatrix(x,y,degree+1)
    j = 0
    for train_inds, test_inds in kfold.split(x):
        X_train = X[train_inds]
        z_train = z_scaled[train_inds]

        X_test = X[test_inds]
        z_test = z_scaled[test_inds]
  
        z_fit, z_pred, beta = LinReg(X_train, X_test, z_train)

        scores_KFold_Train[i,j] = MSE(z_train, z_fit)
        scores_KFold_Test[i,j] = MSE(z_test, z_pred)

        j += 1
    i += 1
    
estimated_mse_KFold_train = np.mean(scores_KFold_Train, axis = 1)
estimated_mse_KFold_test = np.mean(scores_KFold_Test, axis = 1)

plt.figure()
plt.plot(polydegree, estimated_mse_KFold_train, label = 'KFold train')
plt.plot(polydegree, estimated_mse_KFold_test, label = 'KFold test')
plt.xlabel('Complexity')
plt.ylabel('mse')
plt.xticks(np.arange(1, maxdegree+1, step=1))  # Set label locations.
plt.legend()
plt.title('K-fold Cross Validation, k = 10, OLS')
#plt.savefig("plots/Terrain/CV_OLS.png",dpi=150)
plt.show()

#%% Run it also for maxdegree=10
maxdegree=10

MSE_test = np.zeros(maxdegree)
MSE_train = np.zeros(maxdegree)
k=10

# Decide degree on polynomial to fit
poly = PolynomialFeatures(degree = 6)

# Initialize a KFold instance
k = 10
kfold = KFold(n_splits = k)

# Perform the cross-validation to estimate MSE
scores_KFold_Train = np.zeros((maxdegree, k))
scores_KFold_Test = np.zeros((maxdegree, k))

#
polydegree = np.zeros(maxdegree)

i = 0
for degree in range(maxdegree):
    polydegree[degree] = degree+1
    X = DesignMatrix(x,y,degree+1)
    j = 0
    for train_inds, test_inds in kfold.split(x):
        X_train = X[train_inds]
        z_train = z_scaled[train_inds]

        X_test = X[test_inds]
        z_test = z_scaled[test_inds]
  
        z_fit, z_pred, beta = LinReg(X_train, X_test, z_train)

        scores_KFold_Train[i,j] = MSE(z_train, z_fit)
        scores_KFold_Test[i,j] = MSE(z_test, z_pred)

        j += 1
    i += 1
    
estimated_mse_KFold_train = np.mean(scores_KFold_Train, axis = 1)
estimated_mse_KFold_test = np.mean(scores_KFold_Test, axis = 1)

plt.figure()
plt.plot(polydegree, estimated_mse_KFold_train, label = 'KFold train')
plt.plot(polydegree, estimated_mse_KFold_test, label = 'KFold test')
plt.xlabel('Complexity')
plt.ylabel('mse')
plt.xticks(np.arange(1, maxdegree+1, step=1))  # Set label locations.
plt.legend()
plt.title('K-fold Cross Validation, k = 10, OLS')
plt.savefig("Results/Terrain/CV_OLS_pol1to10.png",dpi=150)
plt.show()

#%% Make some more OLS plots
#For complexity=4

x = np.linspace(0,1, np.shape(terrain)[0])
y = np.linspace(0,1, np.shape(terrain)[1])
z=terrain

deg=3
X1 = DesignMatrix(x,y,deg)
OLSbeta1 = np.linalg.pinv(X1.T @ X1) @ X1.T @ z
ytilde1 = X1 @ OLSbeta1

# Show the terrain
plt.figure()
plt.title('Terrain over Norway, OLS, pol=4')
plt.imshow(ytilde1, cmap='viridis')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig("Results/Terrain/Map_v03_OLS_pol3.png",dpi=150)
plt.show()




#Plot 3D
x = np.linspace(0,1, np.shape(terrain)[0])
y = np.linspace(0,1, np.shape(terrain)[1])

fig = plt.figure()
ax = fig.gca(projection='3d')

x, y = np.meshgrid(x,y)
#z=np.reshape(terrain,x)

# Plot the surface without noise
surf = ax.plot_surface(x, y, ytilde1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_zticks([0, 500, 1000, 1500])
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
ax.axes.zaxis.set_ticklabels([])
plt.title('OLS, pol=3')
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.savefig("plots/Terrain/Map_3d_OLS_pol3.png", dpi=150)
plt.show()

#%%
# CV in Ridge
maxdegree = 10 # polynomial order

#set up the hyper-parameters to investigate
nlambdas = 9
lambdas = np.logspace(-4, 4, nlambdas)


# Plot all in the same figure as subplots

#Initialize before looping:
polydegree = np.zeros(maxdegree)
error_Kfold_train = np.zeros((maxdegree,k))
error_Kfold_test = np.zeros((maxdegree,k))
estimated_mse_Kfold_train = np.zeros(maxdegree)
estimated_mse_Kfold_test = np.zeros(maxdegree)
bias = np.zeros(maxdegree)
variance = np.zeros(maxdegree)

Etest = np.zeros((maxdegree,9))
Etrain = np.zeros((maxdegree,9))

# Create a matplotlib figure
fig, ax = plt.subplots()

for l in range(nlambdas):   
    i=0
    for degree in range(maxdegree): 
        j=0
        for train_inds, test_inds in kfold.split(x):
            
            X = DesignMatrix(x,y,degree+1)
            
            X_train = X[train_inds]
            z_train = z_scaled[train_inds]   
            X_test = X[test_inds]
            z_test = z_scaled[test_inds]
                 
            z_fit, z_pred, Beta = RidgeReg(X_train, X_test, z_train, z_test,lambdas[l])
            
            error_Kfold_test[i,j] = MSE(z_test,z_pred)
            error_Kfold_train[i,j] = MSE(z_train,z_fit)
            
            j+=1
        
        estimated_mse_Kfold_test[degree] = np.mean(error_Kfold_test[i,:])
        estimated_mse_Kfold_train[degree] = np.mean(error_Kfold_train[i,:])
        polydegree[degree] = degree+1
                
        i+=1
    
    Etest[:,l] = estimated_mse_Kfold_test
    Etrain[:,l] = estimated_mse_Kfold_train
    ax.plot(polydegree, estimated_mse_Kfold_test, label='%.0e' %lambdas[l])

plt.xlabel('Model complexity')    
plt.xticks(np.arange(1, len(polydegree)+1, step=1))  # Set label locations.
plt.ylabel('MSE')
plt.title('MSE Ridge regression for different lambdas (kfold=10)')

# Add a legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], title='lambda', loc='center right', bbox_to_anchor=(1.27, 0.5))

#Save figure
plt.savefig("plots/Terrain/CV_Ridge_lambda_pol1to10.png",dpi=150, bbox_inches='tight')
plt.show()

#Compare train and test performance
plt.figure()
plt.plot(polydegree, Etrain[:,2], label = 'KFold train')
plt.plot(polydegree, Etest[:,2], label = 'KFold test')
plt.xlabel('Complexity')
plt.ylabel('mse')
plt.xticks(np.arange(1, maxdegree+1, step=1))  # Set label locations.
plt.legend()
plt.title('K-fold Cross Validation, k=10, Ridge, lambda=10')
plt.savefig("Results/Terrain/CV_Ridge_pol1to10.png",dpi=150)
plt.show()

#Create a heatmap with the error per nlambdas and polynomial degree
heatmap = sb.heatmap(Etest,annot=True, annot_kws={"size":7}, cmap="coolwarm", xticklabels=lambdas, yticklabels=range(1,maxdegree+1), cbar_kws={'label': 'Mean squared error'})
heatmap.invert_yaxis()
heatmap.set_ylabel("Complexity")
heatmap.set_xlabel("lambda")
heatmap.set_title("MSE heatmap, Cross Validation, kfold = {:}".format(k))
plt.tight_layout()
plt.savefig("Results/Terrain/CV_Ridge_heatmap_pol1to10.png",dpi=150)
plt.show()

#%%
#Make some more Ridge plots
#For complexity=1, lambda=10^3 (MSE_train=0.82)
x = np.linspace(0,1, np.shape(terrain)[0])
y = np.linspace(0,1, np.shape(terrain)[1])
z=terrain


deg=5
lmb=10

X1 = DesignMatrix(x,y,deg)
Ridgebeta1 = np.linalg.pinv(X1.T @ X1 + lmb*np.identity(X1.shape[1])) @ X1.T @ z
ytilde2 = X1 @ Ridgebeta1



# Show the terrain
plt.figure()
plt.title('Terrain over Norway, Ridge')
plt.imshow(ytilde2, cmap='viridis')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig("Results/Terrain/Map_v04_Ridge_pol5_lmb10.png",dpi=150)
plt.show()




#Plot 3D
x = np.linspace(0,1, np.shape(terrain)[0])
y = np.linspace(0,1, np.shape(terrain)[1])

fig = plt.figure()
ax = fig.gca(projection='3d')

x, y = np.meshgrid(x,y)
#z=np.reshape(terrain,x)

# Plot the surface without noise
surf = ax.plot_surface(x, y, ytilde2, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_zticks([0, 500,1000,1500])
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
ax.axes.zaxis.set_ticklabels([])
plt.title('Ridge, pol=4, lmb=10')
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig("Results/Terrain/Map_3d_Ridge_pol4_lmb10.png", dpi=150)
plt.show()

#Compare train and test performance
plt.figure()
plt.plot(polydegree, Etrain[:,2], label = 'KFold train')
plt.plot(polydegree, Etest[:,2], label = 'KFold test')
plt.xlabel('Complexity')
plt.ylabel('mse')
plt.xticks(np.arange(1, maxdegree+1, step=1))  # Set label locations.
plt.legend()
plt.title('K-fold Cross Validation, k = 10, Ridge, lambda=10')
plt.savefig("Results/Terrain/CV_Ridge.png",dpi=150)
plt.show()



#%%
#CV in Lasso

#set up the hyper-parameters to investigate
maxdegree=10
nlambdas = 9
lambdas = np.logspace(-6, 2, nlambdas)

#Initialize before looping:
polydegree = np.zeros(maxdegree)
error_Kfold_test = np.zeros((maxdegree,k))
error_Kfold_train = np.zeros((maxdegree,k))
estimated_mse_Kfold_train = np.zeros(maxdegree)
estimated_mse_Kfold_test = np.zeros(maxdegree)
bias = np.zeros(maxdegree)
variance = np.zeros(maxdegree)

Etrain = np.zeros((maxdegree,9))
Etest = np.zeros((maxdegree,9))

# Create a matplotlib figure
fig, ax = plt.subplots()

for l in range(nlambdas):   
    i=0
    for degree in range(maxdegree): 
        j=0
        for train_inds, test_inds in kfold.split(x):
            
            X = DesignMatrix(x,y,degree+1)
            
            X_train = X[train_inds]
            z_train = z_scaled[train_inds]   
            X_test = X[test_inds]
            z_test = z_scaled[test_inds]
            
            z_fit, z_pred = LassoReg(X_train, X_test, z_train, z_test,lambdas[l])
            
            error_Kfold_test[i,j] = MSE(z_test,z_pred)
            error_Kfold_train[i,j] = MSE(z_train,z_fit)
            
            j+=1
            
        estimated_mse_Kfold_test[degree] = np.mean(error_Kfold_test[i,:])
        estimated_mse_Kfold_train[degree] = np.mean(error_Kfold_train[i,:])
        polydegree[degree] = degree+1
                
        i+=1

    Etest[:,l] = estimated_mse_Kfold_test
    Etrain[:,l] = estimated_mse_Kfold_train
    ax.plot(polydegree, estimated_mse_Kfold_test, label='%.0e' %lambdas[l])

plt.xlabel('Model complexity')    
plt.xticks(np.arange(1, len(polydegree)+1, step=1))  # Set label locations.
plt.ylabel('MSE')
plt.title('MSE Lasso regression for different lambdas (Kfold=10), pol1to10')

# Add a legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], title='lambda', loc='center right', bbox_to_anchor=(1.27, 0.5))

#Save figure
#plt.savefig("plots/Terrain/CV_Lasso_lambda.png",dpi=150, bbox_inches='tight')
plt.show()

#Compare train and test performance
plt.figure()
plt.plot(polydegree, Etrain[:,4], label = 'KFold train')
plt.plot(polydegree, Etest[:,4], label = 'KFold test')
plt.xlabel('Complexity')
plt.ylabel('mse')
plt.xticks(np.arange(1, maxdegree+1, step=1))  # Set label locations.
plt.legend()
plt.title('K-fold Cross Validation, k = 10, Lasso, lambda=0.01')
plt.savefig("Results/Terrain/CV_Lasso_pol1to10.png",dpi=150)
plt.show()

#%%
#Create a heatmap with the error per nlambdas and polynomial degree

heatmap = sb.heatmap(Etest,annot=True, annot_kws={"size":7}, cmap="coolwarm", xticklabels=lambdas, yticklabels=range(1,maxdegree+1), cbar_kws={'label': 'Mean squared error'})
heatmap.invert_yaxis()
heatmap.set_ylabel("Complexity")
heatmap.set_xlabel("lambda")
heatmap.set_title("MSE heatmap, Cross Validation, kfold = {:}".format(k))
plt.tight_layout()
plt.savefig("Results/Terrain/CV_Lasso_heatmap_pol1to10.png",dpi=150)
plt.show()

#%%
#Make some more Lasso plots
#For complexity=4, lambda=10^-1 (MSE_train=0.82)
x = np.linspace(0,1, np.shape(terrain)[0])
y = np.linspace(0,1, np.shape(terrain)[1])
z=terrain


deg=4
lmb=0.01

X1 = DesignMatrix(x,y,deg)
modelLasso = Lasso(lmb,fit_intercept=False)
modelLasso.fit(X1,z)
ytilde3 = modelLasso.predict(X1)

# Show the terrain
plt.figure()
plt.title('Terrain over Norway, Lasso')
plt.imshow(ytilde3, cmap='viridis')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig("Results/Terrain/Map_v05_Lasso_pol4_lmb0_01.png",dpi=150)
plt.show()




#Plot 3D
x = np.linspace(0,1, np.shape(terrain)[0])
y = np.linspace(0,1, np.shape(terrain)[1])

fig = plt.figure()
ax = fig.gca(projection='3d')

x, y = np.meshgrid(x,y)
#z=np.reshape(terrain,x)

# Plot the surface without noise
surf = ax.plot_surface(x, y, ytilde3, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_zticks([0, 500, 1000, 1500])
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
ax.axes.zaxis.set_ticklabels([])
plt.title('Lasso, pol=4, lmd=0.01')
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig("Results/Terrain/Map_3d_Lasso_pol4_lmb0_01.png", dpi=150)
plt.show()