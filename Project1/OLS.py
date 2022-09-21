import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline


import FrankeFunction as FF
import Calculate_MSE_R2 as error


# Make data set.
n = 100

x1 = np.random.uniform(0,1,n)
x2 = np.random.uniform(0,1,n)

y = FF.FrankeFunction(x1,x2)#+np.random.normal(0,1,n)

x1 = np.array(x1).reshape(n,1)
x2 = np.array(x2).reshape(n,1)

x = np.hstack((x1,x2)).reshape(n,2)



#Split train (80%) and test(20%) data before looping on polynomial degree
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

y_train = (y_train-np.mean(y_train))/np.std(y_train)
y_test= (y_test-np.mean(y_test))/np.std(y_test)

#Perform polynomial regression using scikit-learn
#Define maximal model complexity
maxdegree= 1

model = Pipeline([('poly', PolynomialFeatures(degree=maxdegree)),('linear',\
                  LinearRegression(fit_intercept=False))])
model = model.fit(x_train,y_train)
Beta = model.named_steps['linear'].coef_

y_fit = model.predict(x_train)
y_pred = model.predict(x_test) 


print("\nOptimal estimator Beta")
print(Beta)

print("\n\nTraining error")
print("MSE =",error.MSE(y_train,y_fit))
print("R2 =",error.R2(y_train,y_fit))

print("\nTesting error")
print("MSE =",error.MSE(y_test,y_pred))
print("R2  =",error.R2(y_test,y_pred))













