import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import FrankeFunction as FF
import Calculate_MSE_R2 as error


# Make data set.
n = 1000

x = np.random.rand(n)
y = np.random.rand(n)
dataset = FF.Frank