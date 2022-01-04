import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat
import utils

Theta2 = np.array([1,2,3,4,5,6,7,8,9,10])
m = Theta2.size
a = np.random.permutation(m)
print(a)


