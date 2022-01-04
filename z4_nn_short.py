import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat
import utils

data = loadmat(os.path.join('Data', 'ex3data1.mat'))
X, y = data['X'], data['y'].ravel()
y[y == 10] = 0
m = y.size

indices = np.random.permutation(m)

input_layer_size  = 400
hidden_layer_size = 25
num_labels = 10

weights = loadmat(os.path.join('Data', 'ex3weights.mat'))
Theta1, Theta2 = weights['Theta1'], weights['Theta2']
Theta2 = np.roll(Theta2, 1, axis=0) # cuộn

def predict(Theta1, Theta2, X):
    if X.ndim == 1:
        X = X[None]
    m = X.shape[0]
    num_labels = Theta2.shape[0]
    p = np.zeros(X.shape[0])
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    #Ví dụ có 3 ảnh cần đoán, ta có X = 3 x 401, m = 3
    for i in range(m):
        input_layer = X[i] # 1 x 401
        hidden_layer = utils.sigmoid(np.dot(input_layer, Theta1.T)) # 1 x 25
        hidden_layer = np.concatenate([np.ones([1]), hidden_layer], axis=0) # 1 x 26
        output_layer = utils.sigmoid(np.dot(hidden_layer, Theta2.T)) # 1 x 10
        p[i] = np.argmax(output_layer, axis=0) # p[i] sẽ bằng CHỈ SỐ của số có giá trị lớn nhất trong mảng 10 phần tử ở trên
    return p

pred = predict(Theta1, Theta2, X)
print('Training Set Accuracy: {:.1f}%'.format(np.mean(pred == y) * 100))

if indices.size > 0:
    i, indices = indices[0], indices[1:]
    utils.displayData(X[i, :], figsize=(4, 4))
    pyplot.show()
    pred = predict(Theta1, Theta2, X[i, :])
    print('Neural Network Prediction: {}'.format(*pred))
else:
    print('No more images to display!')