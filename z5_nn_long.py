import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat
import utils

#  training data stored in arrays X, y
data = loadmat(os.path.join('Data', 'ex3data1.mat'))
X, y = data['X'], data['y'].ravel()

# set the zero digit to 0, rather than its mapped 10 in this dataset
# This is an artifact due to the fact that this dataset was used in 
# MATLAB where there is no index 0
y[y == 10] = 0

# get number of examples in dataset
m = y.size

# các ví dụ hoán vị ngẫu nhiên, được sử dụng để hình dung một hình ảnh tại một thời điểm
indices = np.random.permutation(m)

# Randomly select 100 data points to display
rand_indices = np.random.choice(m, 100, replace=False)
sel = X[rand_indices, :]

utils.displayData(sel)

# Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 0 to 9

# Load the .mat file, which returns a dictionary 
weights = loadmat(os.path.join('Data', 'ex3weights.mat'))

# get the model weights from the dictionary
# Theta1 has size 25 x 401
# Theta2 has size 10 x 26
Theta1, Theta2 = weights['Theta1'], weights['Theta2']

# hoán đổi cột đầu tiên và cột cuối cùng của Theta2, do kế thừa từ lập chỉ mục MATLAB, 
# vì tệp trọng lượng ex3weights.mat được lưu dựa trên lập chỉ mục MATLAB
Theta2 = np.roll(Theta2, 1, axis=0)

def predict(Theta1, Theta2, X):
    """
    Dự đoán nhãn của đầu vào được cung cấp cho một mạng nơ-ron được đào tạo.
    
    Parameters
    ----------
    Theta1 : array_like
        Trọng số của lớp đầu tiên trong mạng nơ-ron. 
        Nó có hình dạng (kích thước lớp ẩn thứ 2 x kích thước đầu vào)
    
    Theta2: array_like
        Trọng số của lớp thứ hai trong mạng nơ-ron.
        Nó có hình dạng (kích thước lớp đầu ra x kích thước lớp ẩn thứ 2)
    
    X : array_like
        Đầu vào hình ảnh có hình dạng (số lượng ví dụ x kích thước hình ảnh).
    
    Return 
    ------
    p : array_like
        Vectơ dự đoán có chứa nhãn dự đoán cho mỗi ví dụ.
        Nó có chiều dài bằng số lượng ví dụ.
    
    Instructions
    ------------
    Hoàn thành đoạn mã sau để đưa ra dự đoán bằng cách sử dụng mạng nơ-ron đã học của bạn.
    Bạn nên đặt p thành một vectơ chứa các nhãn từ 0 đến (num_labels-1).
     
    Hint
    ----
    Mã này có thể được vector hóa tất cả bằng cách sử dụng hàm argmax numpy. 
    Đặc biệt, hàm argmax trả về chỉ số của phần tử max, 
    để biết thêm thông tin, hãy xem '? Np.argmax' hoặc tìm kiếm trực tuyến. 
    Nếu các ví dụ của bạn nằm trong các hàng, 
    thì bạn có thể sử dụng np.argmax (A, axis = 1) để lấy chỉ số của giá trị lớn nhất cho mỗi hàng.
    
    Note
    ----
    Hãy nhớ rằng chúng tôi đã cung cấp hàm `sigmoid` trong tệp` utils.py`. 
    Bạn có thể sử dụng hàm này bằng cách gọi `utils.sigmoid (z)`, 
    trong đó bạn có thể thay thế `z` bằng biến đầu vào bắt buộc thành sigmoid.
    """
    # Make sure the input has two dimensions
    if X.ndim == 1:
        X = X[None]  # promote to 2-dimensions
    
    # useful variables
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    # You need to return the following variables correctly 
    p = np.zeros(X.shape[0])

    # ====================== YOUR CODE HERE ======================

    # Add bias unit to X 
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    for i in range(m):
        input_layer = X[i]
        hidden_layer = utils.sigmoid(np.dot(input_layer, Theta1.T))
        hidden_layer = np.concatenate([np.ones([1]), hidden_layer], axis=0)
        output_layer = utils.sigmoid(np.dot(hidden_layer, Theta2.T))
        p[i] = np.argmax(output_layer, axis=0)

    # =============================================================
    return p

pred = predict(Theta1, Theta2, X)
print('Training Set Accuracy: {:.1f}%'.format(np.mean(pred == y) * 100))

if indices.size > 0:
    i, indices = indices[0], indices[1:]
    utils.displayData(X[i, :], figsize=(4, 4))
    pred = predict(Theta1, Theta2, X[i, :])
    print('Neural Network Prediction: {}'.format(*pred))
else:
    print('No more images to display!')