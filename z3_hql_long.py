import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize
from scipy.io import loadmat
import utils

# 20x20 Input Images of Digits
input_layer_size  = 400

# 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)
num_labels = 10

#  training data stored in arrays X, y
data = loadmat(os.path.join('Data', 'ex3data1.mat'))
X, y = data['X'], data['y'].ravel()

# set the zero digit to 0, rather than its mapped 10 in this dataset
# This is an artifact due to the fact that this dataset was used in 
# MATLAB where there is no index 0
y[y == 10] = 0

m = y.size

# Randomly select 100 data points to display
# Xem 100 số (dạng ảnh 20x20pixel) bất kỳ trong 5000 số
rand_indices = np.random.choice(m, 100, replace=False)
sel = X[rand_indices, :]

utils.displayData(sel)

# test values for the parameters theta
theta_t = np.array([-2, -1, 1, 2], dtype=float)

# test values for the inputs
X_t = np.concatenate([np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order='F')/10.0], axis=1)

# test values for the labels
y_t = np.array([1, 0, 1, 0, 1])

# test value for the regularization parameter
lambda_t = 3

def lrCostFunction(theta, X, y, lambda_):
    """
    Tính toán chi phí sử dụng theta làm tham số 
    cho hồi quy logistic chính quy và gradient của chi phí w.r.t. vào các tham số.
    
    Parameters
    ----------
    theta : array_like
        Logistic regression parameters. A vector with shape (n, ). n is 
        the number of features including any intercept.  
    
    X : array_like
        The data set with shape (m x n). m is the number of examples, and
        n is the number of features (including intercept).
    
    y : array_like
        The data labels. A vector with shape (m, ).
    
    lambda_ : float
        The regularization parameter. 
    
    Returns
    -------
    J : float
        The computed value for the regularized cost function. 
    
    grad : array_like
        Một vectơ có dạng (n,) là gradient của hàm chi phí đối với theta, tại các giá trị hiện tại của theta.
    
    Instructions
    ------------
    Tính toán chi phí của một lựa chọn cụ thể của theta. Bạn nên đặt J thành chi phí.
    Tính các đạo hàm riêng và đặt grad thành đạo hàm riêng của chi phí w.r.t. mỗi tham số trong theta
    
    Hint 1
    ------
    Việc tính toán hàm chi phí và độ dốc có thể được vector hóa một cách hiệu quả. Ví dụ, hãy xem xét tính toán
    
        sigmoid(X * theta)
    
    Mỗi hàng của ma trận kết quả sẽ chứa giá trị của dự đoán cho ví dụ đó. 
    Bạn có thể tận dụng điều này để vectơ hóa hàm chi phí và tính toán độ dốc.
    Hint 2
    ------
    Khi tính toán độ dốc của hàm chi phí chính quy, 
    có nhiều giải pháp được vector hóa, nhưng một giải pháp trông giống như sau:
    
        grad = (unregularized gradient for logistic regression)
        temp = theta 
        temp[0] = 0   # because we don't add anything for j = 0
        grad = grad + YOUR_CODE_HERE (using the temp variable)
    
    Hint 3
    ------
    Chúng tôi đã cung cấp triển khai hàm sigmoid trong tệp `utils.py`. 
    Ở đầu sổ ghi chép, chúng tôi đã nhập tệp này dưới dạng mô-đun. 
    Do đó, để truy cập hàm sigmoid trong tệp đó, bạn có thể thực hiện như sau: `utils.sigmoid (z)`.
    
    """
    #Initialize some useful values
    m = y.size
    n = theta.size
    
    # convert labels to ints if their type is bool
    if y.dtype == bool:
        y = y.astype(int)
    
    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)
    
    # ====================== YOUR CODE HERE ======================

    # Compute h
    h = utils.sigmoid(np.dot(X, theta))
    
    # Compute cost J
    J = (-1/m) * (np.dot(y.T, np.log(h)) + np.dot((1-y).T, np.log(1-h))) + (lambda_/(2*m)) * (np.dot(theta[1:].T, theta[1:]))

    # Compute grad for j = 0
    grad[0] = (1/m) * (np.dot(X[:, 0].T, h-y))
    
    # Compute grad for j >= 1
    for i in range(1, n):
        grad[i] = (1/m) * (np.dot(X[:, i].T, h-y)) + (lambda_/m) * (theta[i])
        
    # =============================================================
    return J, grad

J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)

print('Cost         : {:.6f}'.format(J))
print('Expected cost: 2.534819')
print('-----------------------')
print('Gradients:')
print(' [{:.6f}, {:.6f}, {:.6f}, {:.6f}]'.format(*grad))
print('Expected gradients:')
print(' [0.146561, -0.548558, 0.724722, 1.398003]')

def oneVsAll(X, y, num_labels, lambda_):
    """
    Đào tạo các bộ phân loại hồi quy logistic num_labels và trả về từng bộ phân loại này trong ma trận all_theta, 
    trong đó hàng thứ i của all_theta tương ứng với bộ phân loại cho nhãn i.
    
    Parameters
    ----------
    X : array_like
        Tập dữ liệu đầu vào của hình dạng (m x n). m là số điểm dữ liệu, và n là số lượng tính năng. 
        Lưu ý rằng chúng tôi không giả định rằng thuật ngữ chặn (hoặc thiên vị) nằm trong X, 
        tuy nhiên, chúng tôi cung cấp mã bên dưới để thêm thuật ngữ thiên vị vào X. 
    
    y : array_like
        Các nhãn dữ liệu. Một vectơ có dạng (m,).
    
    num_labels : int
        Số lượng nhãn có thể có.
    
    lambda_ : float
        Tham số quy định hậu cần.
    
    Returns
    -------
    all_theta : array_like
        Các tham số được đào tạo cho hồi quy logistic cho mỗi lớp. 
        Đây là một ma trận có dạng (K x n + 1) 
        trong đó K là số lớp (tức là `số nhãn`) và n là số đối tượng không có sai lệch.
    
    Instructions
    ------------
    Bạn nên hoàn thành đoạn mã sau để đào tạo bộ phân loại hồi quy logistic `num_labels` 
    với tham số chính quy hóa` lambda_`.
    
    Hint
    ----
    Bạn có thể sử dụng y == c để lấy một vectơ gồm 1 và 0 
    cho bạn biết sự thật cơ bản là đúng / sai cho lớp này.
    
    Note
    ----
    Đối với nhiệm vụ này, chúng tôi khuyên bạn nên sử dụng `scipy.optimize.minimize (method = 'CG') '
    để tối ưu hóa hàm chi phí. 
    Có thể sử dụng vòng lặp for (`for c in range (num_labels):`) để lặp qua các lớp khác nhau.
    
    Example Code
    ------------
    
        # Set Initial theta
        initial_theta = np.zeros(n + 1)
      
        # Set options for minimize
        options = {'maxiter': 50}
    
        # Chạy minimize để có được theta tối ưu.
        # Hàm này sẽ trả về một đối tượng lớp trong đó theta nằm trong `res.x` và cost trong` res.fun`
        res = optimize.minimize(lrCostFunction, 
                                initial_theta, 
                                (X, (y == c), lambda_), 
                                jac=True, 
                                method='TNC',
                                options=options) 
    """
    # Some useful variables
    m, n = X.shape
    
    # You need to return the following variables correctly 
    all_theta = np.zeros((num_labels, n + 1))

    # Add ones to the X data matrix
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    # ====================== YOUR CODE HERE ======================
   
    for c in range(num_labels):
        initial_theta = np.zeros(n+1)
        options = {'maxiter': 50}
        res = optimize.minimize(lrCostFunction,
                               initial_theta,
                               (X, (y==c), lambda_),
                                jac=True,
                                method='CG',
                                options=options)
        
        all_theta[c] = res.x
    # ============================================================
    return all_theta

lambda_ = 0.1
all_theta = oneVsAll(X, y, num_labels, lambda_)

def predictOneVsAll(all_theta, X):
    """
    Trả về một vectơ dự đoán cho mỗi ví dụ trong ma trận X. Lưu ý rằng X chứa các ví dụ theo hàng. 
    all_theta là một ma trận trong đó hàng thứ i là một vector hồi quy logistic được huấn luyện theta cho lớp thứ i. 
    Bạn nên đặt p thành vectơ có giá trị từ 0..K-1 (ví dụ: p = [0, 2, 0, 1] dự đoán các lớp 0, 2, 0, 1 cho 4 ví dụ).
    
    Parameters
    ----------
    all_theta : array_like
        Các tham số được đào tạo cho hồi quy logistic cho mỗi lớp. 
        Đây là một ma trận có dạng (K x n + 1) trong đó K là số lớp và n là số đối tượng không có sai lệch.
    
    X : array_like
        Điểm dữ liệu để dự đoán nhãn của chúng. Đây là một ma trận có dạng (m x n) 
        trong đó m là số điểm dữ liệu cần dự đoán và n là số đối tượng không có số hạng thiên vị. 
        Lưu ý rằng chúng tôi thêm thuật ngữ thiên vị cho X trong hàm này.
    
    Returns
    -------
    p : array_like
        Các dự đoán cho mỗi điểm dữ liệu trong X. Đây là một vectơ có dạng (m,).
    
    Instructions
    ------------
    Hoàn thành đoạn mã sau để đưa ra dự đoán bằng cách sử dụng các tham số hồi quy logistic đã học của bạn 
    (một so với tất cả). 
    Bạn nên đặt p thành vectơ dự đoán (từ 0 đến num_labels-1).
    
    Hint
    ----
    Mã này có thể được vector hóa tất cả bằng cách sử dụng hàm argmax numpy. 
    Đặc biệt, hàm argmax trả về chỉ số của phần tử max, để biết thêm thông tin, 
    hãy xem '? Np.argmax' hoặc tìm kiếm trực tuyến. 
    Nếu các ví dụ của bạn nằm trong các hàng, thì bạn có thể sử dụng np.argmax (A, axis = 1) 
    để lấy chỉ số của giá trị lớn nhất cho mỗi hàng.
    """
    m = X.shape[0]
    num_labels = all_theta.shape[0]

    # You need to return the following variables correctly 
    p = np.zeros(m)

    # Add ones to the X data matrix
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    # ====================== YOUR CODE HERE ======================

    Z = np.dot(X, all_theta.T)
    for i in range(m):
        p[i] = np.argmax(Z[i], axis = 0)
    
    # ============================================================
    return p

pred = predictOneVsAll(all_theta, X)
print('Training Set Accuracy: {:.2f}%'.format(np.mean(pred == y) * 100))