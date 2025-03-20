import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



def init_params():
    # Our data has 784 pixels
    # our hidded layer has 10 outputs
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5

    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def d_ReLU(Z):
    return Z > 0

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, x):
    Z1 = W1.dot(x) + b1
    A1 = ReLU(Z1)

    Z2 = W2.dot(A1) + b1
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2

def one_hot(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y

def back_prop(Z1, A1, Z2, A2, W2, x, y):
    # m = y.size
    one_hot_y = one_hot(y)
    print('one_hot_y shape', one_hot_y.shape)
    dZ2 = A2 - one_hot_y
    print('dZ2 shape', dZ2.shape)
    print('A1 shape', A1.T.shape)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)

    dZ1 = W2.T.dot(dZ2) * d_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(x.T)
    db1 = 1 / m * np.sum(dZ1)

    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1

    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, y):
    print(predictions, y)
    return np.sum(predictions == y) / y.size

def gradient_descent(x, y, iterations, alpha):
    accuracy_array = []
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, x)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, x, y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print('Iteration: ', i)
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, y)
            print('Accuracy: ', accuracy)
            accuracy_array.append(accuracy)
    return W1, b1, W2, b2, accuracy_array

# Read the training data
data = np.array(pd.read_csv('../test_files/mnist_train.csv'))

# Get the dimensions of the data
m, n = data.shape
print(m, n)
np.random.shuffle(data)

# Testing data
data_dev = data[0:1000].T
label_dev = data_dev[0]
image_dev = data_dev[1:n]

# Training data
data_train = data[1000:m].T
label_train = data_train[0]
image_train = data_train[1:n]
image_train = image_train / 255.

W1, b1, W2, b2, accuracy_array = gradient_descent(image_train, label_train, 500, 0.1)

plt.plot(accuracy_array)
plt.show()
