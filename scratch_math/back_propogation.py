import numpy as np
import matplotlib.pyplot as plt


# Softmax function to make sure all elements of array only add up to 1
def softmax(arr):
    return np.exp(arr) / sum(np.exp(arr))


# Loss function
def binary_cross_entropy_loss(prediction, label):
    return -(label * np.log(prediction) + (1 - label) * np.log(1 - prediction))




prediction = np.array([1, 2, 3, 4, 5, 6])
label = np.array([6, 5, 4, 3, 2, 1])

print(binary_cross_entropy_loss(softmax(prediction), softmax(label)))
