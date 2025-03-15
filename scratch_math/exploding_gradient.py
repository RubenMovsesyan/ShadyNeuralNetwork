# import numpy as np
# import math


# # matrix = np.random.normal(size = (4, 4))

# # for i in range(100):
# #     matrix = np.dot(matrix, np.random.normal(size = (4, 4)))


# # print(matrix)

# n = 16


# lower, upper = -(1.0 / math.sqrt(n)), (1.0 / math.sqrt(n))
# numbers = np.random.rand(1000)

# scaled = lower + numbers * (upper - lower)

# print(lower, upper)
# print(scaled.min(), scaled.max())
# print(scaled.mean(), scaled.std())
# print(scaled)

from math import sqrt
import matplotlib.pyplot as plt
# define the number of inputs from 1 to 100
values = [i for i in range(1, 101)]
# calculate the range for each number of inputs
results = [1.0 / sqrt(n) for n in values]
# create an error bar plot centered on 0 for each number of inputs
plt.errorbar(values, [0.0 for _ in values], yerr=results)
plt.show()
