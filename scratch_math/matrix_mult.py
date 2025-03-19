import numpy as np
import math

def data(x, y):
    return [x * x, y * y / 2]


def relu(x):
    return x * (x > 0)


inputs = np.array([6.2103214, -4.2103214])

weights_1 = np.array([
    [0.24915643, 3.1761217],
    [2.419036, -3.0791311],
    [-2.7685971, -0.56488425],
])

result = weights_1.dot(inputs)

# print("Inputs with first weights")
# print(result)

biases_1 = np.array(
    [
        0.4186293 * 0.3021832,
        0.07311487 * -0.029371858,
        -0.54608464 * 0.41828114
    ]
)

# print("First Layer with biases")
output = result + biases_1

# print(output)

# print("Frist Layer after activation")

r = relu(output)
# print(r)


weights_2 = np.array([
    [-0.31085804, 0.4052628, -0.2410703],
    [-72.95311, -8.266496, 0.4710372],
])

biases_2 = np.array(
    [
        0.6315368 * 0.3378554,
        -0.3496796 * -0.2242713,
    ]
)

result = weights_2.dot(r)

# print("Second Layer with weights")
# print(result)

# print("Second Layer with biases")
output = result + biases_2

# print(output)

# print("Second Layer after softmax")

output = np.array(data(0.1, 0.5))

max_val = np.max(output)
exp_sum = 0

exp_out = []

for elem in output:
    exp_out.append(math.exp(elem - max_val))

print(max_val)
print(exp_out)

for elem in exp_out:
    exp_sum += elem

for i in range(len(exp_out)):
    exp_out[i] /= exp_sum

print(exp_out)
