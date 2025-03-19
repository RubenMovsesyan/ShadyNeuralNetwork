import numpy as np
import idx2numpy
import matplotlib.pyplot as plt

image_file = "../test_files/train-images-idx3-ubyte"
image_array = idx2numpy.convert_from_file(image_file)

# print(image_array[4])
# image_array[4].astype('int8').tofile('../test_files/image_array_4')
# plt.imshow(image_array[4], cmap=plt.cm.binary)
# plt.show()

def print_progress(curr, total):
    size = 50
    print("\r[", end="")
    ratio = curr / total
    for i in range(int(50 * ratio)):
        print("=", end="")
    for i in range(int(50 - (50 * ratio))):
        print(" ", end="")
    print("]", end="")

for i in range(len(image_array)):
    print_progress(i, len(image_array))
    image_array[i].astype("int8").tofile("../test_files/train_images/image_array_%d" % i)
