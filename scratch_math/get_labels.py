import numpy as np
import idx2numpy
import matplotlib.pyplot as plt

label_file = "../test_files/train-labels-idx1-ubyte"
label_array = idx2numpy.convert_from_file(label_file)

label_array.astype('int8').tofile('../test_files/train_labels')
