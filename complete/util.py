import os
import csv
import numpy as np


MNIST = "large_files/train.csv"


def get_mnist():
    current_folder = os.path.dirname(os.path.abspath(__file__))
    wd = os.path.abspath(os.path.join(current_folder, os.pardir))
    data_path = os.path.join(wd, MNIST)
    csv_file = open(data_path, 'r')
    csv_pointer = csv.reader(csv_file)
    list_mnist = []
    for row in csv_pointer:
        list_mnist.append(np.asarray(row))
    list_mnist = np.asarray(list_mnist[1:], dtype=int)
    Y = list_mnist[:,0]
    X = list_mnist[:, 1:]
    return X/255.0, Y


