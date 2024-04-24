"""
Load MNIST dataset
"""
from pathlib import Path
import numpy as np
import struct
from array import array

class MnistDataloader(object):
    def __init__(self,
                 training_images_filepath,
                 training_labels_filepath,
                 test_images_filepath,
                 test_labels_filepath):

        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath


    def read_labels(self, labels_filepath):
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, '
                                 f'expected 2049, got {magic}.')
            labels = np.array(array("B", file.read()))

        return labels


    def read_images(self, images_filepath):
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, '
                                 f'expected 2051, got {magic}.')
            image_data = array("B", file.read())

        images = np.zeros((size, rows, cols))
        for i in range(size):
            idx_start = i * rows * cols
            idx_end = idx_start + rows * cols

            img = np.array(image_data[idx_start : idx_end])

            images[i][:] = img.reshape(rows, cols)

        return images


    def load_data(self):
        x_train = self.read_images(self.training_images_filepath)
        y_train = self.read_labels(self.training_labels_filepath)
        x_test  = self.read_images(self.test_images_filepath)
        y_test  = self.read_labels(self.test_labels_filepath)

        return (x_train, y_train), (x_test, y_test)


# def main():
#     input_path = Path('./data')
#
#     training_images_filepath = input_path/'train-images-idx3-ubyte/train-images-idx3-ubyte'
#     training_labels_filepath = input_path/'train-labels-idx1-ubyte/train-labels-idx1-ubyte'
#
#     test_images_filepath = input_path/'t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
#     test_labels_filepath = input_path/'t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
#
#     mnist_dataloader = MnistDataloader(training_images_filepath,
#                                        training_labels_filepath,
#                                        test_images_filepath,
#                                        test_labels_filepath)
#
#     (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
#
#     print(f'x_train shape = {x_train.shape}')
#     print(f'y_train shape = {y_train.shape}')
#     print(f'x_test shape  = {x_test.shape}')
#     print(f'y_test shape  = {y_test.shape}')

# main()
