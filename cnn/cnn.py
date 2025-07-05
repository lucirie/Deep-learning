import numpy as np
from scipy import signal

class Convolution:
    def __init__(self, input_shape, kernel_size, filters, padding='valid'):
        input_depth, input_height, input_width = input_shape
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.filters = filters
        self.padding = padding

        if padding == 'valid':
            output_height = input_height - kernel_size + 1
            output_width = input_width - kernel_size + 1
        elif padding == 'full':
            output_height = input_height + kernel_size - 1
            output_width = input_width + kernel_size - 1
        elif padding == 'same':
            output_height = input_height
            output_width = input_width
        else:
            raise ValueError("Incorrect padding value")
        output_shape = (filters, output_height, output_width)
        kernel_shape = (filters, input_depth, kernel_size, kernel_size)

        self.kernels = np.random.randn(*kernel_shape)
        self.biases = np.random.randn(*output_shape)