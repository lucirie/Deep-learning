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
    
    def forward(self, input):
        self.input = input # for back_prop
        self.output = np.copy(self.biases)

        for i in range(self.filters):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], self.padding)
        
        return self.output
    
    def backwards(self, output_gradient, learning_rate):
        input_gradients = np.zeros_like(self.input)
        kernels_gradients = np.zeros_like(self.kernels)
        biases_gradients = np.zeros_like(self.biases)

        for i in range(self.filters):
            for j in range(self.input_depth):
                kernels_gradients[i][j] = signal.correlate2d(self.input[j], output_gradient[i], mode='valid')
                input_gradients[j] += signal.convolve2d(output_gradient[j], self.kernels[i, j])
                biases_gradients = output_gradient
        self.kernels -= learning_rate * kernels_gradients
        self.biases -= learning_rate * biases_gradients
        return input_gradients
    
class Reshape:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)
    
    def backwards(self, output_gradient):
        return np.reshape(output_gradient, self.input_shape)
    
def max_pooling(input, pool_size=2, stride=2):
    C, H, W = input.shape
    out_H, out_W = (H - pool_size) // stride + 1, (W - pool_size) // stride + 1
    output = np.zeros((C, out_H, out_W))
    for c in range(C):
        for i in range(out_H):
            for j in range(out_W):
                h_start, w_start = i * stride, j * stride
                h_end, w_end = h_start + pool_size, w_start + pool_size
                output[c, i, j] = np.max(input[c, h_start:h_end, w_start:w_end])
    return output