import numpy as np

class Conv2D:
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights and biases
        self.weights = np.random.randn(output_channels, input_channels, kernel_size, kernel_size)
        self.biases = np.random.randn(output_channels)

    def conv_forward(self, x):
        self.input = x  # Store input for backward pass
        #padding
        x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        
        #output dimensions
        batch_size, in_channels, input_height, input_width = x.shape
        out_height = (input_height - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_width = (input_width - self.kernel_size + 2 * self.padding) // self.stride + 1
        
        output = np.zeros((batch_size, self.output_channels, out_height, out_width))

        #Convolution
        for b in range(batch_size):
            for o in range(self.output_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        # Get the current region for the convolution
                        region = x_padded[b, :, i * self.stride : i * self.stride + self.kernel_size, j * self.stride : j * self.stride + self.kernel_size]
                        # Perform element-wise multiplication and sum with the filter, add bias
                        output[b, o, i, j] = np.sum(region * self.weights[o]) + self.biases[o]
        
        return output

    def conv_backward(self, d_output, learning_rate):
        batch_size, _, out_height, out_width = d_output.shape
        d_input = np.zeros_like(self.input)
        d_weights = np.zeros_like(self.weights)
        d_biases = np.zeros_like(self.biases)
        
        x_padded = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        d_x_padded = np.pad(d_input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        
        for b in range(batch_size):
            for o in range(self.output_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size
                        
                        region = x_padded[b, :, h_start:h_end, w_start:w_end]
                        d_weights[o] += region * d_output[b, o, i, j]
                        d_biases[o] += d_output[b, o, i, j]
                        d_x_padded[b, :, h_start:h_end, w_start:w_end] += self.weights[o] * d_output[b, o, i, j]
        
        if self.padding != 0:
            d_input = d_x_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            d_input = d_x_padded
        
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases
        
        return d_input

