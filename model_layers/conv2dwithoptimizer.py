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
        self.input = x
        # Convolution operation (not fully implemented here)
        pass

    def conv_backward(self, d_output):
        # Compute gradients for weights and biases (not fully implemented here)
        self.d_weights = ...
        self.d_biases = ...
        
        # Compute gradient for the input (to propagate backward)
        d_input = ...
        
        return d_input

    def get_params_and_grads(self):
        return [(self.weights, self.d_weights), (self.biases, self.d_biases)]