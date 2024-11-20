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

    def forward(self, x):
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
