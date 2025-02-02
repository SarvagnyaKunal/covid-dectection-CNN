import numpy as np

class ReLULayer:
    def forward(self, input_array):
        self.input = input_array
        return np.maximum(0, input_array)
    
    def backward(self, d_output):
        return d_output * (self.input > 0)