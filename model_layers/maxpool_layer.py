import numpy as np

class MaxPoolLayer:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input_array):
        self.input = input_array
        if input_array.ndim == 2:
            input_array = input_array[np.newaxis, ...] 
        
        C, H, W = input_array.shape 
        out_height = (H - self.pool_size) // self.stride + 1
        out_width = (W - self.pool_size) // self.stride + 1
        
        pooled_output = np.zeros((C, out_height, out_width))
        
        for c in range(C):
            for i in range(out_height):
                for j in range(out_width):
                    h_start = i * self.stride
                    h_end = h_start + self.pool_size
                    w_start = j * self.stride
                    w_end = w_start + self.pool_size
                    
                    pooled_output[c, i, j] = np.max(input_array[c, h_start:h_end, w_start:w_end])
        
        return pooled_output.squeeze()
    
    def backward(self, d_output):
        d_input = np.zeros_like(self.input)
        C, H, W = self.input.shape
        out_height = (H - self.pool_size) // self.stride + 1
        out_width = (W - self.pool_size) // self.stride + 1
        
        for c in range(C):
            for i in range(out_height):
                for j in range(out_width):
                    h_start = i * self.stride
                    h_end = h_start + self.pool_size
                    w_start = j * self.stride
                    w_end = w_start + self.pool_size
                    
                    region = self.input[c, h_start:h_end, w_start:w_end]
                    max_val = np.max(region)
                    d_input[c, h_start:h_end, w_start:w_end] += (region == max_val) * d_output[c, i, j]
        
        return d_input
