import numpy as np

def max_pooling(input_array, pool_size=2, stride=2):
    
    if input_array.ndim == 2:
        input_array = input_array[np.newaxis, ...] 
    
    C, H, W = input_array.shape 
    out_height = (H - pool_size) // stride + 1
    out_width = (W - pool_size) // stride + 1
    
    pooled_output = np.zeros((C, out_height, out_width))
    
    for c in range(C):
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * stride
                h_end = h_start + pool_size
                w_start = j * stride
                w_end = w_start + pool_size
                
                pooled_output[c, i, j] = np.max(input_array[c, h_start:h_end, w_start:w_end])
    
    return pooled_output.squeeze()
