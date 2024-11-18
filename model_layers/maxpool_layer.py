import numpy as np

def max_pooling(input_array, pool_size=2, stride=2):
    """
    Applies 2D max pooling to the input array.

    Parameters:
    - input_array: numpy array of shape (H, W) for a single channel or (C, H, W) for multiple channels.
    - pool_size: Size of the pooling window (e.g., 2 for a 2x2 window).
    - stride: Stride (step size) of the pooling window.

    Returns:
    - Pooled output as a numpy array.
    """
    
    if input_array.ndim == 2:  # Single channel
        input_array = input_array[np.newaxis, ...]  # Add channel dimension
    
    C, H, W = input_array.shape  # Channels, Height, Width
    out_height = (H - pool_size) // stride + 1
    out_width = (W - pool_size) // stride + 1
    
    # Initialize output array
    pooled_output = np.zeros((C, out_height, out_width))
    
    for c in range(C):  # Loop over each channel
        for i in range(out_height):
            for j in range(out_width):
                # Find the start and end of the current pooling window
                h_start = i * stride
                h_end = h_start + pool_size
                w_start = j * stride
                w_end = w_start + pool_size
                
                # Apply max pooling on the current window
                pooled_output[c, i, j] = np.max(input_array[c, h_start:h_end, w_start:w_end])
    
    return pooled_output.squeeze()  # Remove single-dimensional entries for single-channel output

# Test the MaxPooling function
if __name__ == "__main__":
    # Example input (random 4x4 data)
    input_data = np.array([[1, 2, 3, 0],
                           [4, 6, 1, 2],
                           [7, 8, 9, 5],
                           [2, 1, 4, 3]])

    print("Input Data:")
    print(input_data)

    # Apply Max Pooling
    pooled_data = max_pooling(input_data, pool_size=2, stride=2)
    
    print("\nPooled Data:")
    print(pooled_data)
