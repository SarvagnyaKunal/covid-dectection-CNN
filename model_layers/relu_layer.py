import numpy as np

def relu(input_array):
    """Applies ReLU activation to the input array."""
    return np.maximum(0, input_array)

# Test the ReLU function
if __name__ == "__main__":
    # Example input: random data between 0 and 1
    input_data = np.random.rand(2, 3)  # Generates a 2x3 array with values between 0 and 1
    print("Input Data:")
    print(input_data)

    # Apply ReLU
    output_data = relu(input_data)
    
    print("\nOutput Data after ReLU:")
    print(output_data.shape)
