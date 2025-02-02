import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#sigmoid derivative for backpropagation
def sigmoid_derivative(x):
    return x * (1 - x)

class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.biases = np.random.rand(1, output_size) - 0.5

    def fc_forward(self, input_data):
        self.input = input_data
        self.z = np.dot(self.input, self.weights) + self.biases
        self.output = sigmoid(self.z)
        return self.output

    def backward(self, d_loss, learning_rate):
        d_sigmoid = sigmoid_derivative(self.output)
        d_output = d_loss * d_sigmoid
        
        # Gradients for weights and biases
        d_weights = np.dot(self.input.T, d_output)
        d_biases = np.sum(d_output, axis=0, keepdims=True)
        
        # Gradients for the input (to propagate backward)
        d_input = np.dot(d_output, self.weights.T)
        
        # Update weights and biases
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases
        
        return d_input