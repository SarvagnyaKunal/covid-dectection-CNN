
import numpy as np
from conv_layer import Conv2D
from relu_layer import relu
from maxpool_layer import max_pooling
from fc_layer import FullyConnectedLayer

class CNNModel:
    def __init__(self):
        self.conv1 = Conv2D(input_channels=1, output_channels=8, kernel_size=3, stride=1, padding=1)
        self.relu1 = relu
        self.pool1 = max_pooling
        self.fc1 = FullyConnectedLayer(input_size=8*14*14, output_size=10)

    def forward(self, x):
        x = self.conv1.conv_forward(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = x.reshape(x.shape[0], -1)  # Flatten the output for the fully connected layer
        x = self.fc1.fc_forward(x)
        return x

    def backward(self, d_loss, learning_rate):
        d_loss = self.fc1.backward(d_loss, learning_rate)
        d_loss = d_loss.reshape(-1, 8, 14, 14)  # Reshape back to the shape before flattening
        # Implement backward pass for pooling and convolution if needed
        pass