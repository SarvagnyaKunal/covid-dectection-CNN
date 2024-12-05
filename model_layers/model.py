import numpy as np
from conv_layer import Conv2D
from relu_layer import ReLULayer
from maxpool_layer import MaxPoolLayer
from fc_layer import FullyConnectedLayer

class CNNModel:
    def __init__(self):
        self.conv1 = Conv2D(input_channels=1, output_channels=8, kernel_size=3, stride=1, padding=1)
        self.relu1 = ReLULayer()
        self.pool1 = MaxPoolLayer(pool_size=2, stride=2)
        
        self.conv2 = Conv2D(input_channels=8, output_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu2 = ReLULayer()
        self.pool2 = MaxPoolLayer(pool_size=2, stride=2)
        
        self.fc1 = FullyConnectedLayer(input_size=16*61*61, output_size=10)

    def forward(self, x):
        x = self.conv1.conv_forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        
        x = self.conv2.conv_forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)
        
        x = x.reshape(x.shape[0], -1)  # Flatten the output for the fully connected layer
        x = self.fc1.fc_forward(x)
        return x

    def backward(self, d_loss, learning_rate):
        d_loss = self.fc1.backward(d_loss, learning_rate)
        d_loss = d_loss.reshape(-1, 16, 61, 61)  # Reshape back to the shape before flattening
        
        # Backward pass through second conv layer
        # ...existing code...
        pass