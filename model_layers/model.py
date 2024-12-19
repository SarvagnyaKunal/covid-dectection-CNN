import numpy as np
from conv_layer import Conv2D
from relu_layer import ReLULayer
from maxpool_layer import MaxPoolLayer
from fc_layer import FullyConnectedLayer
from dropout_layer import DropoutLayer

class CNNModel:
    def __init__(self):
        self.conv1 = Conv2D(input_channels=3, output_channels=32, kernel_size=3, stride=1, padding=0)
        self.relu1 = ReLULayer()
        self.pool1 = MaxPoolLayer(pool_size=2, stride=2)
        self.dropout1 = DropoutLayer(dropout_rate=0.25)
        
        self.conv2 = Conv2D(input_channels=32, output_channels=64, kernel_size=3, stride=1, padding=0)
        self.relu2 = ReLULayer()
        self.pool2 = MaxPoolLayer(pool_size=2, stride=2)
        self.dropout2 = DropoutLayer(dropout_rate=0.25)
        
        self.fc1 = FullyConnectedLayer(input_size=64*52*52, output_size=10)

    def forward(self, x, training=True):
        x = self.conv1.conv_forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        x = self.dropout1.forward(x, training)
        
        x = self.conv2.conv_forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)
        x = self.dropout2.forward(x, training)
        
        x = x.reshape(x.shape[0], -1)  # Flatten the output for the fully connected layer
        x = self.fc1.fc_forward(x)
        return x

    def backward(self, d_loss, learning_rate):
        d_loss = self.fc1.backward(d_loss, learning_rate)
        d_loss = d_loss.reshape(-1, 64, 52, 52)  # Reshape back to the shape before flattening
        
        # Backward pass through second dropout layer
        d_loss = self.dropout2.backward(d_loss)
        # Backward pass through second conv layer
        d_loss = self.pool2.backward(d_loss)
        d_loss = self.relu2.backward(d_loss)
        d_loss = self.conv2.conv_backward(d_loss, learning_rate)
        
        # Backward pass through first dropout layer
        d_loss = self.dropout1.backward(d_loss)
        # Backward pass through first conv layer
        d_loss = self.pool1.backward(d_loss)
        d_loss = self.relu1.backward(d_loss)
        d_loss = self.conv1.conv_backward(d_loss, learning_rate)
        
        return d_loss