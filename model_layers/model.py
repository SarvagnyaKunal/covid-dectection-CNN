import numpy as np
from conv_layer import Conv2D
from relu_layer import ReLULayer
from maxpool_layer import MaxPoolLayer
from fc_layer import FullyConnectedLayer
from dropout_layer import DropoutLayer

class CNNModel:
    def __init__(self):
        self.conv1 = Conv2D(input_channels=1, output_channels=32, kernel_size=3, stride=1, padding=0)
        self.pool = MaxPoolLayer(pool_size=2, stride=2)
        self.dropout = DropoutLayer(dropout_rate=0.25)
        
        self.conv2 = Conv2D(input_channels=32, output_channels=64, kernel_size=3, stride=1, padding=0)
        self.conv3 = Conv2D(input_channels=64, output_channels=128, kernel_size=3, stride=1, padding=0)
        
        self.fc1 = FullyConnectedLayer(input_size=128*26*26, output_size=64)
        self.fc2 = FullyConnectedLayer(input_size=64, output_size=1)

    def forward(self, x, training=True):
        x = self.pool.forward(self.relu1.forward(self.conv1.conv_forward(x)))
        x = self.dropout.forward(x, training)
        
        x = self.pool.forward(self.relu2.forward(self.conv2.conv_forward(x)))
        x = self.dropout.forward(x, training)
        
        x = self.pool.forward(self.relu3.forward(self.conv3.conv_forward(x)))
        x = self.dropout.forward(x, training)
        
        x = x.reshape(x.shape[0], -1)  # Flatten the output for the fully connected layer
        x = self.dropout.forward(self.relu4.forward(self.fc1.fc_forward(x)), training)
        x = self.fc2.fc_forward(x)  # No activation function here
        return x

    def backward(self, d_loss, learning_rate):
        d_loss = self.fc2.backward(d_loss, learning_rate)
        d_loss = self.dropout.backward(d_loss)
        d_loss = self.relu4.backward(d_loss)
        d_loss = self.fc1.backward(d_loss, learning_rate)
        d_loss = d_loss.reshape(-1, 128, 26, 26)  # Reshape back to the shape before flattening
        
        d_loss = self.dropout.backward(d_loss)
        d_loss = self.pool.backward(d_loss)
        d_loss = self.relu3.backward(d_loss)
        d_loss = self.conv3.conv_backward(d_loss, learning_rate)
        
        d_loss = self.dropout.backward(d_loss)
        d_loss = self.pool.backward(d_loss)
        d_loss = self.relu2.backward(d_loss)
        d_loss = self.conv2.conv_backward(d_loss, learning_rate)
        
        d_loss = self.dropout.backward(d_loss)
        d_loss = self.pool.backward(d_loss)
        d_loss = self.relu1.backward(d_loss)
        d_loss = self.conv1.conv_backward(d_loss, learning_rate)
        
        return d_loss