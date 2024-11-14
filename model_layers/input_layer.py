# do not use this code
# its a direct pytorch code

'''
import torch
import torch.nn as nn
from dataset_prep import training_xray_prep 

class CNNModel(nn.Module):
    def __init__(self, num_classes=3):
        super(CNNModel, self).__init__()
        
        # Define convolutional layers with ReLU and max-pooling
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 64 * 64, 512) 
        self.fc2 = nn.Linear(512, num_classes)  
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass through first convolutional layer, then ReLU and pooling
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        
        # Pass through second convolutional layer, then ReLU and pooling
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        # Flatten for the fully connected layers
        x = x.view(-1, 64 * 64 * 64)
        
        # Fully connected layers with ReLU and output layer
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
'''