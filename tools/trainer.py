import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model_layers')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset_prep')))

import numpy as np
from model import CNNModel
from training_xray_prep import load_all_data

def binary_cross_entropy_loss(predictions, targets):
    epsilon = 1e-12
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))

def binary_cross_entropy_loss_derivative(predictions, targets):
    epsilon = 1e-12
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    return (predictions - targets) / (predictions * (1 - predictions))

def train(model, train_images, train_labels, learning_rate, epochs):
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(train_images)):
            x = train_images[i]
            y = train_labels[i]
            
            # Forward pass
            predictions = model.forward(x, training=True)
            
            # Compute loss
            loss = binary_cross_entropy_loss(predictions, y)
            total_loss += loss
            
            # Backward pass
            d_loss = binary_cross_entropy_loss_derivative(predictions, y)
            model.backward(d_loss, learning_rate)
        
        avg_loss = total_loss / len(train_images)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")

if __name__ == "__main__":
    base_folder = "c:/Users/sarva/Desktop/nf/code/covid-dectection-CNN/dataset_container/train_data/Infection Segmentation Data"
    
    # Load data for test, train, and validation sets
    test_images, train_images, val_images = load_all_data(base_folder, max_images=200)
    
    # Create labels for the training data (assuming binary classification)    train_labels = np.array([0] * len(train_images[0]) + [1] * len(train_images[1]) + [2] * len(train_images[2]))
    9
    # Combine all training images into a single array and add a channel dimension
    train_images = np.concatenate(train_images, axis=0)
    train_images = train_images[:, np.newaxis, :, :]  # Add channel dimension
    
    # Initialize the model
    model = CNNModel()
    
    # Train the model
    train(model, train_images, train_labels, learning_rate=0.001, epochs=10)
