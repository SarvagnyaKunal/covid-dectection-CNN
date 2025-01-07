import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from model import CNNModel
from training_xray_prep import load_all_data
from optimizer import SGD

def cross_entropy_loss(predictions, targets):
    """Compute the cross-entropy loss."""
    loss = -np.sum(targets * np.log(predictions + 1e-9)) / targets.shape[0]
    return loss

def compute_accuracy(predictions, targets):
    """Compute the accuracy of the predictions."""
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(targets, axis=1)
    accuracy = np.mean(pred_labels == true_labels)
    return accuracy

def train(model, train_images, train_labels, val_images, val_labels, epochs=10, learning_rate=0.001):
    """Train the CNN model."""
    optimizer = SGD(learning_rate=learning_rate)
    
    for epoch in range(epochs):
        # Forward pass
        predictions = model.forward(train_images, training=True)
        loss = cross_entropy_loss(predictions, train_labels)
        accuracy = compute_accuracy(predictions, train_labels)
        
        # Backward pass
        d_loss = predictions - train_labels
        model.backward(d_loss)
        
        # Update weights and biases using the optimizer
        for param, grad in model.get_params_and_grads():
            optimizer.update([param], [grad])
        
        # Validation
        val_predictions = model.forward(val_images, training=False)
        val_loss = cross_entropy_loss(val_predictions, val_labels)
        val_accuracy = compute_accuracy(val_predictions, val_labels)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

if __name__ == "__main__":
    base_folder = "c:/Users/sarva/Desktop/nf/code/covid-dectection-CNN/dataset_container/train_data/Infection Segmentation Data"
    
    # Load data for test, train, and validation sets
    test_images, train_images, val_images = load_all_data(base_folder, max_images=200)
    
    # Assuming labels are loaded or generated separately
    train_labels = np.array([...])  # Replace with actual labels
    val_labels = np.array([...])    # Replace with actual labels
    test_labels = np.array([...])   # Replace with actual labels
    
    # Initialize the model
    model = CNNModel()
    
    # Train the model
    train(model, train_images, train_labels, val_images, val_labels, epochs=10, learning_rate=0.001)
    
    # Test the model
    test_predictions = model.forward(test_images, training=False)
    test_accuracy = compute_accuracy(test_predictions, test_labels)
    print(f"Test Accuracy: {test_accuracy:.4f}")