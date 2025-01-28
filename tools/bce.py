import numpy as np

def binary_cross_entropy_loss(predictions, targets):
    epsilon = 1e-12
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    return -np.mean(np.sum(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions), axis=1))

def binary_cross_entropy_loss_derivative(predictions, targets):
    epsilon = 1e-12
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    return (predictions - targets) / (predictions * (1 - predictions))
