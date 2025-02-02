
import numpy as np

class DropoutLayer:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate

    def forward(self, x, training=True):
        if training:
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=x.shape) / (1 - self.dropout_rate)
            return x * self.mask
        else:
            return x

    def backward(self, d_output):
        return d_output * self.mask