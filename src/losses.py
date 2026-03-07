import numpy as np
class MSELoss:
    def __init__(self):
        self.y_true = None
        self.y_pred = None

    def forward(self, y_pred, y_true):
        self.delta_y = y_pred-y_true
        loss = np.mean(self.delta_y **2)
        return loss
    
    def backward(self):
        grad = 2/self.delta_y.size * self.delta_y
        return grad