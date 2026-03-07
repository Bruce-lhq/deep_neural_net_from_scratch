import numpy as np
class LinearLayer:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = np.random.randn(input_dim, output_dim)
        self.b = np.zeros(output_dim)
        self.x = None
        self.grad_W = None
        self.grad_b = None

    def forward(self, x):
        self.x = x
        z = x @ self.W + self.b
        return z
    
    def backward(self, grad_z):
        grad_x = grad_z @ self.W.T
        self.grad_W = self.x.T @ grad_z
        self.grad_b = np.sum(grad_z, axis=0)
        return grad_x
    
    def update(self, learning_rate):
        self.W -= learning_rate * self.grad_W
        self.b -= learning_rate * self.grad_b

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, z):
        self.mask = (z>0)
        return z * self.mask
    
    def backward(self, input_grad):
        return input_grad * self.mask
    
    # 为了方法名称一致以便于多态调用，加入update方法
    def update(self, learning_rate):
        pass
