class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for unit in self.layers:
            x = unit.forward(x)
        return x

    def backward(self, grad):
        for unit in self.layers[::-1]:
            grad = unit.backward(grad)
        return grad
    
    def update(self, learning_rate):
        for unit in self.layers:
            unit.update(learning_rate)