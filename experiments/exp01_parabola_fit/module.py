# 1. Standard Library
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

# 2. Third-party
import numpy as np

# 3. Local Project Modules
from src.layers import LinearLayer, ReLU
from src.losses import MSELoss
from src.network import NeuralNetwork

# 实验主体
class parabola_fit:
    def __init__(self, 
                 true_data_noise_ratio = 0.01, 
                 dot_number = 101, 
                 epochs = 10000, 
                 learning_rate = 0.002, 
                 layers = ["LinearLayer(1, 100)", "ReLU()", "LinearLayer(100, 10)", "LinearLayer(10, 1)"]):
        self.true_data_noise_ratio = true_data_noise_ratio
        self.dot_number = dot_number
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.layers = [eval(i) for i in layers]
        self.epoch_x = None
        self.y_final_pred = None
        self.final_loss = None
        self.loss_history = None

    def run_experiment(self):
        # 数据与网络准备
        x_data = np.linspace(-1,1,self.dot_number).reshape(-1,1) # 把行向量变为列向量，batch=100（同时观察100个粒子），input=1（每个粒子有1个自由度需要观测）
        y_ideal = x_data ** 2
        noise = self.true_data_noise_ratio * np.random.randn(self.dot_number,1)
        y_true = y_ideal + noise
        my_net = NeuralNetwork(self.layers)
        my_loss = MSELoss()
        loss_history = []

        # 开始训练
        for i in range(self.epochs):
            y_pred = my_net.forward(x_data)
            loss = my_loss.forward(y_pred, y_true)
            grad_from_loss = my_loss.backward()
            my_net.backward(grad_from_loss)
            my_net.update(self.learning_rate)
            if i%100 == 0:
                loss_history.append(loss)

        # 最终结果
        self.x_data = x_data
        self.y_ideal = y_ideal
        self.y_true = y_true
        self.y_final_pred = my_net.forward(x_data)
        self.final_loss = my_loss.forward(y_pred, y_true)
        loss_history.append(self.final_loss)
        self.loss_history = loss_history
        self.epoch_x = np.linspace(0,self.epochs,len(loss_history))
        self.log_loss = np.log10(self.final_loss)
        

