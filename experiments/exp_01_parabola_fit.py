# 实验 01: 拟合抛物线物理规律 (Parabola Fitting)

# 1. Standard Library
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

# 2. Third-party
import numpy as np
import matplotlib.pyplot as plt

# 3. Local Project Modules
from src.layers import LinearLayer, ReLU
from src.losses import MSELoss
from src.network import NeuralNetwork

# 实验主体
def run_experiment():
    # 可调参数
    seed=42
    true_data_noise_ratio = 0.01
    dot_number = 101
    epochs = 10000
    learning_rate = 0.001
    layers = [LinearLayer(1, 50), ReLU(), LinearLayer(50, 10), LinearLayer(10, 1)]

    # 数据与网络准备
    np.random.seed(seed)
    x_data = np.linspace(-1,1,dot_number).reshape(-1,1) # 把行向量变为列向量，batch=100（同时观察100个粒子），input=1（每个粒子有1个自由度需要观测）
    y_ideal = x_data ** 2
    noise = true_data_noise_ratio * np.random.randn(dot_number,1)
    y_true = y_ideal + noise
    my_net = NeuralNetwork(layers)
    my_loss = MSELoss()

    # 开始训练
    for _ in range(epochs):
        y_pred = my_net.forward(x_data)
        loss = my_loss.forward(y_pred, y_true)
        grad_from_loss = my_loss.backward()
        my_net.backward(grad_from_loss)
        my_net.update(learning_rate)

    # 最终结果
    y_final_pred = my_net.forward(x_data)

    # 画图
    plt.figure(figsize=(8, 6))
    plt.scatter(x_data, y_true, color='blue', alpha=0.5, s=20, label='Noisy Observations (y_true)')
    plt.plot(x_data, y_ideal, color='red', linestyle='--', linewidth=2, label='Hidden Physics Law ($y=x^2$)')
    plt.plot(x_data, y_final_pred, color='green', linewidth=3, label='Neural Network Prediction')
    stats_text = (
        f"Experiment Results\n"
        f"------------------\n"
        f"Epochs: {epochs}\n"
        f"Learning Rate: {learning_rate}\n"
        f"Final Loss (MSE): {loss:.6e}"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gca().text(0.7, 0.2, stats_text, transform=plt.gca().transAxes, fontsize=10,
                   verticalalignment='top', bbox=props, family='monospace')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Discover Physics Law by Neural Network")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

if __name__ == "__main__":
    run_experiment()