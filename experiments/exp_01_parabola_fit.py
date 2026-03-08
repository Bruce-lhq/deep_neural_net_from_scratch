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
from matplotlib.widgets import Button

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
    loss_history = []

    # 开始训练
    for i in range(epochs):
        y_pred = my_net.forward(x_data)
        loss = my_loss.forward(y_pred, y_true)
        grad_from_loss = my_loss.backward()
        my_net.backward(grad_from_loss)
        my_net.update(learning_rate)
        if i%100 == 0:
            loss_history.append(loss)

    # 最终结果
    y_final_pred = my_net.forward(x_data)
    loss = my_loss.forward(y_pred, y_true)
    loss_history.append(loss)
    epoch_x = np.linspace(0,epochs,len(loss_history))

    # 画图
    fig = plt.figure(num="Physics Law Discovery", figsize=(12, 9))
    ax1 = fig.add_axes([0.1, 0.15, 0.8, 0.75]) 
    ax2 = fig.add_axes([0.1, 0.15, 0.8, 0.75])
    ax2.set_visible(False) # 初始状态下隐藏第二张图 (Loss)

    ## 第一张图：拟合曲线
    ax1.scatter(x_data, y_true, color='blue', alpha=0.5, s=20, label='Noisy Observations (y_true)')
    ax1.plot(x_data, y_ideal, color='red', linestyle='--', linewidth=2, label='Hidden Physics Law ($y=x^2$)')
    ax1.plot(x_data, y_final_pred, color='green', linewidth=3, label='Neural Network Prediction')

    stats_text = (
        f"Experiment Results\n"
        f"------------------\n"
        f"Epochs: {epochs}\n"
        f"Learning Rate: {learning_rate}\n"
        f"Final Loss (MSE): {loss:.6e}"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.72, 0.18, stats_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, family='monospace')
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title("Fitting")
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    ## 第二张图：损失收敛曲线
    ax2.plot(epoch_x, loss_history, color='darkorange', linewidth=2, label='MSE Loss')
    ax2.set_title("Loss Convergence", fontsize=14)
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Mean Squared Error (Log Scale)")
    ax2.set_yscale('log') 
    ax2.grid(True, which="both", linestyle=':', alpha=0.6)
    ax2.legend()

    ax_button = fig.add_axes([0.4, 0.03, 0.2, 0.06])
    btn = Button(ax_button, 'Switch to Loss Plot')
    def toggle_plots(event):
        if ax1.get_visible():
            ax1.set_visible(False)
            ax2.set_visible(True)
            btn.label.set_text('Switch to Fitting Plot')
        else:
            ax1.set_visible(True)
            ax2.set_visible(False)
            btn.label.set_text('Switch to Loss Plot')
        fig.canvas.draw_idle()
    btn.on_clicked(toggle_plots)
    plt.show()

if __name__ == "__main__":
    run_experiment()