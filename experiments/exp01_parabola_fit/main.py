# 实验 01: 拟合抛物线物理规律 (Parabola Fitting)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import module


# 实验主体
def run_experiment():
    # 可调超参数
    my_experiment = module.parabola_fit(
        true_data_noise_ratio = 0.01, 
        dot_number = 101, 
        epochs = 10000, 
        learning_rate = 0.002, 
        layers = ["LinearLayer(1, 200)", "ReLU()", "LinearLayer(200, 6)", "LinearLayer(6, 1)"]
        )

    my_experiment.run_experiment()

    # 画图
    fig = plt.figure(num="Physics Law Discovery", figsize=(12, 9))
    ax1 = fig.add_axes([0.1, 0.15, 0.8, 0.75]) 
    ax2 = fig.add_axes([0.1, 0.15, 0.8, 0.75])
    ax2.set_visible(False) # 初始状态下隐藏第二张图 (Loss)

    ## 第一张图：拟合曲线
    ax1.scatter(my_experiment.x_data, my_experiment.y_true, color='blue', alpha=0.5, s=20, label='Noisy Observations (y_true)')
    ax1.plot(my_experiment.x_data, my_experiment.y_ideal, color='red', linestyle='--', linewidth=2, label='Hidden Physics Law ($y=x^2$)')
    ax1.plot(my_experiment.x_data, my_experiment.y_final_pred, color='green', linewidth=3, label='Neural Network Prediction')

    stats_text = (
        f"Experiment Results\n"
        f"------------------\n"
        f"Epochs: {my_experiment.epochs}\n"
        f"Learning Rate: {my_experiment.learning_rate}\n"
        f"Final Loss (MSE): {my_experiment.final_loss:.6e}"
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
    ax2.plot(my_experiment.epoch_x, my_experiment.loss_history, color='darkorange', linewidth=2, label='MSE Loss')
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