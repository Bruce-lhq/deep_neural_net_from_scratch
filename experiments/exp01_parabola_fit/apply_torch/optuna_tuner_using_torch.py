# 通过贝叶斯优化对 parabola_fit 的超参数进行优化

from module_using_torch import parabola_fit
import optuna
from optuna.visualization import plot_param_importances
import torch

def run_optimization():
    n_trials = 30 # 实验次数
    def objective(trial):
        # 需要优化的参数如下
        learning_rate = trial.suggest_float("learning_rate", 5e-4, 3e-3, log=True)

        hidden_size_1 = trial.suggest_int("hidden_size_1", 500, 1000)
        hidden_size_2 = trial.suggest_int("hidden_size_2", 5, 20)
        net = f"nn.Linear(1, {hidden_size_1}), nn.ReLU(), nn.Linear({hidden_size_1}, {hidden_size_2}), nn.Linear({hidden_size_2}, 1)"
        
        true_data_noise_ratio = trial.suggest_float("true_data_noise_ratio", 5e-5, 1e-3, log=True)
        dot_number = trial.suggest_int("dot_number", 1000, 2000)
        epochs = trial.suggest_int("epochs", 1000, 10000)
        
        # 创建实验并得到结果
        my_experiment = parabola_fit(learning_rate=learning_rate, 
                                     net=net, 
                                     true_data_noise_ratio=true_data_noise_ratio,
                                     dot_number=dot_number,
                                     epochs=epochs)
        my_experiment.run_experiment()
        loss = my_experiment.log_loss

        # 对禁区剪枝
        if torch.isnan(loss) or torch.isinf(loss):
            raise optuna.TrialPruned() 
        
        return loss
    
    # 开始优化
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials) 

    # 保留三位小数
    print("找到了最优的超参数：", {k:float(f"{v:.3g}") if isinstance(v, float) else v for k,v in study.best_params.items()})
    print("此超参数下 Loss = ", float(f"{10**study.best_value:.3g}"))

    # 各参数对结果的影响力
    importance = optuna.importance.get_param_importances(study)
    print(f"参数敏感度分析: ", {k:float(f"{v:.3g}") if isinstance(v, float) else v for k,v in importance.items()})
    
    # 用图直观展现各参数对结果的影响力
    fig = plot_param_importances(study)
    fig.show()

if __name__ == "__main__":
    run_optimization()