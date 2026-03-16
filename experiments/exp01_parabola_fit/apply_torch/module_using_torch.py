import torch
import torch.nn as nn
# 实验主体
class parabola_fit:
    def __init__(self, 
                 true_data_noise_ratio = 0.01, 
                 dot_number = 101, 
                 epochs = 10000, 
                 learning_rate = 0.002, 
                 net = "nn.Linear(1,100),nn.ReLU(),nn.Linear(100,10),nn.Tanh(),nn.Linear(10,1)"
                ):
        self.true_data_noise_ratio = true_data_noise_ratio
        self.dot_number = dot_number
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.net = nn.Sequential(*eval(net))
        self.epoch_x = None
        self.y_final_pred = None
        self.final_loss = None
        self.loss_history = None

    def run_experiment(self):
        # 数据与网络准备
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        x_data = torch.linspace(-1,1,self.dot_number).reshape(-1,1).to(device) # 把行向量变为列向量，batch=100（同时观察100个粒子），input=1（每个粒子有1个自由度需要观测）
        y_ideal = x_data ** 2
        noise = self.true_data_noise_ratio * torch.randn(self.dot_number,1)
        noise = noise.to(device)
        y_true = y_ideal + noise
        my_net = self.net.to(device)
        my_loss = nn.MSELoss()
        my_optimizer = torch.optim.SGD(my_net.parameters(),lr = self.learning_rate)
        loss_history = []

        # 开始训练
        for i in range(self.epochs):
            my_optimizer.zero_grad()
            y_pred = my_net(x_data)
            loss = my_loss(y_pred, y_true)
            loss.backward()
            my_optimizer.step()
            if i%100 == 0:
                loss_history.append(loss.item())

        # 最终结果
        with torch.no_grad():
            self.x_data = x_data
            self.y_ideal = y_ideal
            self.y_true = y_true
            self.y_final_pred = my_net.forward(x_data)
            self.final_loss = my_loss.forward(y_pred, y_true)
            loss_history.append(self.final_loss)
            self.loss_history = loss_history
            self.epoch_x = torch.linspace(0,self.epochs,len(loss_history))
            self.log_loss = torch.log10(self.final_loss)
        

