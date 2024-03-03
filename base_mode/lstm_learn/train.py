import torch
from torch import nn


from utils import MSE
from data_process import get_data
from dataset import data_set
from model import LstmModel

from config import *

def train_main():
    df = get_data()
    ratio = 0.67
    train, test, train_tensor, test_tensor = data_set(df,ratio)
    # 实例化模型
    model = LstmModel(input_size, hidden_size, num_layers, output_size)
    # 定义损失函数与优化算法
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    # 开始进行训练
    for epoch in range(num_epochs):
        outputs = model(train_tensor)
        optimizer.zero_grad()
        loss = criterion(outputs, train_tensor[:, :, :])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    print("训练完成")

    model.eval()
    test_outputs = model(test_tensor).detach().numpy()
    test_outputs = list(test_outputs[0, :, 0])
    # print("平均：",MSE(train,train.mean()),MSE(test,test.mean()))
    MSE(test, test_outputs)

if __name__ == '__main__':


