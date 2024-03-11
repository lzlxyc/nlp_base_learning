import random
import numpy as np
from torchinfo import summary
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import LSTM, GRU, LSTMCell
# 定义优化器和损失函数
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import os
from dataset import load_data, creat_dataset, train_test_split
from utils import fit
from model import LstmModel
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
def train_main(df, model, seq_len=96, label='出力(MW)', batch_size=8, epochs=20, learning_rate=0.01, weight_decay=1e-3):
    torch.manual_seed(1412)
    train_tensor, label_tensor = creat_dataset(df, seq_len, label=label)

    x_train, y_train, x_test, y_test = train_test_split(train_tensor, label_tensor)
    # 批次处理
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=False,
                              num_workers=4, pin_memory=True)
    test_tensor = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False, drop_last=False,
                             num_workers=4, pin_memory=True)
    # 优化器、损失函数、学习率
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss(reduction='mean')
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    # 训练
    fit(model, train_loader, test_tensor, criterion, optimizer, scheduler, epochs)

if __name__ == "__main__":
    # 设置全局的随机种子
    torch.backends.cudnn.deterministic = True  # 将cudnn框架中的随机数生成器设为确定性模式
    torch.backends.cudnn.benchmark = False  # 关闭CuDNN框架的自动寻找最优卷积算法的功能，以避免不同的算法对结果产生影响
    torch.manual_seed(1412)
    random.seed(1412)
    np.random.seed(1412)
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 超参数设置
    input_size = 12
    hidden_size = 256
    num_layers = 10
    output_size = 1
    learning_rate = 0.1
    weight_decay = 1e-4
    num_epochs = 30
    batch_size= 30

    label = '出力(MW)'
    seq_len = 12


    df = load_data()
    input_size = len(df.columns) - 1
    model = LstmModel(input_size, hidden_size, num_layers,output_size)
    train_main(df, model, seq_len=96, label='出力(MW)', batch_size=batch_size,epochs=num_epochs, learning_rate=learning_rate, weight_decay=weight_decay)