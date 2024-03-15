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
from sklearn.preprocessing import MinMaxScaler

# 数据处理
def load_data(topn=None):
    # 创建包含数据的字典
    path = './data/A榜-训练集_海上风电预测_气象变量及实际功率数据.csv'
    #path = 'F:/learning/power_prediction/ans/test_0.047370373032452714.csv'

    df = pd.read_csv(path,encoding='gbk',nrows=topn)
    df = df[df['站点编号'] == 'f1']
    df['时间'] = pd.to_datetime(df['时间'])

    df['mooth'] = df['时间'].dt.month
    df['hour'] = df['时间'].dt.hour

    cols = ['气压(Pa）', '相对湿度（%）', '云量', '10米风速（10m/s）', '10米风向（°)',
            '温度（K）', '辐照强度（J/m2）', '降水（m）', '100m风速（100m/s）', '100m风向（°)', '出力(MW)',
            'mooth', 'hour']
    df = df[df['出力(MW)'] != '<NULL>']
    df = df[cols]
    df['v^3'] = df['100m风速（100m/s）'].apply(lambda x:x*x*x)
    df['1/tmp'] = df['温度（K）'].apply(lambda x:1/(x+0.0001))
    df['new'] = df['v^3']*df['气压(Pa）']*df['1/tmp']

    for col in df.columns:
        df[col] = df[col].astype('float32')
    print(df.columns)
    # 打印输出整理后的数据框
    # 创建MinMaxScaler对象
    scaler = MinMaxScaler()
    # 将数据进行归一化
    cols = df.columns.tolist()
    df = scaler.fit_transform(df.values)
    return pd.DataFrame(df,columns=cols)
    # pred_y = scaler.inverse_transform(pred_y)


def creat_dataset(data, seq_len=96, label=None):
    """用特征对标签进行预测，只考虑特征之间的时序特性，没有利用标签之间的时序特性，进行建模的数据预处理方式
    :param data: 训练数据
    :param seq_len: 时间步
    :param label: 预测的标签名
    :return: 
    """
    feats = [feat for feat in data.columns if feat != label] if label else list(data.columns)
    train_data, target = [], []
    for i in tqdm(range(0, len(data) - seq_len)):
        x = data[i:i + seq_len][feats].values
        y = data.iloc[i+seq_len][label] if label else []
        train_data.append(x)
        target.append(y)
    return torch.tensor(np.array(train_data)), torch.tensor(np.array(target)).view(len(target), 1)


def train_test_split(train_tensor, label_tensor, ratio=0.8):
    train_size = int(len(train_tensor) * ratio)
    x_train, y_train = train_tensor[:train_size], label_tensor[:train_size]
    x_test, y_test = train_tensor[train_size:], label_tensor[train_size:]
    return x_train, y_train, x_test, y_test

