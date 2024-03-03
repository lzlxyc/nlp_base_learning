import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import LSTM, GRU, LSTMCell



def plot_fit(y_test, y_pred):
    plt.figure(figsize=(4, 2))
    plt.plot(y_test, color="red", label="actual")
    plt.plot(y_pred, color="blue", label="predict")
    plt.title(f"true--pred")
    plt.xlabel("Time")
    plt.ylabel('power')
    plt.legend()
    plt.show()
    plt.savefig('test.png', dpi=400)


# 模型验证
def MSE(Y_ture, Y_predict):
    plot_fit(Y_ture, Y_predict)
    return (((Y_ture - Y_predict) ** 2).sum() / Y_ture.shape[0])**0.5