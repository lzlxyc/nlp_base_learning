import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import LSTM, GRU, LSTMCell
import os

# 一次完整的迭代
def train_once(model, criterion, opt, x, y_true):
    """对模型进行一次迭代
    :param model: 实例化后的模型
    :param criterion: 损失函数
    :param opt: 优化算法
    :param x:
    :param y:
    """
    opt.zero_grad(set_to_none=True)  # 设置梯度为None节省内存
    y_pred = model(x)
    loss = criterion(y_pred, y_true)
    loss.backward()
    opt.step()
    return y_pred.detach().cpu().numpy(), loss.item()


def test_once(model, criterion, x, y_true):
    """进行一次测试，阻止计算图追踪,节省内存，加快速度
    :param model:
    :param criterion:
    :param opt:
    :param x:
    :param y_true:
    """
    with torch.no_grad():
        y_pred = model(x)
        loss = criterion(y_pred, y_true)
        return y_pred.detach().cpu().numpy(), loss.item()


def plotloss(train_loss_list, test_loss_list, is_loss=True):
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss_list, color='red', label='Train_loss' if is_loss else 'pred_values')
    plt.plot(test_loss_list, color='blue', label='Test_loss' if is_loss else 'true_values')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    if not os.path.exists('./pic'):
        os.mkdir('./pic')
    plt.savefig('./pic/loss.png' if is_loss else './picpred_true.png', dpi=400)



# 定义提前停止损失函数
class EarlyStopping():
    """
    在测试集上的损失连续几个epochs不在降低时，提前停止
    """
    def __init__(self, patiende=5, tol=0.000005):
        """
        :param patiende:连续 patiende个epoch上损失不再降低，停止迭代
        :param tol: 当前损失和旧损失的差值小于tol，就认定为模型不再提升
        """
        self.patience = patiende
        self.tol = tol
        self.counter = 0
        self.lowest_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if not self.lowest_loss:
            self.lowest_loss = val_loss
        elif self.lowest_loss - val_loss > self.tol:
            self.counter = 0
            self.lowest_loss = val_loss
        elif self.lowest_loss - val_loss <= self.tol:
            self.counter += 1
            print(f"\t NOTICE: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print("\t NOTICE: Early Stopping Actived")
                self.early_stop = True

        return self.early_stop


# %%
# 训练与测试的函数
def fit(model, batch_train, batch_test, criterion, optimizer, scheduler, epochs, patiende=5, tol=0.000005,
        save_model_path='./model/', model_name='lstm'):
    """
    对模型进行训练，在每个epoch上监控模型训练效果
    :param model:
    :param batchdata:
    :param testdata:
    :param criterion:
    :param optimizer:
    :param epochs:
    :param tol:
    :param model_name:
    :param path:
    :return:
    """
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)
    all_train_num = len(batch_train.dataset)
    all_test_num = len(batch_test.dataset)
    print("开始训练..............................")
    print("\tall_train_num:", all_train_num, "\tall_test_num", all_test_num)
    train_loss_list = []
    test_loss_list = []
    early_stopping = EarlyStopping(patiende, tol)
    best_score = None
    best_epoch = 0
    bset_epoch_predictions, bset_epoch_true_values = [], []

    for epoch in range(1, epochs + 1):
        # 训练
        model.train()
        train_num = 0
        loss_train = 0
        for idx, (x, y) in enumerate(batch_train, 1):
            _, loss = train_once(model, criterion, optimizer, x, y)
            loss_train += loss * x.size(0)
            train_num += x.shape[0]
            # # 监控训练过程
            # if idx % 100 == 0:
            #     print(f"Epoch:{epoch}, train_num:{train_num}")

        loss_train = loss_train / all_train_num
        train_loss_list.append(loss_train)

        model.eval()
        epoch_predictions = []
        epoch_true_values = []

        loss_test = 0
        for x, y in batch_test:
            y_pred, loss = test_once(model, criterion, x, y)
            loss_test += loss * x.size(0)
            epoch_predictions.append(y_pred)
            epoch_true_values.append(y.numpy())

        epoch_predictions = np.concatenate(epoch_predictions, axis=0)
        epoch_true_values = np.concatenate(epoch_true_values, axis=0)
        rmse = np.sqrt(np.mean((epoch_predictions - epoch_true_values) ** 2))

        loss_test = loss_test / all_test_num
        plot_num = 1000
        test_loss_list.append(loss_test)
        # 对每一个epoch,打印训练和测试结果
        print(f"Epoch:{epoch}, Train_loss:{round(loss_train, 5)}, Test_loss:{round(loss_test, 5)}", "Rmse:", rmse)
        scheduler.step(loss_test)

        # 对每一个epoch，保存分数最高的权重  这里用loss做评价指标
        if not best_score or best_score > rmse:
            best_epoch = epoch
            best_score = rmse
            torch.save(model.state_dict(), os.path.join(save_model_path, 'best_' + str(epoch) + model_name + '.pt'))
            bset_epoch_predictions, bset_epoch_true_values = list(epoch_predictions.reshape(-1))[:plot_num], list(epoch_true_values.reshape(-1))[:plot_num]


            # print('\t Weights Save')

        early_stop = early_stopping(loss_test)
        if early_stop: break

    print("\tBest_epoch", best_epoch, "\tBest_loss:", best_score)
    print("Done")

    print("bset_epoch_predictions:", bset_epoch_predictions, "\nbset_epoch_true_values:",bset_epoch_true_values)
    plotloss(bset_epoch_predictions, bset_epoch_true_values, is_loss=False)
    plotloss(train_loss_list, test_loss_list)



# 模型验证
def MSE(Y_ture, Y_predict):
    return (((Y_ture - Y_predict) ** 2).sum() / Y_ture.shape[0])**0.5