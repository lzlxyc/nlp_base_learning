import torch


def data_set(df,ratio=0.67):
    time_data = df[['Close']].values.astype('float32')
    train_size = int(len(time_data) * ratio)
    train, test = time_data[:train_size], time_data[train_size:]
    # 转为tensor[batch_size, seq_len, emb_size]
    train_tensor = torch.FloatTensor(train).view(-1, train.shape[0], 1)
    test_tensor = torch.FloatTensor(test).view(-1, test.shape[0], 1)
    print('train_tensor.shape:',train_tensor.shape,'test_tensor.shape:',test_tensor.shape)
    return train, test, train_tensor, test_tensor