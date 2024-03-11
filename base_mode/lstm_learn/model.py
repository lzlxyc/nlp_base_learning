import torch
from torch import nn
from torch.nn import LSTM, GRU, LSTMCell

# 定义网路架构LSTM
# input_size   hidden_size  num_layers out_size
class LstmModel(nn.Module):
    def __init__(self,input_size=12, hidden_size=50, num_layers=1, out_size=1, bidirectional=False, batch_first=True):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = 2 if bidirectional else 1
        self.lstm = LSTM(input_size,hidden_size,num_layers,dropout=0.2, batch_first=batch_first, bidirectional=bidirectional)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bidirectional*hidden_size, out_size)

    def forward(self,x):
        h0 = torch.randn(self.bidirectional*self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.randn(self.bidirectional*self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        output, (_, _) = self.lstm(x, (h0.detach(), c0.detach()))
        # output = output.contiguous().view(x.size(0), x.size(1), 2, self.hidden_size)
        # output = torch.mean(output,dim=2)
        # print(output.size())
        output = self.dropout(output)
        return self.fc(output[:,-1,:])



# 构建双层LSTMcell
# input_size, hidden_size, num_layers, output_size
class LstmCellModel(nn.Module):
    def __init__(self, input_size=1, hidden_size1=100, hidden_size2=50, output_size=1, dropout=0.1):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.lstm0 = nn.LSTMCell(input_size, hidden_size1)
        self.lstm1 = nn.LSTMCell(hidden_size1, hidden_size2)
        self.fc = nn.Linear(hidden_size2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        # 四个初始化参数
        h_10 = torch.randon(batch_size, seq_len, self.hidden_size1).requires_grad_()
        c_10 = torch.randon(batch_size, seq_len, self.hidden_size1).requires_grad_()
        h_11 = torch.randon(batch_size, seq_len, self.hidden_size2).requires_grad_()
        c_11 = torch.randon(batch_size, seq_len, self.hidden_size2).requires_grad_()

        outputs = []
        for t in range(seq_len):  # 遍历每个时间步
            h_10, c_10 = self.lstm0(x[:, t, :], (h_10, c_10))
            h_10, c_10 = self.dropout(h_10), self.dropout(c_10)
            h_11, c_11 = self.lstm1(h_10, (h_11, c_11))
            h_11, c_11 = self.dropout(h_11), self.dropout(c_11)

            outputs.append(h_11)

        return self.fc(outputs[-1])




