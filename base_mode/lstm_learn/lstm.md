


nn.LSTM(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False, proj_size=0, device=None, dtype=None)

input_size: 输入特征的数量，也是输入层的神经元数量。对输入数据为三维数据的LSTM来说，input_size应当等于三维时间序列数据中的input_sizeension。

num_layers：隐藏层的数量。如下图所示，左侧就是一个隐藏层数量为1的LSTM，右侧就是隐藏层数量为3的LSTM。

hidden_siz**e: 隐藏层的神经元数量，对LSTM来说就是隐藏层上的记忆细胞的数量。在下图中，每个隐藏层上的记忆细胞（神经元）数量是5个。与RNN一致，在LSTM中我们一般默认全部隐藏层上的神经元数量是一致的。**

drop_out: 在神经网络中常见的抗过拟合机制，在Pytorch被内置在LSTM里帮助对抗过拟合。Dropout是在神经网络传播过程中，随机让部分参数为0的抗过拟合方法。令参数为0，就可以切断神经网络层与层之间的链接，从而切断信息的传播，以此来阻碍神经网络的学习，达到对抗过拟合的目的：

batch_first: 如果为True，输入和输出Tensor的形状为 [batch_size, seq_len, input_sizeension]，否则为[seq_len, batch_size, input_sizeension]。当数据是时间序列数据时，seq_len是time_step。注意，默认值为False，所以pytorch官方所使用的结构是[seq_len, batch_size, input_sizeension]。注意，和循环神经网络一致的是，LSTM一定会遵循时间步的方式进行循环，因此确定正确的时间步维度非常重要！


LSTM类有三个输出，一个是output，一个是hn，还有一个是cn

output：代表所有时间步上最后一个隐藏层上输出的隐藏状态的集合（如图所示，是红色方框中的方向上的隐藏状态的集合）。很显然，对单层的LSTM来说，output代表了唯一一个隐藏层上的隐藏状态。output的形状都为 [seq_len, batch_size, hidden_size]，不受隐藏层数量的影响。注意，当参数batch_first=True时，output的形状为[batch_size, seq_len, hidden_size]。

hn: 最后一个时间步的、所有隐藏层上的隐藏状态。形状为 [num_layers, batch_size, hidden_size]。无论batch_first参数是什么取值，都不改变hn的输出。

cn：最后一个时间步的、所有隐藏层上的细胞状态。形状为 [num_layers, batch_size, hidden_size]。无论batch_first参数是什么取值，都不改变cn的输出。