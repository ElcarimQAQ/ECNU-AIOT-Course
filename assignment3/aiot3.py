import torch
import numpy as np
import tqdm
from torch.autograd import Variable
from torch.nn import *
from torch.optim import Adam


def min_max_normalize(data):
    mins = data.min(0)
    maxs = data.max(0)
    ranges = maxs - mins
    row = data.shape[0]
    normalized_data = data - np.tile(mins, (row, 1))
    normalized_data = normalized_data / np.tile(ranges, (row, 1))
    return normalized_data


raw_data = np.loadtxt('data.csv', dtype=str, delimiter=',', skiprows=1, unpack=False)
raw_data = raw_data.astype(float)
# 按列归一化
data_normed = min_max_normalize(raw_data)
# 求每列均值

# 设置模型超参数
input_feature = 3
hidden_feature = 7
output_feature = 1
learning_rate = 1e-6
epochs = 500
loss_f = MSELoss()


# 分别预测温度/压强/湿度
pred = ['Temperature', 'Pressure', 'Humidity']
for i in range(3):
    # 参数初始化
    x = Variable(torch.from_numpy(data_normed), requires_grad=False)
    x = x.to(torch.float32)
    y = Variable(torch.from_numpy(np.array(data_normed[i])), requires_grad=False)
    y = y.to(torch.float32)
    w1 = Variable(torch.randn(input_feature, hidden_feature), requires_grad=True)
    w2 = Variable(torch.randn(hidden_feature, output_feature), requires_grad=True)

    Epoch = []
    Loss = []
    model = Sequential(
        Linear(input_feature, hidden_feature),
        Linear(hidden_feature, output_feature)
    )
    # optimizer需要传入训练参数和lr
    optim = Adam(model.parameters(), lr=learning_rate)
    print(model)
    # 迭代训练
    for epoch in tqdm.tqdm(range(1, epochs + 1)):
        # 前向传播
        y_pred = model(x)
        loss = loss_f(y_pred, y)
        Epoch.append(epoch)
        Loss.append(loss.data)
        optim.zero_grad()
        loss.backward()
        optim.step()

    test_raw_pre = model(x[-1])
    # 反归一化
    mini = raw_data.min(0)[i]
    maxi = raw_data.max(0)[i]
    ranges = maxi - mini
    print("pred " + pred[i] + ' : ' + str(ranges * test_raw_pre + mini))
