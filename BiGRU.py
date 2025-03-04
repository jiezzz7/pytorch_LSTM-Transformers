import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

current_dir = Path(__file__).resolve().parent
data_file = current_dir / '..' / 'archive' / 'AAPL.csv'

model_save_path = current_dir / 'model_best.pth'

AAPL = pd.read_csv(data_file)
print(type(AAPL['Close'].iloc[0]), type(AAPL['Date'].iloc[0]))
# Let's convert the data type of timestamp column to datatime format
AAPL['Date'] = pd.to_datetime(AAPL['Date'])
print(type(AAPL['Close'].iloc[0]), type(AAPL['Date'].iloc[0]))

# Selecting subset
cond_1 = AAPL['Date'] >= '2021-04-23 00:00:00'
cond_2 = AAPL['Date'] <= '2024-04-23 00:00:00'
AAPL = AAPL[cond_1 & cond_2].set_index('Date')
print(AAPL.shape)

# plt.style.available
plt.style.use('_mpl-gallery')
plt.figure(figsize=(18, 6))
plt.title('Close Price History')
plt.plot(AAPL['Close'], label='AAPL')
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.legend()
plt.show()  # 添加 plt.show()


# 设置时间回溯窗口大小
window_size = 60


# 构造序列数据函数，若target为 Close，即索引为 3
def create_dataset(dataset, lookback=1):
    X, y = [], []
    for i in range(len(dataset) - lookback):
        feature = dataset[i:(i + lookback), :]
        target = dataset[i + lookback, 3]
        X.append(feature)
        y.append(target)
    return np.array(X), np.array(y)


# 选取 AAPL[['Open', 'High', 'Low', 'Close']]作为特征, 归一化数据
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(AAPL[['Open', 'High', 'Low', 'Close']].values)

# 获取反归一化参数(即原始数据的最小值和最大值)
original_min = scaler.data_min_
original_max = scaler.data_max_

# scale_params是一个包含所有特征反归一化参数的列表或数组，
# 其中第四个元素是Close价格的反归一化参数
scale_params = original_max - original_min

# 创建数据集,数据形状为 [samples, time steps, features]
X, y = create_dataset(scaled_data, lookback=window_size)
print(X.shape, y.shape)


# 使用TimeSeriesSplit划分数据集，根据需要调整n_splits
tscv = TimeSeriesSplit(n_splits=3, test_size=90)
# 遍历所有划分进行交叉验证
for i, (train_index, test_index) in enumerate(tscv.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # print(f"Fold {i}:")
    # print(f"  Train: index={train_index}")
    # print(f"  Test:  index={test_index}")

# 查看最后一个 fold 数据帧的维度
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 将 NumPy数组转换为 tensor张量
X_train_tensor = torch.from_numpy(X_train).type(torch.Tensor).to(device)  # 移动到 GPU (步骤三)
X_test_tensor = torch.from_numpy(X_test).type(torch.Tensor).to(device)  # 移动到 GPU
y_train_tensor = torch.from_numpy(y_train).type(torch.Tensor).view(-1, 1).to(device)  # 移动到 GPU
y_test_tensor = torch.from_numpy(y_test).type(torch.Tensor).view(-1, 1).to(device)  # 移动到 GPU

print(X_train_tensor.shape, X_test_tensor.shape, y_train_tensor.shape, y_test_tensor.shape)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # 保证 h0 与 x 在同一设备
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out


model = BiGRU(input_size=X_train.shape[2], hidden_size=64, num_layers=1, output_size=1).to(device)  # 移动到 GPU (步骤二)
criterion = torch.nn.MSELoss()  # 定义均方误差损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 定义优化器

summary(model, (32, 60, 4), device=device)  # 确保 summary 也使用正确的设备

train_losses = []
test_losses = []
best_loss = float('inf')
best_model = None

# 训练循环
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", ncols=80)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)  # 数据移动到 GPU (步骤三)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item() * data.size(0)
        loss.backward()
        optimizer.step()
        pbar.update()

    average_loss = train_loss / len(train_loader.dataset)
    train_losses.append(average_loss)

    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # 数据移动到 GPU (步骤三)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)

        test_loss = test_loss / len(test_loader.dataset)
        test_losses.append(test_loss)

    if test_loss < best_loss:
        best_loss = test_loss
        best_model = model.state_dict()

if best_model is not None:
    torch.save(best_model, model_save_path)
    print(f'Best model saved with test loss {best_loss:.4f}')
else:
    print('No model saved as no improvement was made.')

best_epoch_idx = test_losses.index(min(test_losses))
print(f"Best epoch: {best_epoch_idx + 1}, with test loss: {min(test_losses):.4f}")

plt.figure(figsize=(20, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Training and Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.load_state_dict(torch.load(model_save_path))


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else 0


model.eval()
y_pred_all = []
y_true_all = []

with torch.no_grad():
    pbar = tqdm(test_loader, desc='Evaluating')
    for data, target in pbar:
        data, target = data.to(device), target.to(device)  # 移动到GPU
        y_pred = model(data).detach().cpu().numpy()  # 移动到 CPU 并转为 NumPy (步骤四、五)
        y_true = target.detach().cpu().numpy()  # 移动到 CPU 并转为 NumPy
        y_pred_all.append(y_pred)
        y_true_all.append(y_true)
        pbar.update()

y_pred_all = np.concatenate(y_pred_all)
y_true_all = np.concatenate(y_true_all)

mae = np.mean(np.abs(y_pred_all - y_true_all))
print(f"MAE: {mae:.4f}")

rmse = np.sqrt(np.mean((y_pred_all - y_true_all) ** 2))
print(f"RMSE: {rmse:.4f}")

mape = mean_absolute_percentage_error(y_true_all, y_pred_all)
print(f"MAPE: {mape:.4f}%")

close_denorm_param = scale_params[3]

# 反归一化，并确保在 CPU 上进行 (步骤五)
y_train_pred = model(X_train_tensor).detach().cpu().numpy()  # 移动到 CPU 并转为 NumPy
y_train_denormalized_predictions = (y_train_pred * scale_params[3]) + original_min[3]

y_test_pred = model(X_test_tensor).detach().cpu().numpy()  # 移动到 CPU 并转为 NumPy
y_test_denormalized_predictions = (y_test_pred * scale_params[3]) + original_min[3]

trainPredict = AAPL[window_size:X_train.shape[0] + X_train.shape[1]]
trainPredictPlot = trainPredict.assign(TrainPrediction=y_train_denormalized_predictions)

testPredict = AAPL[X_train.shape[0] + X_train.shape[1]:]
testPredictPlot = testPredict.assign(TestPrediction=y_test_denormalized_predictions)

plt.figure(figsize=(20, 5))
plt.title('BiGRU Close Price Validation',fontsize=40, pad=20)
plt.plot(AAPL['Close'], color='blue', label='original')
plt.plot(trainPredictPlot['TrainPrediction'], color='orange', label='Train Prediction')
plt.plot(testPredictPlot['TestPrediction'], color='red', label='Test Prediction')
plt.legend()
plt.show()

latest_closes = AAPL[['Open', 'High', 'Low', 'Close']][-window_size:].values
scaled_latest_closes = scaler.transform(latest_closes)
tensor_latest_closes = torch.from_numpy(scaled_latest_closes).type(torch.Tensor).view(1, window_size, 4).to(device) # 移动到GPU
print(tensor_latest_closes.shape)

next_close_pred = model(tensor_latest_closes).detach().cpu().numpy()  # 移动到 CPU 并转为 NumPy
next_close_denormalized_pred = (next_close_pred * scale_params[3]) + original_min[3]
print(next_close_denormalized_pred) # 使用print而不是直接返回值
