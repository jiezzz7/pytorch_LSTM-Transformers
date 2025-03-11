import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary
from tqdm import tqdm

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")  # [^2]

# 获取当前工作目录
current_dir = Path.cwd()

# 定义数据文件路径
data_file = current_dir / '..' / 'archive' / 'AAPL.csv'

# 定义图像和模型保存文件夹
images_dir = current_dir / 'BiGRU_image'
models_dir = current_dir / 'BiGRU_model'

# 创建文件夹（若不存在）
images_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)

# 定义模型保存路径
model_save_path = models_dir / 'best_model.pth'

# 加载数据
AAPL = pd.read_csv(data_file)
print(f"Initial data types - Close: {type(AAPL['Close'].iloc[0])}, Date: {type(AAPL['Date'].iloc[0])}")

# 将日期列转换为 datetime 格式
AAPL['Date'] = pd.to_datetime(AAPL['Date'])
print(f"Converted data types - Close: {type(AAPL['Close'].iloc[0])}, Date: {type(AAPL['Date'].iloc[0])}")

# 筛选指定时间范围的数据并设置索引
cond_1 = AAPL['Date'] >= '2021-04-23 00:00:00'
cond_2 = AAPL['Date'] <= '2024-04-23 00:00:00'
AAPL = AAPL[cond_1 & cond_2].set_index('Date')
print(f"Data shape after filtering: {AAPL.shape}")

# 绘制收盘价历史图
plt.style.use('_mpl-gallery')
plt.figure(figsize=(20, 5))
plt.title('Close Price History', fontsize=20, pad=20)
plt.plot(AAPL['Close'], label='AAPL')
plt.ylabel('Close Price USD ($)', fontsize=18)

# 自动调整日期刻度
locator = mdates.AutoDateLocator(minticks=8, maxticks=12)  # [^5]
formatter = mdates.DateFormatter('%Y-%m-%d')  # [^5]
plt.gca().xaxis.set_major_locator(locator)
plt.gca().xaxis.set_major_formatter(formatter)
plt.xticks(rotation=45)  # 提高可读性 [^4]

plt.legend(loc="upper right")  # [^4]
plt.tight_layout()
plt.savefig(images_dir / 'close_price_history.png')
plt.show()

# 设置时间窗口大小
window_size = 60

# 定义创建时间序列数据集的函数
def create_dataset(dataset, lookback=1):
    X, y = [], []
    for i in range(len(dataset) - lookback):
        feature = dataset[i:(i + lookback), :]
        target = dataset[i + lookback, 3]  # Close 列索引为 3
        X.append(feature)
        y.append(target)
    return np.array(X), np.array(y)

# 特征选择与归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(AAPL[['Open', 'High', 'Low', 'Close']].values)

# 获取反归一化参数
original_min = scaler.data_min_
original_max = scaler.data_max_
scale_params = original_max - original_min

# 创建数据集
X, y = create_dataset(scaled_data, lookback=window_size)
print(f"Dataset shapes - X: {X.shape}, y: {y.shape}")

# 使用 TimeSeriesSplit 划分数据集
tscv = TimeSeriesSplit(n_splits=3, test_size=90)
for train_index, test_index in tscv.split(X):
    X_train_full, X_test = X[train_index], X[test_index]
    y_train_full, y_test = y[train_index], y[test_index]

# 从训练集中划分验证集（20% 作为验证集）
val_size = int(len(X_train_full) * 0.2)
train_size = len(X_train_full) - val_size
X_train, X_val = X_train_full[:train_size], X_train_full[train_size:]
y_train, y_val = y_train_full[:train_size], y_train_full[train_size:]

print(f"Train, val, test shapes - X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

# 转换为 PyTorch 张量
X_train_tensor = torch.from_numpy(X_train).type(torch.Tensor).to(device)
X_val_tensor = torch.from_numpy(X_val).type(torch.Tensor).to(device)
X_test_tensor = torch.from_numpy(X_test).type(torch.Tensor).to(device)
y_train_tensor = torch.from_numpy(y_train).type(torch.Tensor).view(-1, 1).to(device)
y_val_tensor = torch.from_numpy(y_val).type(torch.Tensor).view(-1, 1).to(device)
y_test_tensor = torch.from_numpy(y_test).type(torch.Tensor).view(-1, 1).to(device)

# 创建 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义 BiGRU 模型
class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# 初始化模型
model = BiGRU(input_size=X_train.shape[2], hidden_size=64, num_layers=1, output_size=1).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 打印模型概要
summary(model, (32, window_size, X_train.shape[2]), device=device)

# 训练模型
train_losses, val_losses, test_losses = [], [], []
best_loss = float('inf')
best_model = None
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", ncols=80)
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item() * data.size(0)
        loss.backward()
        optimizer.step()
        pbar.update()

    train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # 验证集评估
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item() * data.size(0)
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

    # 测试集评估
    test_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
        test_loss = test_loss / len(test_loader.dataset)
        test_losses.append(test_loss)

    if test_loss < best_loss:
        best_loss = test_loss
        best_model = model.state_dict()

# 保存最佳模型
if best_model:
    torch.save(best_model, model_save_path)
    print(f"Best model saved at {model_save_path} with test loss {best_loss:.4f}")

# 绘制训练、验证和测试损失
plt.figure(figsize=(20, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Training, Validation, and Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)  # 添加网格线
plt.savefig(images_dir / 'train_val_test_loss.png')
plt.show()

# 加载最佳模型
model.load_state_dict(torch.load(model_save_path))

# 定义评估指标函数
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else 0

# 模型评估
model.eval()
y_pred_all, y_true_all = [], []
with torch.no_grad():
    for data, target in tqdm(test_loader, desc='Evaluating'):
        data, target = data.to(device), target.to(device)
        y_pred = model(data).detach().cpu().numpy()
        y_true = target.detach().cpu().numpy()
        y_pred_all.append(y_pred)
        y_true_all.append(y_true)

y_pred_all = np.concatenate(y_pred_all)
y_true_all = np.concatenate(y_true_all)

mae = np.mean(np.abs(y_pred_all - y_true_all))
rmse = np.sqrt(np.mean((y_pred_all - y_true_all) ** 2))
mape = mean_absolute_percentage_error(y_true_all, y_pred_all)
print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}%")

# 反归一化预测结果
y_train_pred = model(X_train_tensor).detach().cpu().numpy()
y_train_denorm = (y_train_pred * scale_params[3]) + original_min[3]
y_val_pred = model(X_val_tensor).detach().cpu().numpy()
y_val_denorm = (y_val_pred * scale_params[3]) + original_min[3]
y_test_pred = model(X_test_tensor).detach().cpu().numpy()
y_test_denorm = (y_test_pred * scale_params[3]) + original_min[3]

# 创建预测 DataFrame
trainPredict = AAPL.iloc[window_size:window_size + len(y_train)]
trainPredictPlot = trainPredict.assign(TrainPrediction=y_train_denorm)

valPredict = AAPL.iloc[window_size + len(y_train):window_size + len(y_train) + len(y_val)]
valPredictPlot = valPredict.assign(ValPrediction=y_val_denorm)

testPredict = AAPL.iloc[window_size + len(y_train) + len(y_val):window_size + len(y_train) + len(y_val) + len(y_test)]
testPredictPlot = testPredict.assign(TestPrediction=y_test_denorm)

# 绘制预测结果并划分阶段
plt.figure(figsize=(20, 5))
plt.title('BiGRU Close Price Validation', fontsize=20, pad=20)
plt.plot(AAPL['Close'], color='blue', label='Original')
plt.plot(trainPredictPlot['TrainPrediction'], color='orange', label='Train Prediction')
plt.plot(valPredictPlot['ValPrediction'], color='green', label='Validation Prediction')
plt.plot(testPredictPlot['TestPrediction'], color='red', label='Test Prediction')

# 添加划分线和文本
max_close = AAPL['Close'].max()
if not trainPredictPlot.empty:
    train_start = trainPredictPlot.index[0]
    plt.axvline(x=train_start, color='black', linestyle='-.')
    plt.text(train_start, max_close, 'Train', fontsize=15, verticalalignment='top')

if not valPredictPlot.empty:
    val_start = valPredictPlot.index[0]
    plt.axvline(x=val_start, color='black', linestyle='-.')
    plt.text(val_start, max_close, 'Validation', fontsize=15, verticalalignment='top')

if not testPredictPlot.empty:
    test_start = testPredictPlot.index[0]
    plt.axvline(x=test_start, color='black', linestyle='-.')
    plt.text(test_start, max_close, 'Test', fontsize=15, verticalalignment='top')  # [^1]

plt.legend()
plt.savefig(images_dir / 'prediction_results.png')
plt.show()

# 预测下一个收盘价
latest_data = AAPL[['Open', 'High', 'Low', 'Close']][-window_size:].values
scaled_latest = scaler.transform(latest_data)
tensor_latest = torch.from_numpy(scaled_latest).type(torch.Tensor).view(1, window_size, 4).to(device)

next_pred = model(tensor_latest).detach().cpu().numpy()
next_denorm_pred = (next_pred * scale_params[3]) + original_min[3]
print(f"Predicted next close price: {next_denorm_pred[0][0]:.4f}")

# 绘制回归图
y_val_true_denorm = (y_val * scale_params[3]) + original_min[3]
y_test_true_denorm = (y_test * scale_params[3]) + original_min[3]

plt.figure(figsize=(10, 5))
sns.regplot(x=y_val_true_denorm, y=y_val_denorm.flatten(), line_kws={'color': 'red'})  # [^6]
plt.title('Validation Prediction Regression')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.savefig(images_dir / 'val_regression.png')
plt.show()

plt.figure(figsize=(10, 5))
sns.regplot(x=y_test_true_denorm, y=y_test_denorm.flatten(), line_kws={'color': 'red'})  # [^6]
plt.title('Test Prediction Regression')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.savefig(images_dir / 'test_regression.png')
plt.show()
