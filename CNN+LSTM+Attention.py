import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary
from tqdm import tqdm
from pathlib import Path

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 设置随机种子
np.random.seed(0)
torch.manual_seed(0)

class DataHandler:
    """负责数据加载、预处理和数据集创建的类"""
    def __init__(self, data_path, window_size=60):
        self.data_path = data_path
        self.window_size = window_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.original_min = None
        self.original_max = None
        self.scale_params = None

    def load_data(self):
        """加载和筛选数据"""
        AAPL = pd.read_csv(self.data_path)
        AAPL['Date'] = pd.to_datetime(AAPL['Date'])
        cond_1 = AAPL['Date'] >= '2021-04-23 00:00:00'
        cond_2 = AAPL['Date'] <= '2024-04-23 00:00:00'
        self.data = AAPL[cond_1 & cond_2].set_index('Date')
        print(f"Data shape after filtering: {self.data.shape}")

    def plot_close_price(self, images_dir):
        """绘制并保存收盘价历史图"""
        plt.style.use('_mpl-gallery')
        plt.figure(figsize=(20, 5))
        plt.title('Close Price History', fontsize=20, pad=20)
        plt.plot(self.data['Close'], label='AAPL')
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.xticks(rotation=45)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(images_dir / 'close_price_history.png')
        plt.show()

    def create_dataset(self):
        """创建时间序列数据集"""
        scaled_data = self.scaler.fit_transform(self.data[['Open', 'High', 'Low', 'Close']].values)
        self.original_min = self.scaler.data_min_
        self.original_max = self.scaler.data_max_
        self.scale_params = self.original_max - self.original_min
        X, y = [], []
        for i in range(len(scaled_data) - self.window_size):
            feature = scaled_data[i:(i + self.window_size), :]
            target = scaled_data[i + self.window_size, 3]  # Close 列索引为 3
            X.append(feature)
            y.append(target)
        self.X, self.y = np.array(X), np.array(y)
        print(f"Dataset shapes - X: {self.X.shape}, y: {self.y.shape}")

    def split_data(self):
        """划分训练、验证和测试数据集"""
        tscv = TimeSeriesSplit(n_splits=3, test_size=90)
        for train_index, test_index in tscv.split(self.X):
            X_train_full, self.X_test = self.X[train_index], self.X[test_index]
            y_train_full, self.y_test = self.y[train_index], self.y[test_index]

        # 从训练集中划分验证集（20% 作为验证集）
        val_size = int(len(X_train_full) * 0.2)
        train_size = len(X_train_full) - val_size
        self.X_train, self.X_val = X_train_full[:train_size], X_train_full[train_size:]
        self.y_train, self.y_val = y_train_full[:train_size], y_train_full[train_size:]
        print(f"Train, val, test shapes - X_train: {self.X_train.shape}, X_val: {self.X_val.shape}, X_test: {self.X_test.shape}")

    def prepare_tensors(self):
        """将数据转换为 PyTorch tensor并移到指定设备"""
        X_train_tensor = torch.from_numpy(self.X_train).type(torch.Tensor).to(device)
        X_val_tensor = torch.from_numpy(self.X_val).type(torch.Tensor).to(device)
        X_test_tensor = torch.from_numpy(self.X_test).type(torch.Tensor).to(device)
        y_train_tensor = torch.from_numpy(self.y_train).type(torch.Tensor).view(-1, 1).to(device)
        y_val_tensor = torch.from_numpy(self.y_val).type(torch.Tensor).view(-1, 1).to(device)
        y_test_tensor = torch.from_numpy(self.y_test).type(torch.Tensor).view(-1, 1).to(device)
        return X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor

    def get_dataloaders(self, batch_size=32):
        """创建 DataLoader"""
        X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor = self.prepare_tensors()
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

class CNN_LSTM_MultiheadAttention(nn.Module):
    """CNN-LSTM-MultiheadAttention 模型类"""
    def __init__(self, in_channels=4, out_channels=64, hidden_size=128, num_layers=1,
                 kernel_size=3, num_heads=8, output_size=1):
        super(CNN_LSTM_MultiheadAttention, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1)
        self.lstm = nn.LSTM(out_channels, hidden_size, num_layers, batch_first=True)
        self.lstm_fc = nn.Linear(hidden_size, hidden_size)
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.fc_out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x, (h0, c0))
        query, key, value = out[:, -1, :], self.lstm_fc(out), self.lstm_fc(out)
        attn_output, attn_weights = self.multihead_attn(query.unsqueeze(1), key, value)
        combination = torch.cat((query.squeeze(1), attn_output.squeeze(1)), dim=1)
        output = self.fc_out(combination)
        return output

def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs=20):
    """训练模型并保存最佳权重"""
    train_losses = []
    val_losses = []
    test_losses = []
    best_loss = float('inf')
    best_model = None

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

    if best_model:
        current_dir = Path.cwd()
        model_dir = current_dir / 'CNN_LSTM_MultiheadAttention_model'
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_model, model_dir / 'model_best.pth')
        print(f"Best model saved with test loss {best_loss:.4f}")

    return train_losses, val_losses, test_losses

def evaluate_model(model, test_loader):
    """评估模型性能"""
    model.eval()
    y_pred_all, y_true_all = [], []
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            y_pred = model(data).detach().cpu().numpy()
            y_true = target.detach().cpu().numpy()
            y_pred_all.append(y_pred)
            y_true_all.append(y_true)
            pbar.update()

    y_pred_all = np.concatenate(y_pred_all)
    y_true_all = np.concatenate(y_true_all)

    mae = np.mean(np.abs(y_pred_all - y_true_all))
    rmse = np.sqrt(np.mean((y_pred_all - y_true_all) ** 2))
    mape = mean_absolute_percentage_error(y_true_all, y_pred_all)
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}%")

    # 计算 R²
    y_bar = np.mean(y_true_all)
    SS_res = np.sum((y_true_all - y_pred_all) ** 2)
    SS_tot = np.sum((y_true_all - y_bar) ** 2)
    R2 = 1 - (SS_res / SS_tot)
    print(f"R²: {R2:.4f}")

def mean_absolute_percentage_error(y_true, y_pred):
    """计算 MAPE"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else 0

def plot_losses(train_losses, val_losses, test_losses, images_dir):
    """绘制并保存训练、验证和测试损失图"""
    plt.figure(figsize=(18, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training, Validation, and Test Loss', fontsize=20)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(images_dir / 'train_val_test_loss.png')
    plt.show()

def predict_next_close(model, data_handler):
    """预测下一个收盘价"""
    latest_data = data_handler.data[['Open', 'High', 'Low', 'Close']][-data_handler.window_size:].values
    scaled_latest = data_handler.scaler.transform(latest_data)
    tensor_latest = torch.from_numpy(scaled_latest).type(torch.Tensor).view(1, data_handler.window_size, 4).to(device)
    next_pred = model(tensor_latest).detach().cpu().numpy()
    next_denorm_pred = (next_pred * data_handler.scale_params[3]) + data_handler.original_min[3]
    print(f"Predicted next close price: {next_denorm_pred[0][0]:.4f}")

def plot_predictions(data_handler, model, images_dir):
    """绘制并保存预测结果图，划分为三段"""
    X_train_tensor, X_val_tensor, X_test_tensor, _, _, _ = data_handler.prepare_tensors()
    y_train_pred = model(X_train_tensor).detach().cpu().numpy()
    y_val_pred = model(X_val_tensor).detach().cpu().numpy()
    y_test_pred = model(X_test_tensor).detach().cpu().numpy()
    y_train_denorm = (y_train_pred * data_handler.scale_params[3]) + data_handler.original_min[3]
    y_val_denorm = (y_val_pred * data_handler.scale_params[3]) + data_handler.original_min[3]
    y_test_denorm = (y_test_pred * data_handler.scale_params[3]) + data_handler.original_min[3]

    trainPredict = data_handler.data.iloc[data_handler.window_size:data_handler.window_size + len(y_train_denorm)]
    trainPredictPlot = trainPredict.assign(TrainPrediction=y_train_denorm)

    valPredict = data_handler.data.iloc[data_handler.window_size + len(y_train_denorm):data_handler.window_size + len(y_train_denorm) + len(y_val_denorm)]
    valPredictPlot = valPredict.assign(ValPrediction=y_val_denorm)

    testPredict = data_handler.data.iloc[data_handler.window_size + len(y_train_denorm) + len(y_val_denorm):]
    testPredictPlot = testPredict.assign(TestPrediction=y_test_denorm)

    plt.figure(figsize=(20, 5))
    plt.title('CNN-LSTM-MultiheadAttention Close Price Validation', fontsize=20, pad=20)
    plt.plot(data_handler.data['Close'], color='blue', label='Original')
    plt.plot(trainPredictPlot['TrainPrediction'], color='orange', label='Train Prediction')
    plt.plot(valPredictPlot['ValPrediction'], color='green', label='Validation Prediction')
    plt.plot(testPredictPlot['TestPrediction'], color='red', label='Test Prediction')

    # 添加划分线和文本
    max_close = data_handler.data['Close'].max()
    if not trainPredictPlot.empty:
        train_start = trainPredictPlot.index[0]
        plt.axvline(x=train_start, color='black', linestyle='--')
        plt.text(train_start, max_close, 'Train', fontsize=15, verticalalignment='top')  # [^1]

    if not valPredictPlot.empty:
        val_start = valPredictPlot.index[0]
        plt.axvline(x=val_start, color='black', linestyle='--')
        plt.text(val_start, max_close, 'Validation', fontsize=15, verticalalignment='top')  # [^1]

    if not testPredictPlot.empty:
        test_start = testPredictPlot.index[0]
        plt.axvline(x=test_start, color='black', linestyle='--')
        plt.text(test_start, max_close, 'Test', fontsize=15, verticalalignment='top')  # [^1]

    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.xlabel('Date', fontsize=16)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(images_dir / 'prediction_results.png')
    plt.show()

def main():
    """主函数，整合所有功能"""
    # 定义路径
    current_dir = Path.cwd()
    data_file = current_dir / '..' / 'archive' / 'AAPL.csv'
    images_dir = current_dir / 'CNN_LSTM_MultiheadAttention_image'
    images_dir.mkdir(parents=True, exist_ok=True)

    # 数据处理
    data_handler = DataHandler(data_path=data_file, window_size=60)
    data_handler.load_data()
    data_handler.plot_close_price(images_dir)
    data_handler.create_dataset()
    data_handler.split_data()
    train_loader, val_loader, test_loader = data_handler.get_dataloaders()

    # 模型初始化
    model = CNN_LSTM_MultiheadAttention().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # 调整学习率

    # 输出模型概要
    summary(model, (32, 60, 4), device=device)

    # 训练模型
    train_losses, val_losses, test_losses = train_model(model, train_loader, val_loader, test_loader, criterion, optimizer)
    plot_losses(train_losses, val_losses, test_losses, images_dir)

    # 加载最佳模型
    model.load_state_dict(torch.load(current_dir / 'CNN_LSTM_MultiheadAttention_model' / 'model_best.pth'))

    # 评估模型
    evaluate_model(model, test_loader)

    # 绘制预测结果
    plot_predictions(data_handler, model, images_dir)

    # 预测下一个收盘价
    predict_next_close(model, data_handler)

if __name__ == "__main__":
    main()
