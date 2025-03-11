import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from torchinfo import summary
from torchmetrics.functional.regression import mean_absolute_error
from torchmetrics.functional.regression import mean_absolute_percentage_error
from torchmetrics.functional.regression import mean_squared_error
from torchmetrics.functional.regression import normalized_root_mean_squared_error
from torchmetrics.functional.regression import r2_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using device:{device}")

# 设置随机种子
np.random.seed(0)
torch.manual_seed(0)


def create_directories():
    results_dir = Path('lstm_transformer_results')
    models_dir = Path('lstm_transformer_models')
    images_dir = Path('lstm_transformer_images')

    results_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)

    return results_dir, models_dir, images_dir


# 定义时间序列数据类,特征缩放进行归一化
class TimeseriesData(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.scaler_features = MinMaxScaler()
        self.scaler_target = MinMaxScaler()
        self._values_cache = None
        self._labels_cache = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def values(self):
        # 如果缓存中已经存储了值，则直接返回缓存的值
        if self._values_cache is None:
            # 使用所有特征列作为输入
            data = self.dataset[['mile', 'Vs', 'Vp/Vs']].values
            self._values_cache = self.scaler_features.fit_transform(data)
        return self._values_cache

    def labels(self):
        # 如果缓存中已经存储了标签，则直接返回缓存的标签
        if self._labels_cache is None:
            targets = self.dataset['Vp'].values.reshape(-1, 1)
            self._labels_cache = self.scaler_target.fit_transform(targets)
        return self._labels_cache


# 定义监督学习时间序列数据类
class SupervisedTimeseriesData(TimeseriesData):
    def __init__(self, dataset: pd.DataFrame, lag: int = 10):
        super(SupervisedTimeseriesData, self).__init__(dataset=dataset)
        self.lag = lag
        self._supervised_values_cache = None
        self._supervised_labels_cache = None

    @property
    def supervised_values(self):
        if self._supervised_values_cache is None:
            self._supervised_values_cache = self._compute_supervised_values()
        return self._supervised_values_cache

    @property
    def supervised_labels(self):
        if self._supervised_labels_cache is None:
            self._supervised_labels_cache = self._compute_supervised_labels()
        return self._supervised_labels_cache

    def _compute_supervised_values(self):
        x = [self.values()[i:i + self.lag] for i in range(self.__len__() - self.lag)]
        return torch.tensor(np.array(x), dtype=torch.float)

    def _compute_supervised_labels(self):
        return torch.tensor(self.labels()[self.lag:], dtype=torch.float)


# 定义监督学习时间序列数据集
class SupervisedTimeseriesDataset(Dataset):
    def __init__(self, dataset, lag=30):
        super(SupervisedTimeseriesDataset, self).__init__()
        self.set = SupervisedTimeseriesData(dataset=dataset, lag=lag)
        self.dataset = TensorDataset(self.set.supervised_values, self.set.supervised_labels)
        self.train_idx = list(range(self.__len__() * 3 // 5))
        self.val_idx = list(range(self.__len__() * 3 // 5, self.__len__() * 4 // 5))
        self.test_idx = list(range(self.__len__() * 4 // 5, self.__len__()))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.set.supervised_values[index], self.set.supervised_labels[index]

    @property
    def train_set(self):
        return Subset(self.dataset, indices=self.train_idx)

    @property
    def val_set(self):
        return Subset(self.dataset, indices=self.val_idx)

    @property
    def test_set(self):
        return Subset(self.dataset, indices=self.test_idx)


# 定义LSTM-Transformer模型
class LSTMTransformer(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 lstm_layers,
                 transformer_heads,
                 transformer_layers,
                 output_dim,
                 dropout=0.5
                 ):
        super(LSTMTransformer, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers, dropout=dropout, batch_first=True)
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_encoder_layer,
            num_layers=transformer_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        transformer_input = lstm_out
        transformer_out = self.transformer_encoder(transformer_input)
        output = self.fc(transformer_out[:, -1, :])
        return output


# 定义训练函数
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0

    for batch_idx, (data, target) in enumerate(iterable=iterator):
        # 统一设备
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(iterator)

    return avg_loss


# 定义评估函数
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(iterable=iterator):
            # 统一设备
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(iterator)

        return avg_loss


# 定义预测函数
def predict(model, iterator, device):
    model.eval()
    targets = []
    predictions = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(iterable=iterator):
            # 统一设备
            data, target = data.to(device), target.to(device)
            output = model(data)
            # numpy不能处理 cuda向量，转回cpu处理
            targets.append(target.cpu())
            predictions.append(output.cpu())

    targets = torch.cat(targets)
    predictions = torch.cat(predictions, dim=0)
    return predictions, targets


def main():
    results_dir, models_dir, images_dir = create_directories()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device:{device}")

    # 加载数据
    df = pd.read_csv(r'E:\Anaconda3\envs\lstm\final_TSP_data（leftmajor）')  # 替换为您的数据路径
    print(df.head(5))

    # 绘制Vp历史
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(df['mile'], df['Vp'], color='darkorange', label='Vp')

    plt.title('Vp original')  # 设置标题
    plt.xlabel('mile')
    plt.ylabel('m/s')
    plt.legend(loc="upper right")
    plt.show()

    # 创建时间序列数据实例
    ts = TimeseriesData(df)
    print(ts.values()[-1])
    print(ts.labels()[-1])
    print(ts.__len__())

    # 创建监督学习时间序列数据实例
    sv = SupervisedTimeseriesData(dataset=df, lag=20)
    print(sv.values().shape)
    print(sv.labels().shape)
    print(sv.supervised_values.shape)
    print(sv.supervised_labels.shape)

    # 创建数据集
    svd = SupervisedTimeseriesDataset(dataset=df, lag=30)
    print(svd.__len__())
    print(len(svd.train_set), len(svd.val_set), len(svd.test_set))

    # 创建数据加载器
    train_loader = DataLoader(dataset=svd.train_set, batch_size=64, shuffle=True)
    valid_loader = DataLoader(dataset=svd.val_set, batch_size=64, shuffle=False)
    test_loader = DataLoader(dataset=svd.test_set, batch_size=64, shuffle=False)

    # 模型参数
    params = {
        'input_dim': 3,  # 修改为新数据集的特征数量
        'hidden_dim': 64,
        'lstm_layers': 2,
        'transformer_heads': 8,
        'transformer_layers': 1,
        'output_dim': 1,
        'dropout': .5,
    }

    # 创建模型
    lstm_tf = LSTMTransformer(**params).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=lstm_tf.parameters(), lr=0.0001)

    # 打印模型结构
    summary(model=lstm_tf, input_size=(64, 30, 3))

    # 训练模型
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(300):
        train_loss = train(model=lstm_tf, iterator=train_loader, optimizer=optimizer, criterion=criterion)
        val_loss = evaluate(model=lstm_tf, iterator=valid_loader, criterion=criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch: {epoch + 1:02}, Train MSELoss: {train_loss:.5f}, Val. MSELoss: {val_loss:.5f}')

        # 保存最佳权重
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(lstm_tf.state_dict(), models_dir / 'best_model.pth')
            print(f"Saved best model with validation loss: {best_val_loss:.5f}")

    # 保存最终模型
    torch.save({
        'epoch': epoch,
        'model_state_dict': lstm_tf.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
    }, models_dir / 'final_model.pth')

    # 绘制训练和验证损失
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSELoss')
    plt.title(' ')
    plt.legend()
    plt.grid(True)
    plt.savefig(images_dir / 'training_loss.png')
    plt.show()

    # 验证集预测
    val_pred, val_true = predict(lstm_tf, valid_loader, device)

    # 绘制验证集回归图
    plt.figure(figsize=(5, 5), dpi=100)
    sns.regplot(x=val_true.numpy(), y=val_pred.numpy(), scatter=True, marker="*", color='orange',
                line_kws={'color': 'red'})
    plt.title('Vel Prediction')
    plt.xlabel('Ture value (m/s)')
    plt.ylabel('Predicted value (m/s)')
    plt.savefig(images_dir / 'validation_regression.png')
    plt.show()

    # 输出验证集评估指标
    mae = mean_absolute_error(preds=val_pred, target=val_true)
    print(f"Mean Absolute Error: {mae:.5f}")

    mape = mean_absolute_percentage_error(preds=val_pred, target=val_true)
    print(f"Mean Absolute Percentage Error: {mape * 100:.4f}%")

    mse = mean_squared_error(preds=val_pred, target=val_true)
    print(f"Mean Squared Error: {mse:.4f}")

    nrmse = normalized_root_mean_squared_error(preds=val_pred, target=val_true)
    print(f"Normalized Root Mean Squared Error: {nrmse:.4f}")

    r2 = r2_score(preds=val_pred, target=val_true)
    print(f"R²: {r2:.4f}")

    # 测试集预测
    test_pred, test_true = predict(lstm_tf, test_loader, device)

    # 绘制测试集回归图
    plt.figure(figsize=(5, 5), dpi=100)
    sns.regplot(x=test_true.numpy(), y=test_pred.numpy(), scatter=True, marker="*", color='orange',
                line_kws={'color': 'red'})
    plt.title('Test Prediction')
    plt.xlabel('Ture value (m/s)')
    plt.ylabel('Predicted value (m/s)')
    plt.savefig(images_dir / 'test_regression.png')
    plt.show()

    # 输出测试集评估指标
    mae = mean_absolute_error(preds=test_pred, target=test_true)
    print(f"Mean Absolute Error: {mae:.5f}")

    mape = mean_absolute_percentage_error(preds=test_pred, target=test_true)
    print(f"Mean Absolute Percentage Error: {mape * 100:.4f}%")

    mse = mean_squared_error(preds=test_pred, target=test_true)
    print(f"Mean Squared Error: {mse:.4f}")

    nrmse = normalized_root_mean_squared_error(preds=test_pred, target=test_true)
    print(f"Normalized Root Mean Squared Error: {nrmse:.4f}")

    r2 = r2_score(preds=test_pred, target=test_true)
    print(f"R²: {r2:.4f}")

    # 获取原始里程数据用于绘图
    miles = df['mile'].values

    # 计算验证集和测试集对应的里程位置
    val_miles = miles[svd.val_idx]
    test_miles = miles[svd.test_idx]

    # 反归一化预测结果和真实值
    val_pred_denorm = svd.set.scaler_target.inverse_transform(val_pred.numpy())
    val_true_denorm = svd.set.scaler_target.inverse_transform(val_true.numpy())
    test_pred_denorm = svd.set.scaler_target.inverse_transform(test_pred.numpy())
    test_true_denorm = svd.set.scaler_target.inverse_transform(test_true.numpy())

    # 绘制预测结果对比图，以里程为横轴
    plt.figure(figsize=(20, 5))
    plt.plot(val_miles, val_true_denorm, 'b-', label='Ture value (m/s)')
    plt.plot(val_miles, val_pred_denorm, 'r--', label='Val. value (m/s)')
    plt.plot(test_miles, test_true_denorm, 'g-', label='Test value (m/s)')
    plt.plot(test_miles, test_pred_denorm, 'm--', label='Test value (m/s)')

    plt.axvline(val_miles[0], color='black', linestyle='-.')
    plt.axvline(test_miles[0], color='black', linestyle='-.')

    plt.text(val_miles[0], df['Vp'].max() * 0.9, 'val', fontsize=15)
    plt.text(test_miles[0], df['Vp'].max() * 0.9, 'test', fontsize=15)

    plt.xlabel('mile')
    plt.ylabel('Vp (m/s)')
    plt.title('Vp predict', fontsize=15)
    plt.legend()
    plt.grid(True)
    plt.savefig(images_dir / 'predictions_by_mile.png')
    plt.show()

    # 绘制最终预测结果
    plt.figure(figsize=(16, 8))
    plt.title('LSTM-Transformer Vp FinalPredict', fontsize=40, pad=20)

    # 绘制所有真实数据
    plt.plot(miles, df['Vp'].values, 'b-', label='Actual value (m/s)')

    # 绘制验证集和测试集预测结果
    plt.plot(val_miles, val_pred_denorm, 'r-', label='Val value (m/s)')
    plt.plot(test_miles, test_pred_denorm, 'g-', label='Test value (m/s)')

    # 添加分隔线
    plt.axvline(val_miles[0], color='red', linestyle='-.')
    plt.axvline(test_miles[0], color='red', linestyle='-.')

    plt.text(val_miles[0], df['Vp'].max() * 0.9, 'val', fontsize=20)
    plt.text(test_miles[0], df['Vp'].max() * 0.9, 'test', fontsize=20)

    plt.xlabel('Miles', fontsize=30)
    plt.ylabel('Vp (m/s)', fontsize=30)
    plt.legend(fontsize=15)
    plt.savefig(images_dir / 'final_predictions.png')
    plt.show()


if __name__ == "__main__":
    main()
