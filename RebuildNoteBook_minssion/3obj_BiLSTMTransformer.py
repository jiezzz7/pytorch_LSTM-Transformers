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
    """创建必要的目录"""
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
        self.scaler_targets = {
            'Vp': MinMaxScaler(),
            'Vs': MinMaxScaler(),
            'Ratio': MinMaxScaler(),  # 使用Ratio代替Vp/Vs
            'mile': MinMaxScaler()
        }
        self._values_cache = None
        self._labels_cache = {}

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

    def labels(self, target_col='Vp'):
        # 处理Ratio特殊情况
        col_name = 'Vp/Vs' if target_col == 'Ratio' else target_col

        if target_col not in self._labels_cache:
            targets = self.dataset[col_name].values.reshape(-1, 1)
            self._labels_cache[target_col] = self.scaler_targets[target_col].fit_transform(targets)
        return self._labels_cache[target_col]


# 定义监督学习时间序列数据类
class SupervisedTimeseriesData(TimeseriesData):
    def __init__(self, dataset: pd.DataFrame, lag: int = 10):
        super(SupervisedTimeseriesData, self).__init__(dataset=dataset)
        self.lag = lag
        self._supervised_values_cache = None
        self._supervised_labels_cache = {}

    @property
    def supervised_values(self):
        if self._supervised_values_cache is None:
            self._supervised_values_cache = self._compute_supervised_values()
        return self._supervised_values_cache

    def supervised_labels(self, target_col='Vp'):
        if target_col not in self._supervised_labels_cache:
            self._supervised_labels_cache[target_col] = self._compute_supervised_labels(target_col)
        return self._supervised_labels_cache[target_col]

    def _compute_supervised_values(self):
        x = [self.values()[i:i + self.lag] for i in range(self.__len__() - self.lag)]
        return torch.tensor(np.array(x), dtype=torch.float)

    def _compute_supervised_labels(self, target_col='Vp'):
        return torch.tensor(self.labels(target_col)[self.lag:], dtype=torch.float)


# 定义监督学习时间序列数据集
class SupervisedTimeseriesDataset(Dataset):
    def __init__(self, dataset, target_col='Vp', lag=30):
        super(SupervisedTimeseriesDataset, self).__init__()
        self.target_col = target_col
        self.set = SupervisedTimeseriesData(dataset=dataset, lag=lag)
        self.dataset = TensorDataset(self.set.supervised_values, self.set.supervised_labels(target_col))
        self.train_idx = list(range(self.__len__() * 3 // 5))
        self.val_idx = list(range(self.__len__() * 3 // 5, self.__len__() * 4 // 5))
        self.test_idx = list(range(self.__len__() * 4 // 5, self.__len__()))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.set.supervised_values[index], self.set.supervised_labels(self.target_col)[index]

    @property
    def train_set(self):
        return Subset(self.dataset, indices=self.train_idx)

    @property
    def val_set(self):
        return Subset(self.dataset, indices=self.val_idx)

    @property
    def test_set(self):
        return Subset(self.dataset, indices=self.test_idx)


# 定义多输出LSTM-Transformer模型
class MultiOutputBiLSTMTransformer(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 lstm_layers,
                 transformer_heads,
                 transformer_layers,
                 dropout=0.5
                 ):
        super(MultiOutputBiLSTMTransformer, self).__init__()
        # 共享的BiLSTM和Transformer层
        self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers,
                            dropout=dropout, batch_first=True,
                            bidirectional=True)  # 改为双向LSTM

        # 注意：双向LSTM输出的维度是hidden_dim*2
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim * 2,  # 调整为双向LSTM输出维度
            nhead=transformer_heads,  # 确保transformer_heads能被hidden_dim*2整除
            dim_feedforward=hidden_dim * 4,  # 增加feedforward维度
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_encoder_layer,
            num_layers=transformer_layers)

        # 各个任务的输出层，调整为接收hidden_dim*2维度的输入
        self.fc_vp = nn.Linear(hidden_dim * 2, 1)
        self.fc_vs = nn.Linear(hidden_dim * 2, 1)
        self.fc_ratio = nn.Linear(hidden_dim * 2, 1)
        self.fc_mile = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, target='Vp'):
        # 初始化LSTM隐藏状态和记忆状态
        h0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)

        # 双向LSTM处理
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Transformer编码器处理BiLSTM输出
        transformer_out = self.transformer_encoder(lstm_out)

        # 根据目标选择对应的输出层
        if target == 'Vp':
            return self.fc_vp(transformer_out[:, -1, :])
        elif target == 'Vs':
            return self.fc_vs(transformer_out[:, -1, :])
        elif target == 'Ratio':
            return self.fc_ratio(transformer_out[:, -1, :])
        elif target == 'mile':
            return self.fc_mile(transformer_out[:, -1, :])
        else:
            raise ValueError(f"Unknown target: {target}")


# 定义训练函数
def train(model, iterator, optimizer, criterion, target_col='Vp'):
    model.train()
    epoch_loss = 0

    for batch_idx, (data, target) in enumerate(iterable=iterator):
        # 统一设备
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data, target=target_col)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(iterator)

    return avg_loss


# 定义评估函数
def evaluate(model, iterator, criterion, target_col='Vp'):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(iterable=iterator):
            # 统一设备
            data, target = data.to(device), target.to(device)
            output = model(data, target=target_col)
            loss = criterion(output, target)
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(iterator)

        return avg_loss


# 定义预测函数
def predict(model, iterator, device, target_col='Vp'):
    model.eval()
    targets = []
    predictions = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(iterable=iterator):
            # 统一设备
            data, target = data.to(device), target.to(device)
            output = model(data, target=target_col)
            # numpy不能处理 cuda向量，转回cpu处理
            targets.append(target.cpu())
            predictions.append(output.cpu())

    targets = torch.cat(targets)
    predictions = torch.cat(predictions, dim=0)
    return predictions, targets


def display_data_info(df):
    """显示数据加载结果"""
    print("\n===== 数据加载完成 =====")
    print(f"数据集大小: {len(df)} 行")
    print("\n数据集前5行:")
    print(df.head(5))

    print("\n数据集统计信息:")
    print(df.describe())

    # 绘制各列数据分布
    plt.figure(figsize=(20, 15))

    # 绘制mile vs Vp
    plt.subplot(2, 2, 1)
    plt.plot(df['mile'], df['Vp'], 'b-')
    plt.title('Mile vs Vp')
    plt.xlabel('Mile')
    plt.ylabel('Vp (m/s)')
    plt.grid(True)

    # 绘制mile vs Vs
    plt.subplot(2, 2, 2)
    plt.plot(df['mile'], df['Vs'], 'g-')
    plt.title('Mile vs Vs')
    plt.xlabel('Mile')
    plt.ylabel('Vs (m/s)')
    plt.grid(True)

    # 绘制mile vs Vp/Vs (Ratio)
    plt.subplot(2, 2, 3)
    plt.plot(df['mile'], df['Vp/Vs'], 'r-')
    plt.title('Mile vs Ratio (Vp/Vs)')
    plt.xlabel('Mile')
    plt.ylabel('Ratio')
    plt.grid(True)

    # 绘制Vp vs Vs散点图
    plt.subplot(2, 2, 4)
    plt.scatter(df['Vp'], df['Vs'], alpha=0.5)
    plt.title('Vp vs Vs')
    plt.xlabel('Vp (m/s)')
    plt.ylabel('Vs (m/s)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_training_losses(train_losses, val_losses, target_cols, images_dir):
    """绘制训练和验证损失"""
    plt.figure(figsize=(15, 10))

    for i, col in enumerate(target_cols):
        plt.subplot(2, 2, i + 1)
        plt.plot(train_losses[col], label=f'{col} Training Loss')
        plt.plot(val_losses[col], label=f'{col} Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSELoss')
        plt.title(f'{col} Training Progress')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(images_dir / 'training_losses.png')
    plt.show()


def evaluate_and_plot(model, datasets, df, target_cols, device, images_dir):
    """评估模型并绘制预测结果"""
    results = {}
    metrics = {}

    for col in target_cols:
        # 创建数据加载器
        valid_loader = DataLoader(dataset=datasets[col].val_set, batch_size=64, shuffle=False)
        test_loader = DataLoader(dataset=datasets[col].test_set, batch_size=64, shuffle=False)

        # 预测
        val_pred, val_true = predict(model, valid_loader, device, target_col=col)
        test_pred, test_true = predict(model, test_loader, device, target_col=col)

        # 获取原始里程数据
        miles = df['mile'].values

        # 计算验证集和测试集对应的里程位置
        val_miles = miles[datasets[col].val_idx]
        test_miles = miles[datasets[col].test_idx]

        # 反归一化预测结果和真实值
        val_pred_denorm = datasets[col].set.scaler_targets[col].inverse_transform(val_pred.numpy())
        val_true_denorm = datasets[col].set.scaler_targets[col].inverse_transform(val_true.numpy())
        test_pred_denorm = datasets[col].set.scaler_targets[col].inverse_transform(test_pred.numpy())
        test_true_denorm = datasets[col].set.scaler_targets[col].inverse_transform(test_true.numpy())

        # 绘制验证集回归图
        plt.figure(figsize=(5, 5), dpi=100)
        sns.regplot(x=val_true.numpy(), y=val_pred.numpy(), scatter=True, marker="*", color='orange',
                    line_kws={'color': 'red'})
        plt.title(f'{col} Validation Prediction')
        plt.xlabel('True value')
        plt.ylabel('Predicted value')
        plt.savefig(images_dir / f'{col}_validation_regression.png')
        plt.close()

        # 绘制测试集回归图
        plt.figure(figsize=(5, 5), dpi=100)
        sns.regplot(x=test_true.numpy(), y=test_pred.numpy(), scatter=True, marker="*", color='orange',
                    line_kws={'color': 'red'})
        plt.title(f'{col} Test Prediction')
        plt.xlabel('True value')
        plt.ylabel('Predicted value')
        plt.savefig(images_dir / f'{col}_test_regression.png')
        plt.close()

        # 绘制预测结果对比图，以里程为横轴
        plt.figure(figsize=(20, 5))
        plt.plot(val_miles, val_true_denorm, 'b-', label=f'True {col}')
        plt.plot(val_miles, val_pred_denorm, 'r--', label=f'Predicted {col} (Val)')
        plt.plot(test_miles, test_true_denorm, 'g-', label=f'True {col}')
        plt.plot(test_miles, test_pred_denorm, 'm--', label=f'Predicted {col} (Test)')

        plt.axvline(val_miles[0], color='black', linestyle='-.')
        plt.axvline(test_miles[0], color='black', linestyle='-.')

        plt.text(val_miles[0], np.max(val_true_denorm) * 0.9, 'val', fontsize=15)
        plt.text(test_miles[0], np.max(test_true_denorm) * 0.9, 'test', fontsize=15)

        plt.xlabel('mile')
        plt.ylabel(col)
        plt.title(f'{col} prediction', fontsize=15)
        plt.legend()
        plt.grid(True)
        plt.savefig(images_dir / f'{col}_predictions_by_mile.png')
        plt.close()

        # 绘制最终预测结果
        plt.figure(figsize=(16, 8))
        plt.title(f'LSTM-Transformer {col} Final Prediction', fontsize=40, pad=20)

        # 获取原始数据列名
        col_name = 'Vp/Vs' if col == 'Ratio' else col

        # 绘制所有真实数据
        plt.plot(miles, df[col_name].values, 'b-', label=f'Actual {col}')

        # 绘制验证集和测试集预测结果
        plt.plot(val_miles, val_pred_denorm, 'r-', label=f'Val {col}')
        plt.plot(test_miles, test_pred_denorm, 'g-', label=f'Test {col}')

        # 添加分隔线
        plt.axvline(val_miles[0], color='red', linestyle='-.')
        plt.axvline(test_miles[0], color='red', linestyle='-.')

        plt.text(val_miles[0], np.max(df[col_name].values) * 0.9, 'val', fontsize=20)
        plt.text(test_miles[0], np.max(df[col_name].values) * 0.9, 'test', fontsize=20)

        plt.xlabel('Miles', fontsize=30)
        plt.ylabel(col, fontsize=30)
        plt.legend(fontsize=15)
        plt.savefig(images_dir / f'{col}_final_predictions.png')
        plt.close()

        # 计算评估指标
        metrics[col] = {
            'validation': {
                'mae': mean_absolute_error(preds=val_pred, target=val_true).item(),
                'mape': mean_absolute_percentage_error(preds=val_pred, target=val_true).item() * 100,
                'mse': mean_squared_error(preds=val_pred, target=val_true).item(),
                'nrmse': normalized_root_mean_squared_error(preds=val_pred, target=val_true).item(),
                'r2': r2_score(preds=val_pred, target=val_true).item()
            },
            'test': {
                'mae': mean_absolute_error(preds=test_pred, target=test_true).item(),
                'mape': mean_absolute_percentage_error(preds=test_pred, target=test_true).item() * 100,
                'mse': mean_squared_error(preds=test_pred, target=test_true).item(),
                'nrmse': normalized_root_mean_squared_error(preds=test_pred, target=test_true).item(),
                'r2': r2_score(preds=test_pred, target=test_true).item()
            }
        }

        results[col] = {
            'val_miles': val_miles,
            'test_miles': test_miles,
            'val_true': val_true_denorm,
            'val_pred': val_pred_denorm,
            'test_true': test_true_denorm,
            'test_pred': test_pred_denorm
        }

    # 绘制综合预测结果
    plot_combined_predictions(results, df, target_cols, images_dir)

    # 输出评估指标
    print_evaluation_metrics(metrics)

    return results, metrics


def plot_combined_predictions(results, df, target_cols, images_dir):
    """将所有特征的预测结果绘制到一个图中"""
    plt.figure(figsize=(20, 15))

    colors = {
        'Vp': ['blue', 'red'],
        'Vs': ['green', 'orange'],
        'Ratio': ['purple', 'brown'],
        'mile': ['gray', 'pink']
    }

    for i, col in enumerate(target_cols):
        plt.subplot(2, 2, i + 1)
        result = results[col]

        # 绘制验证集
        plt.plot(result['val_miles'], result['val_true'], '-', color=colors[col][0],
                 label=f'{col} True')
        plt.plot(result['val_miles'], result['val_pred'], '--', color=colors[col][1],
                 label=f'{col} Pred')

        # 绘制测试集
        plt.plot(result['test_miles'], result['test_true'], '-', color=colors[col][0])
        plt.plot(result['test_miles'], result['test_pred'], '--', color=colors[col][1])

        # 添加分隔线
        plt.axvline(result['val_miles'][0], color='black', linestyle='-.')
        plt.axvline(result['test_miles'][0], color='black', linestyle='-.')

        # 获取原始数据列名
        col_name = 'Vp/Vs' if col == 'Ratio' else col

        # 添加文本标记
        plt.text(result['val_miles'][0], np.max(df[col_name].values) * 0.9, 'Val', fontsize=12)
        plt.text(result['test_miles'][0], np.max(df[col_name].values) * 0.9, 'Test', fontsize=12)

        plt.xlabel('Mile')
        plt.ylabel(col)
        plt.title(f'{col} Prediction')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(images_dir / 'combined_predictions.png')
    plt.show()


def print_evaluation_metrics(metrics):
    """打印评估指标"""
    print("\n===== 模型评估指标 =====")

    for col, metric in metrics.items():
        print(f"\n----- {col} 评估指标 -----")

        print("\nValidation Set Metrics:")
        print(f"Mean Absolute Error: {metric['validation']['mae']:.5f}")
        print(f"Mean Absolute Percentage Error: {metric['validation']['mape']:.4f}%")
        print(f"Mean Squared Error: {metric['validation']['mse']:.4f}")
        print(f"Normalized Root Mean Squared Error: {metric['validation']['nrmse']:.4f}")
        print(f"R²: {metric['validation']['r2']:.4f}")

        print("\nTest Set Metrics:")
        print(f"Mean Absolute Error: {metric['test']['mae']:.5f}")
        print(f"Mean Absolute Percentage Error: {metric['test']['mape']:.4f}%")
        print(f"Mean Squared Error: {metric['test']['mse']:.4f}")
        print(f"Normalized Root Mean Squared Error: {metric['test']['nrmse']:.4f}")
        print(f"R²: {metric['test']['r2']:.4f}")


def main():
    """主函数，整合所有功能"""
    results_dir, models_dir, images_dir = create_directories()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device:{device}")

    # 加载数据
    df = pd.read_csv(r'E:\Anaconda3\envs\lstm\final_TSP_data（leftmajor）')  # 替换为您的数据路径

    # 显示数据加载结果
    display_data_info(df)

    # 设置目标列
    target_cols = ['Vp', 'Vs', 'Ratio', 'mile']

    # 创建数据集
    datasets = {}
    for col in target_cols:
        datasets[col] = SupervisedTimeseriesDataset(dataset=df, target_col=col, lag=30)

    # 创建数据加载器
    dataloaders = {}
    for col in target_cols:
        dataloaders[col] = {
            'train': DataLoader(dataset=datasets[col].train_set, batch_size=64, shuffle=True),
            'val': DataLoader(dataset=datasets[col].val_set, batch_size=64, shuffle=False),
            'test': DataLoader(dataset=datasets[col].test_set, batch_size=64, shuffle=False)
        }

    # 模型参数
    params = {
        'input_dim': 3,  # 保持原始特征数量
        'hidden_dim': 64,
        'lstm_layers': 2,
        'transformer_heads': 8,
        'transformer_layers': 1,
        'dropout': .5,
    }

    # 创建模型
    model = MultiOutputBiLSTMTransformer(**params).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

    # 打印模型结构
    summary(model=model, input_size=(64, 30, 3))

    # 训练模型
    train_losses = {col: [] for col in target_cols}
    val_losses = {col: [] for col in target_cols}
    best_val_losses = {col: float('inf') for col in target_cols}

    epochs = 300
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        for col in target_cols:
            train_loss = train(model=model, iterator=dataloaders[col]['train'],
                               optimizer=optimizer, criterion=criterion, target_col=col)
            val_loss = evaluate(model=model, iterator=dataloaders[col]['val'],
                                criterion=criterion, target_col=col)

            train_losses[col].append(train_loss)
            val_losses[col].append(val_loss)

            print(f"{col} - Train MSELoss: {train_loss:.5f}, Val. MSELoss: {val_loss:.5f}")

            # 保存最佳权重
            if val_loss < best_val_losses[col]:
                best_val_losses[col] = val_loss
                torch.save(model.state_dict(), models_dir / f'best_model_{col}.pth')
                print(f"Saved best {col} model with validation loss: {best_val_losses[col]:.5f}")

    # 保存最终模型
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, models_dir / 'final_model.pth')

    # 绘制训练和验证损失
    plot_training_losses(train_losses, val_losses, target_cols, images_dir)

    # 加载最佳模型
    model.load_state_dict(torch.load(models_dir / 'best_model_Vp.pth'))

    # 评估模型并绘制预测结果
    results, metrics = evaluate_and_plot(model, datasets, df, target_cols, device, images_dir)


if __name__ == "__main__":
    main()
