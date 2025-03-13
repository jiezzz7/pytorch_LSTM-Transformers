import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

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

# 定义地质条件分类规则的常量
VP_THRESHOLD = 3500  # P波速度阈值
VS_THRESHOLD = 2000  # S波速度阈值
RATIO_THRESHOLD = 1.65  # Vp/Vs比值阈值
VP_CHANGE_THRESHOLD = 30  # P波变化率阈值
VS_CHANGE_THRESHOLD = 20  # S波变化率阈值


def create_directories():
    """创建必要的目录"""
    results_dir = Path('lstm_transformer_results')
    models_dir = Path('lstm_transformer_models')
    images_dir = Path('lstm_transformer_images')

    results_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)

    return results_dir, models_dir, images_dir


def create_class_labels(df):
    """根据波速特征创建分类标签
    类别0: 流体存在 - S波反射弱/不存在，P波显示低速度
    类别1: 围岩破碎 - P波快速变化，S波变化小
    类别2: 富水围岩 - Vp/Vs突然增大(P波速度增加，S波速度减少)
    类别3: 正常
    """
    # 创建新列存储类别标签
    df['class_label'] = 3  # 默认为正常类别

    # 计算波速变化率
    df['Vp_change'] = df['Vp'].diff().abs()
    df['Vs_change'] = df['Vs'].diff().abs()

    # 应用分类规则 - 使用固定阈值而不是百分位数
    # 规则1: 流体存在 - S波反射弱，P波显示低速度
    mask_fluid = (df['Vs'] < VS_THRESHOLD) & (df['Vp'] < VP_THRESHOLD)
    df.loc[mask_fluid, 'class_label'] = 0

    # 规则2: 围岩破碎 - P波快速变化，S波变化小
    mask_broken = (df['Vp_change'] > VP_CHANGE_THRESHOLD) & (df['Vs_change'] < VS_CHANGE_THRESHOLD)
    df.loc[mask_broken, 'class_label'] = 1

    # 规则3: 富水围岩 - Vp/Vs突然增大
    mask_water = ((df['Vp/Vs'] > RATIO_THRESHOLD) &
                  (df['Vp'] > df['Vp'].shift(1)) &
                  (df['Vs'] < df['Vs'].shift(1)))
    df.loc[mask_water, 'class_label'] = 2

    # 打印各类别的样本数量
    class_counts = df['class_label'].value_counts().sort_index()
    print("类别分布:")
    for i, count in enumerate(class_counts):
        if i in class_counts.index:
            print(f"类别 {i}: {count} 样本")

    # 删除辅助列
    df.drop(['Vp_change', 'Vs_change'], axis=1, inplace=True)

    return df


def display_data_info(df):
    """显示数据基本信息"""
    print(f"数据集形状: {df.shape}")
    print("\n数据集前5行:")
    print(df.head())
    print("\n数据集描述统计:")
    print(df.describe())
    print("\n数据类型:")
    print(df.dtypes)
    print("\n缺失值统计:")
    print(df.isna().sum())


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
        self._class_labels_cache = None

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

    def class_labels(self):
        """获取分类标签"""
        if self._class_labels_cache is None:
            self._class_labels_cache = self.dataset['class_label'].values
        return self._class_labels_cache

    def inverse_transform_labels(self, scaled_labels, target_col='Vp'):
        """将归一化的标签值转换回原始尺度"""
        return self.scaler_targets[target_col].inverse_transform(scaled_labels)


# 定义监督学习时间序列数据类
class SupervisedTimeseriesData(TimeseriesData):
    def __init__(self, dataset: pd.DataFrame, lag: int = 10):
        super(SupervisedTimeseriesData, self).__init__(dataset=dataset)
        self.lag = lag
        self._supervised_values_cache = None
        self._supervised_labels_cache = {}
        self._supervised_class_labels_cache = None

    @property
    def supervised_values(self):
        if self._supervised_values_cache is None:
            self._supervised_values_cache = self._compute_supervised_values()
        return self._supervised_values_cache

    def supervised_labels(self, target_col='Vp'):
        if target_col not in self._supervised_labels_cache:
            self._supervised_labels_cache[target_col] = self._compute_supervised_labels(target_col)
        return self._supervised_labels_cache[target_col]

    @property
    def supervised_class_labels(self):
        """获取用于监督学习的分类标签"""
        if self._supervised_class_labels_cache is None:
            self._supervised_class_labels_cache = torch.tensor(
                self.class_labels()[self.lag:], dtype=torch.long)
        return self._supervised_class_labels_cache

    def _compute_supervised_values(self):
        x = [self.values()[i:i + self.lag] for i in range(self.__len__() - self.lag)]
        return torch.tensor(np.array(x), dtype=torch.float)

    def _compute_supervised_labels(self, target_col='Vp'):
        return torch.tensor(self.labels(target_col)[self.lag:], dtype=torch.float)


def create_stratified_indices(supervised_data):
    """创建分层抽样的训练、验证和测试索引"""
    # 获取所有类别标签
    all_labels = supervised_data.supervised_class_labels.numpy()

    # 先分割训练集和临时集 (60% 训练)
    train_idx, temp_idx = train_test_split(
        np.arange(len(all_labels)),
        test_size=0.4,
        random_state=42,
        stratify=all_labels
    )

    # 再将临时集分割为验证集和测试集 (20% 验证, 20% 测试)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=42,
        stratify=all_labels[temp_idx]
    )

    return list(train_idx), list(val_idx), list(test_idx)


# 定义监督学习时间序列数据集
class SupervisedTimeseriesDataset(Dataset):
    def __init__(self, dataset, target_col='class', lag=30):
        super(SupervisedTimeseriesDataset, self).__init__()
        self.target_col = target_col
        self.set = SupervisedTimeseriesData(dataset=dataset, lag=lag)

        if target_col == 'class':
            self.dataset = TensorDataset(
                self.set.supervised_values,
                self.set.supervised_class_labels
            )
        else:
            self.dataset = TensorDataset(
                self.set.supervised_values,
                self.set.supervised_labels(target_col)
            )

        # 使用分层抽样创建训练、验证和测试索引
        if target_col == 'class':
            self.train_idx, self.val_idx, self.test_idx = create_stratified_indices(self.set)
        else:
            # 保持原有的训练/验证/测试分割
            self.train_idx = list(range(self.__len__() * 3 // 5))
            self.val_idx = list(range(self.__len__() * 3 // 5, self.__len__() * 4 // 5))
            self.test_idx = list(range(self.__len__() * 4 // 5, self.__len__()))

        # 创建子数据集
        self.train_set = Subset(self.dataset, self.train_idx)
        self.val_set = Subset(self.dataset, self.val_idx)
        self.test_set = Subset(self.dataset, self.test_idx)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


# 分类模型结构
class ClassificationBiLSTMTransformer(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 lstm_layers,
                 transformer_heads,
                 transformer_layers,
                 num_classes=4,  # 默认4个类别：流体存在、围岩破碎、富水围岩、正常
                 dropout=0.5
                 ):
        super(ClassificationBiLSTMTransformer, self).__init__()
        # BiLSTM层
        self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers,
                            dropout=dropout, batch_first=True,
                            bidirectional=True)

        # Transformer层
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim * 2,  # 双向LSTM输出维度
            nhead=transformer_heads,
            dim_feedforward=hidden_dim * 4,  # 增加feedforward维度
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_encoder_layer,
            num_layers=transformer_layers)

        # 分类输出层
        self.fc_classifier = nn.Linear(hidden_dim * 2, num_classes)

        # 回归输出层
        self.fc_vp = nn.Linear(hidden_dim * 2, 1)
        self.fc_vs = nn.Linear(hidden_dim * 2, 1)
        self.fc_ratio = nn.Linear(hidden_dim * 2, 1)
        self.fc_mile = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, target='class'):
        # 初始化LSTM隐藏状态和记忆状态
        h0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)

        # 双向LSTM处理
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Transformer编码器处理BiLSTM输出
        transformer_out = self.transformer_encoder(lstm_out)

        # 获取最后一个时间步的输出
        final_out = transformer_out[:, -1, :]

        # 根据目标选择对应的输出层
        if target == 'class':
            return self.fc_classifier(final_out)  # 分类输出
        elif target == 'Vp':
            return self.fc_vp(final_out)
        elif target == 'Vs':
            return self.fc_vs(final_out)
        elif target == 'Ratio':
            return self.fc_ratio(final_out)
        elif target == 'mile':
            return self.fc_mile(final_out)
        else:
            raise ValueError(f"Unknown target: {target}")


# 训练函数
def train(model, iterator, optimizer, criterion, target_col='class',device=device):
    model.train()
    epoch_loss = 0

    # 用于分类任务的指标
    if target_col == 'class':
        correct = 0
        total = 0

    for batch_idx, (data, target) in enumerate(iterable=iterator):
        # 统一设备
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data, target=target_col)

        # 根据任务类型计算损失
        if target_col == 'class':
            loss = criterion(output, target)
            # 计算准确率
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        else:
            loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if target_col == 'class':
        accuracy = 100 * correct / total
        return epoch_loss / len(iterator), accuracy
    else:
        return epoch_loss / len(iterator)


# 评估函数
def evaluate(model, iterator, criterion, target_col='class',device=device):
    model.eval()
    epoch_loss = 0

    # 用于分类任务的指标
    if target_col == 'class':
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(iterable=iterator):
            # 统一设备
            data, target = data.to(device), target.to(device)
            output = model(data, target=target_col)

            # 根据任务类型计算损失和指标
            if target_col == 'class':
                loss = criterion(output, target)
                # 计算准确率
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                # 收集预测和目标用于计算混淆矩阵
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
            else:
                loss = criterion(output, target)

            epoch_loss += loss.item()

        if target_col == 'class':
            accuracy = 100 * correct / total
            return epoch_loss / len(iterator), accuracy, all_preds, all_targets
        else:
            return epoch_loss / len(iterator)


def evaluate_and_visualize(model, dataset, device, images_dir):
    """评估模型并可视化结果"""
    model.eval()

    # 创建验证和测试数据加载器
    val_loader = DataLoader(dataset=dataset.val_set, batch_size=64, shuffle=False)
    test_loader = DataLoader(dataset=dataset.test_set, batch_size=64, shuffle=False)

    # 获取预测结果
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data, target='class')
            _, predicted = torch.max(outputs, 1)
            val_preds.extend(predicted.cpu().numpy())
            val_targets.extend(target.cpu().numpy())

    # 分析预测的类别分布
    unique_classes = np.unique(np.array(val_targets + val_preds))
    print(f"数据中出现的类别: {unique_classes}")

    # 确定报告中使用的类别
    class_names = ['Fluid', 'BrokenState', 'WaterRock', 'Normal']
    used_class_names = [class_names[i] for i in unique_classes]

    print("\n分类报告:")
    # 使用labels参数指定实际存在的类别标签
    print(classification_report(val_targets, val_preds,
                                target_names=used_class_names,
                                labels=unique_classes))
    # 绘制混淆矩阵
    cm = confusion_matrix(val_targets, val_preds, labels=unique_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=used_class_names, yticklabels=used_class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(images_dir / 'confusion_matrix.png')
    plt.close()

    return {"val_preds": val_preds, "val_targets": val_targets}


def plot_training_losses(train_losses, val_losses, target_col, images_dir):
    """绘制训练和验证损失曲线"""
    plt.figure(figsize=(10, 6))

    for col in target_col:
        plt.plot(train_losses[col], label=f'Train Loss - {col}')
        plt.plot(val_losses[col], label=f'Val Loss - {col}')

    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(images_dir / 'loss_curves.png')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, images_dir, title='Confusion Matrix'):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(images_dir / f'{title.lower().replace(" ", "_")}.png')
    plt.close()


def main():
    """主函数，整合所有功能"""
    results_dir, models_dir, images_dir = create_directories()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device:{device}")

    # 加载数据
    df = pd.read_csv(r'E:\Anaconda3\envs\lstm\final_TSP_data（leftmajor）')  # 替换为实际数据文件路径
    display_data_info(df)

    # 创建分类标签
    df = create_class_labels(df)

    # 准备数据集
    lag = 30  # 使用前30个时间步预测
    target_cols = ['Vp', 'Vs', 'Ratio', 'class']
    dataset = SupervisedTimeseriesDataset(df, target_col='class', lag=lag)

    # 创建数据加载器
    batch_size = 64
    train_loader = DataLoader(dataset=dataset.train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=dataset.val_set, batch_size=batch_size, shuffle=False)

    # 创建模型
    params = {
        'input_dim': 3,  # mile, Vs, Vp/Vs
        'hidden_dim': 64,
        'lstm_layers': 2,
        'transformer_heads': 8,
        'transformer_layers': 1,
        'num_classes': 4,  # 4个类别
        'dropout': 0.5,
    }

    model = ClassificationBiLSTMTransformer(**params).to(device)

    # 打印模型结构
    summary(model=model, input_size=(batch_size, lag, 3))

    # 定义损失函数和优化器
    classification_criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    # 训练模型
    train_losses = {col: [] for col in target_cols}
    val_losses = {col: [] for col in target_cols}
    best_val_losses = {col: float('inf') for col in target_cols}
    best_accuracy = 0

    epochs = 125
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # 首先训练分类任务
        train_loss, train_acc = train(model, train_loader, optimizer, classification_criterion, target_col='class',
                                      device=device)
        val_loss, val_acc, val_preds, val_targets = evaluate(model, val_loader, classification_criterion,
                                                             target_col='class', device=device)

        train_losses['class'].append(train_loss)
        val_losses['class'].append(val_loss)

        print(
            f"分类任务 - Train Loss: {train_loss:.5f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.5f}, Val Acc: {val_acc:.2f}%")

        # 如果验证精度提高，保存模型
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), models_dir / 'best_model_class.pth')
            print(f"保存最佳分类模型，验证精度: {val_acc:.2f}%")

        # 训练回归任务（可选，多任务学习）
        for col in ['Vp', 'Vs', 'Ratio']:
            # 为当前任务创建数据加载器
            regression_dataset = SupervisedTimeseriesDataset(df, target_col=col, lag=lag)
            regression_train_loader = DataLoader(dataset=regression_dataset.train_set, batch_size=batch_size,
                                                 shuffle=True)
            regression_val_loader = DataLoader(dataset=regression_dataset.val_set, batch_size=batch_size, shuffle=False)

            # 训练回归任务
            train_loss = train(model, regression_train_loader, optimizer, regression_criterion, target_col=col,
                               device=device)
            val_loss = evaluate(model, regression_val_loader, regression_criterion, target_col=col, device=device)

            train_losses[col].append(train_loss)
            val_losses[col].append(val_loss)

            print(f"{col} - Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}")

            # 如果验证损失降低，保存模型
            if val_loss < best_val_losses[col]:
                best_val_losses[col] = val_loss
                torch.save(model.state_dict(), models_dir / f'best_model_{col}.pth')
                print(f"保存最佳{col}模型，验证损失: {best_val_losses[col]:.5f}")

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

    # 加载最佳分类模型
    model.load_state_dict(torch.load(models_dir / 'best_model_class.pth'))

    # 评估模型
    results = evaluate_and_visualize(model, dataset, device, images_dir)

    print("训练完成！")


if __name__ == "__main__":
    main()