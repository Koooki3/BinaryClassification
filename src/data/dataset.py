import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

class BinaryClassificationDataset(Dataset):
    """二分类数据集类"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class DataProcessor:
    """数据处理类"""
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.data = None
        self.X = None
        self.y = None

    def load_data(self):
        """加载并验证数据"""
        try:
            self.data = pd.read_csv(self.config['data']['csv_file'])
            
            # 验证数据维度
            if self.data.shape[1] != 9:
                raise ValueError(f"数据列数应为9，实际为{self.data.shape[1]}")
            
            # 检查缺失值
            if self.data.isnull().any().any():
                self.logger.warning("数据中存在缺失值，将进行处理")
                self.data = self.handle_missing_values()
            
            # 检查异常值
            self._check_outliers()
            
            return True
        except Exception as e:
            self.logger.error(f"加载数据失败: {str(e)}")
            return False

    def handle_missing_values(self):
        """处理缺失值"""
        # 对数值型列使用中位数填充
        return self.data.fillna(self.data.median())

    def _check_outliers(self):
        """检查并记录异常值"""
        for col in self.data.columns[:-1]:  # 排除标签列
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.data[(self.data[col] < (Q1 - 1.5 * IQR)) | 
                               (self.data[col] > (Q3 + 1.5 * IQR))]
            if not outliers.empty:
                self.logger.warning(f"列 {col} 中发现 {len(outliers)} 个异常值")

    def preprocess_data(self):
        """数据预处理"""
        try:
            # 分离特征和标签
            self.X = self.data.iloc[:, :8].values
            self.y = self.data.iloc[:, -1].values

            # 验证标签值
            unique_labels = np.unique(self.y)
            if not np.array_equal(unique_labels, np.array([0, 1])):
                raise ValueError(f"标签值应为0和1，实际为: {unique_labels}")

            # 标准化特征
            self.X = self.scaler.fit_transform(self.X)

            return True
        except Exception as e:
            self.logger.error(f"数据预处理失败: {str(e)}")
            return False

    def create_data_loaders(self):
        """创建数据加载器"""
        try:
            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y,
                test_size=self.config['data']['test_size'],
                random_state=self.config['data']['random_seed'],
                stratify=self.y  # 确保划分后标签分布一致
            )

            # 创建数据集
            train_dataset = BinaryClassificationDataset(X_train, y_train)
            test_dataset = BinaryClassificationDataset(X_test, y_test)

            # 创建数据加载器
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=True,
                num_workers=0,  # Windows环境下建议设为0
                pin_memory=torch.cuda.is_available()  # GPU加速
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available()
            )

            return train_loader, test_loader
        except Exception as e:
            self.logger.error(f"创建数据加载器失败: {str(e)}")
            return None, None

    def get_input_size(self):
        """获取输入特征维度"""
        return self.X.shape[1]

    def get_scaler(self):
        """获取标准化器（用于预测时的数据转换）"""
        return self.scaler