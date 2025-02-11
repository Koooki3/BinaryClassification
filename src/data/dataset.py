import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

class RainfallDataset(Dataset):
    """降水量预测数据集类"""
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
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.data = None
        self.X = None
        self.y = None

    def load_data(self):
        """加载并验证数据"""
        try:
            self.data = pd.read_csv(self.config['data']['csv_file'])
            
            # 删除时间列
            if 'time' in self.data.columns:
                self.data = self.data.drop('time', axis=1)
            
            # 检查并处理缺失值
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
        # 使用插值方法处理缺失值
        return self.data.interpolate(method='linear', limit_direction='both')

    def _check_outliers(self):
        """检查并处理异常值"""
        for col in self.data.columns:
            if col == self.config['data']['target_column']:
                continue  # 跳过目标列
                
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.data[(self.data[col] < lower_bound) | 
                               (self.data[col] > upper_bound)]
            
            if not outliers.empty:
                self.logger.warning(f"列 {col} 中发现 {len(outliers)} 个异常值")
                # 使用截断方法处理异常值
                self.data[col] = self.data[col].clip(lower_bound, upper_bound)

    def preprocess_data(self):
        """数据预处理"""
        try:
            # 分离特征和标签
            feature_columns = self.config['data']['feature_columns']
            target_column = self.config['data']['target_column']
            
            self.X = self.data[feature_columns].values
            self.y = self.data[target_column].values.reshape(-1, 1)

            # 对降水量进行对数转换
            if self.config['data']['scaler']['target_transform'] == 'log1p':
                self.y = np.log1p(self.y)
                self.logger.info("对降水量进行了对数转换")

            # 标准化特征和目标
            self.X = self.scaler_X.fit_transform(self.X)
            self.y = self.scaler_y.fit_transform(self.y)

            return True
        except Exception as e:
            self.logger.error(f"数据预处理失败: {str(e)}")
            return False

    def create_data_loaders(self):
        """创建数据加载器"""
        try:
            # 首先划分训练集和测试集
            X_temp, X_test, y_temp, y_test = train_test_split(
                self.X, self.y,
                test_size=self.config['data']['test_size'],
                random_state=self.config['data']['random_seed']
            )
            
            # 再划分训练集和验证集
            valid_size = self.config['data']['valid_size']
            train_size = 1 - valid_size
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_temp, y_temp,
                test_size=valid_size / train_size,
                random_state=self.config['data']['random_seed']
            )

            # 创建数据集
            train_dataset = RainfallDataset(X_train, y_train)
            valid_dataset = RainfallDataset(X_valid, y_valid)
            test_dataset = RainfallDataset(X_test, y_test)

            # 创建数据加载器
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=True,
                num_workers=0,  # Windows环境下建议设为0
                pin_memory=torch.cuda.is_available()
            )
            
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available()
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available()
            )

            return train_loader, valid_loader, test_loader
        except Exception as e:
            self.logger.error(f"创建数据加载器失败: {str(e)}")
            return None, None, None

    def inverse_transform_target(self, y_scaled):
        """将标准化的目标值转换回原始刻度"""
        y_normalized = self.scaler_y.inverse_transform(y_scaled)
        if self.config['data']['scaler']['target_transform'] == 'log1p':
            return np.expm1(y_normalized)
        return y_normalized