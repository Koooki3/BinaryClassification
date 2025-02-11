import logging
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns

def load_data(file_path):
    """加载数据集"""
    import pandas as pd
    return pd.read_csv(file_path)

def split_data(data, target_column, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def evaluate_model(model, X_test, y_test):
    from sklearn.metrics import accuracy_score, classification_report
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report

def evaluate_regression_model(y_true, y_pred):
    """评估回归模型性能"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    return metrics

def save_model(model, file_path):
    import joblib
    joblib.dump(model, file_path)

def load_model(file_path):
    import joblib
    return joblib.load(file_path)

def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """设置日志记录器"""
    logger = logging.getLogger(name)
    
    # 避免重复添加处理器
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levellevel)s - %(message)s')
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        
        logger.setLevel(level)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    
    return logger

def save_metrics(metrics: dict, filepath: str) -> bool:
    """保存训练指标"""
    try:
        # 将所有 numpy 类型转换为 Python 原生类型
        converted_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, list):
                # 如果是列表，转换列表中的每个元素
                converted_metrics[key] = [float(v) if hasattr(v, 'item') else v for v in value]
            else:
                # 如果是单个值
                converted_metrics[key] = float(value) if hasattr(value, 'item') else value
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(converted_metrics, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        logging.error(f"保存指标失败: {str(e)}")
        return False

def load_metrics(filepath: str) -> dict:
    """加载训练指标"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"加载指标失败: {str(e)}")
        return {}

def plot_training_curves(metrics: dict, save_path: str = None):
    """绘制训练曲线"""
    plt.figure(figsize=(15, 5))
    
    # 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(metrics['train_loss'], label='训练损失')
    plt.plot(metrics['val_loss'], label='验证损失')
    plt.title('损失曲线')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    # RMSE曲线
    plt.subplot(1, 3, 2)
    plt.plot(metrics['train_rmse'], label='训练RMSE')
    plt.plot(metrics['val_rmse'], label='验证RMSE')
    plt.title('RMSE曲线')
    plt.xlabel('轮次')
    plt.ylabel('RMSE (mm)')
    plt.legend()
    plt.grid(True)
    
    # R²曲线
    plt.subplot(1, 3, 3)
    plt.plot(metrics['train_r2'], label='训练R²')
    plt.plot(metrics['val_r2'], label='验证R²')
    plt.title('R²曲线')
    plt.xlabel('轮次')
    plt.ylabel('R²')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_prediction_scatter(y_true, y_pred, save_path: str = None):
    """绘制预测散点图"""
    plt.figure(figsize=(10, 6))
    
    # 创建散点图
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # 添加对角线
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想预测线')
    
    plt.title('降水量预测值与真实值对比')
    plt.xlabel('真实降水量 (mm)')
    plt.ylabel('预测降水量 (mm)')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_residuals(y_true, y_pred, save_path: str = None):
    """绘制残差图"""
    residuals = y_pred - y_true
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_true, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    
    plt.title('残差分布图')
    plt.xlabel('真实降水量 (mm)')
    plt.ylabel('残差 (mm)')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()