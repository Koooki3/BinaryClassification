import os
import yaml
import torch
import numpy as np
import logging
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data.dataset import DataProcessor
from src.models.classifier import RainfallPredictor
from src.utils import setup_logger

def load_config():
    """加载配置文件"""
    try:
        config_path = Path("config/config.yaml")
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"无法加载配置文件: {str(e)}")

def plot_prediction_scatter(y_true, y_pred, save_path):
    """绘制预测值与真实值的散点图"""
    plt.figure(figsize=(10, 6))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号
    
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('真实降水量 (mm)')
    plt.ylabel('预测降水量 (mm)')
    plt.title('降水量预测值与真实值对比')
    plt.grid(True)
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(model, test_loader, device, scaler_y=None):
    """评估模型性能"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    predictions = np.array(all_preds)
    labels = np.array(all_labels)
    
    # 如果使用了标准化，需要转换回原始刻度
    if scaler_y is not None:
        predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
        labels = scaler_y.inverse_transform(labels.reshape(-1, 1)).flatten()
    
    return predictions, labels

def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def setup_logging(config):
    """配置日志系统"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'evaluate_{timestamp}.log'
    
    # 修正日志格式字符串
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 修改 levellevel 为 levelname
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 配置根日志记录器
    logger = logging.getLogger()
    logger.setLevel(config['logging']['log_level'])
    
    # 清除现有的处理器
    logger.handlers.clear()
    
    # 添加新的处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def main():
    try:
        # 加载配置
        config = load_config()
        
        # 设置日志
        logger = setup_logging(config)
        
        logger.info("开始降水量预测模型评估")
        
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {device}")
        
        # 加载数据
        data_processor = DataProcessor(config)
        if not data_processor.load_data():
            logger.error("数据加载失败")
            return
        
        if not data_processor.preprocess_data():
            logger.error("数据预处理失败")
            return
        
        train_loader, valid_loader, test_loader = data_processor.create_data_loaders()
        if test_loader is None:
            logger.error("创建数据加载器失败")
            return
        
        # 加载已训练的模型
        model_path = Path("models/rainfall_model_latest.pth")
        
        if not model_path.exists():
            logger.error(f"模型文件不存在: {model_path}")
            return
        
        # 加载模型数据
        checkpoint = torch.load(model_path)
        
        # 更新模型配置
        config.update(checkpoint['config'])
        
        # 重新初始化模型（使用更新后的配置）
        model = RainfallPredictor(config['model']).to(device)
        
        # 加载模型状态
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 更新数据处理器的缩放器
        data_processor.scaler_X = checkpoint['scaler_X']
        data_processor.scaler_y = checkpoint['scaler_y']
        
        logger.info("模型和数据处理器加载成功")
        
        # 评估模型
        predictions, labels = evaluate_model(
            model, 
            test_loader, 
            device, 
            data_processor.scaler_y
        )
        
        # 计算评估指标
        metrics = calculate_metrics(labels, predictions)
        
        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存评估结果
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        # 保存预测散点图
        plot_prediction_scatter(
            labels,
            predictions,
            results_dir / f'prediction_scatter_{timestamp}.png'
        )
        
        # 保存评估报告
        report = (
            f"降水量预测模型评估报告\n"
            f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"评估指标:\n"
            f"MSE (均方误差): {metrics['MSE']:.4f}\n"
            f"RMSE (均方根误差): {metrics['RMSE']:.4f} mm\n"
            f"MAE (平均绝对误差): {metrics['MAE']:.4f} mm\n"
            f"R² (决定系数): {metrics['R2']:.4f}\n"
        )
        
        # 使用 utf-8 编码写入文件
        with open(results_dir / f'rainfall_evaluation_{timestamp}.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 输出评估结果
        logger.info("\n" + report)
        logger.info("评估完成")
        
    except Exception as e:
        logger.exception(f"评估过程出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()