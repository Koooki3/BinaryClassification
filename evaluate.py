import os
import yaml
import torch
import numpy as np
import logging
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

from src.data.dataset import DataProcessor
from src.models.classifier import BinaryClassifier
from src.utils import setup_logger

def load_config():
    """加载配置文件"""
    try:
        config_path = Path("config/config.yaml")
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"无法加载配置文件: {str(e)}")

def plot_confusion_matrix(cm, save_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.savefig(save_path)
    plt.close()

def evaluate_model(model, test_loader, device):
    """评估模型性能"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = (outputs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)

def main():
    try:
        # 加载配置
        config = load_config()
        
        # 设置日志
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        logger = setup_logger(
            name='evaluate',
            log_file=log_dir / f'evaluation_{timestamp}.log',
            level=logging.INFO
        )
        
        logger.info("开始模型评估")
        
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
        
        _, test_loader = data_processor.create_data_loaders()
        
        # 加载模型
        model = BinaryClassifier(config['model']).to(device)
        
        # 修改文件名以评估不同的模型
        model_path = Path("models/diabetes_model_20250210_174550.pth")
        
        if not model_path.exists():
            logger.error(f"模型文件不存在: {model_path}")
            return
        
        model.load_model(model_path)
        logger.info("模型加载成功")
        
        # 评估模型
        predictions, labels = evaluate_model(model, test_loader, device)
        
        # 计算评估指标
        report = classification_report(labels, predictions, target_names=['Class 0', 'Class 1'])
        cm = confusion_matrix(labels, predictions)
        
        # 保存评估结果
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        # 保存混淆矩阵图
        plot_confusion_matrix(
            cm, 
            results_dir / f'confusion_matrix_{timestamp}.png'
        )
        
        # 保存评估报告
        with open(results_dir / f'evaluation_report_{timestamp}.txt', 'w') as f:
            f.write(report)
        
        # 输出评估结果
        logger.info("\n分类报告：\n" + report)
        logger.info("\n混淆矩阵：\n" + str(cm))
        
        logger.info("评估完成")
        
    except Exception as e:
        logger.exception(f"评估过程出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()