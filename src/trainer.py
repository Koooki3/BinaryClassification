import os
import numpy as np
import yaml
import torch
import torch.optim as optim
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import logging
from src.utils import save_metrics, plot_training_curves, evaluate_regression_model
from src.models.classifier import RainfallPredictor
from src.data.dataset import DataProcessor

class Trainer:
    """训练器类"""
    def __init__(self, model, training_config, full_config, device, train_loader, valid_loader, test_loader):
        self.logger = logging.getLogger(__name__)
        self.model = model.to(device)
        self.device = device
        self.train_config = training_config  # 训练相关配置
        self.config = full_config  # 完整配置，包含paths等信息
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        
        # 使用MSE损失函数
        self.criterion = nn.MSELoss()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 训练指标
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_rmse': [],
            'val_rmse': [],
            'train_r2': [],
            'val_r2': []
        }
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def _create_optimizer(self):
        """创建优化器"""
        if self.train_config['optimizer'].lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.train_config['learning_rate']
            )
        raise ValueError(f"不支持的优化器: {self.train_config['optimizer']}")

    def _create_scheduler(self):
        """创建学习率调度器"""
        if self.train_config['lr_scheduler']['type'] == 'ReduceLROnPlateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.train_config['lr_scheduler']['factor'],
                patience=self.train_config['lr_scheduler']['patience'],
                min_lr=self.train_config['lr_scheduler']['min_lr']
            )
        return None

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        predictions = []
        targets = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            predictions.extend(outputs.cpu().detach().numpy())
            targets.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 计算评估指标
        metrics = evaluate_regression_model(
            np.array(targets),
            np.array(predictions)
        )
        
        return total_loss / len(self.train_loader), metrics

    def evaluate(self, loader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                predictions.extend(outputs.cpu().numpy())
                targets.extend(labels.cpu().numpy())
        
        # 计算评估指标
        metrics = evaluate_regression_model(
            np.array(targets),
            np.array(predictions)
        )
        
        return total_loss / len(loader), metrics

    def train(self):
        """训练模型"""
        num_epochs = self.train_config['num_epochs']
        patience = self.train_config['early_stopping_patience']
        
        self.logger.info("开始训练...")
        best_model = None
        
        for epoch in range(num_epochs):
            # 训练epoch
            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics = self.evaluate(self.valid_loader)
            
            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # 记录指标
            self.metrics['train_loss'].append(train_loss)
            self.metrics['val_loss'].append(val_loss)
            self.metrics['train_rmse'].append(train_metrics['rmse'])
            self.metrics['val_rmse'].append(val_metrics['rmse'])
            self.metrics['train_r2'].append(train_metrics['r2'])
            self.metrics['val_r2'].append(val_metrics['r2'])
            
            # 输出进度
            self.logger.info(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f} "
                f"Val Loss: {val_loss:.4f} "
                f"Train RMSE: {train_metrics['rmse']:.4f} "
                f"Val RMSE: {val_metrics['rmse']:.4f} "
                f"Train R²: {train_metrics['r2']:.4f} "
                f"Val R²: {val_metrics['r2']:.4f}"
            )
            
            # 早停检查
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                best_model = self.model
                
                # 保存检查点
                if (epoch + 1) % self.train_config.get('save_model_interval', 10) == 0:
                    checkpoint_path = Path(self.config['paths']['models_dir']) / f'model_epoch_{epoch+1}.pth'
                    self.save_checkpoint(epoch + 1, checkpoint_path)
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # 保存训练指标
        metrics_path = Path(self.config['paths']['results_dir']) / 'metrics.json'
        save_metrics(self.metrics, metrics_path)
        
        # 绘制训练曲线
        plots_path = Path(self.config['paths']['plots_dir']) / 'training_curves.png'
        plot_training_curves(self.metrics, plots_path)
        
        return best_model

    def save_checkpoint(self, epoch, path):
        """保存检查点"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.metrics,
            'best_val_loss': self.best_val_loss
        }, path)
        self.logger.info(f"检查点已保存到: {path}")

    def load_checkpoint(self, path):
        """加载检查点"""
        if not Path(path).exists():
            self.logger.error(f"检查点文件不存在: {path}")
            return False
        
        try:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.metrics = checkpoint['metrics']
            self.best_val_loss = checkpoint['best_val_loss']
            self.logger.info(f"检查点已从 {path} 加载")
            return True
        except Exception as e:
            self.logger.error(f"加载检查点失败: {str(e)}")
            return False

if __name__ == "__main__":
    try:
        # 加载配置
        config_path = 'config/config.yaml'
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 初始化数据处理器
        data_processor = DataProcessor(config)
        
        # 准备数据
        if not data_processor.load_data():
            raise RuntimeError("数据加载失败")
            
        if not data_processor.preprocess_data():
            raise RuntimeError("数据预处理失败")
            
        # 创建数据加载器
        train_loader, valid_loader, test_loader = data_processor.create_data_loaders()
        if train_loader is None:
            raise RuntimeError("创建数据加载器失败")

        # 初始化模型和训练器
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RainfallPredictor(config['model'])
        trainer = Trainer(model, config['training'], config, device, train_loader, valid_loader, test_loader)
        
        # 开始训练
        best_model = trainer.train()
        
        # 保存最终模型
        if best_model is not None:
            trainer.save_checkpoint(
                epoch=config['training']['num_epochs'],
                path='models/final_model.pth'
            )
            
    except Exception as e:
        logging.error(f"训练过程出错: {str(e)}")
        raise