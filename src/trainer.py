import os
import yaml
import torch
import torch.optim as optim
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import logging
from src.utils import save_metrics, plot_training_curves
from src.models.classifier import BinaryClassifier
from src.data.dataset import DataProcessor

class Trainer:
    """训练器类"""
    def __init__(self, model, config, device, train_loader, test_loader):
        self.logger = logging.getLogger(__name__)
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        # 初始化损失函数和优化器
        self.criterion = nn.BCELoss()
        self.optimizer = self._create_optimizer()
        
        # 训练指标
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def _create_optimizer(self):
        """创建优化器"""
        if self.config['optimizer'].lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate']
            )
        # 可以添加其他优化器的支持
        raise ValueError(f"不支持的优化器: {self.config['optimizer']}")

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets.unsqueeze(1))
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = (outputs > 0.5).float()
            correct += (pred.view(-1) == targets).sum().item()
            total += targets.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader), correct / total

    def evaluate(self):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets.unsqueeze(1))
                
                total_loss += loss.item()
                pred = (outputs > 0.5).float()
                correct += (pred.view(-1) == targets).sum().item()
                total += targets.size(0)
        
        return total_loss / len(self.test_loader), correct / total

    def train(self):
        """训练模型"""
        num_epochs = self.config['num_epochs']
        patience = self.config['early_stopping_patience']
        
        self.logger.info("开始训练...")
        for epoch in range(num_epochs):
            # 训练epoch
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.evaluate()
            
            # 记录指标
            self.metrics['train_loss'].append(train_loss)
            self.metrics['val_loss'].append(val_loss)
            self.metrics['train_acc'].append(train_acc)
            self.metrics['val_acc'].append(val_acc)
            
            # 输出进度
            self.logger.info(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f} "
                f"Val Loss: {val_loss:.4f} "
                f"Train Acc: {train_acc:.4f} "
                f"Val Acc: {val_acc:.4f}"
            )
            
            # 早停检查
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                best_model = self.model
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # 保存训练指标
        save_metrics(self.metrics, 'logs/metrics.json')
        
        # 绘制训练曲线
        plot_training_curves(self.metrics, 'logs/training_curves.png')
        
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
        train_loader, test_loader = data_processor.create_data_loaders()
        if train_loader is None:
            raise RuntimeError("创建数据加载器失败")

        # 初始化模型和训练器
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = BinaryClassifier(config['model'])
        trainer = Trainer(model, config['training'], device, train_loader, test_loader)
        
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