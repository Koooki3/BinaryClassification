import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from pathlib import Path

class RainfallPredictor(nn.Module):
    """降水量预测模型"""
    def __init__(self, config):
        super(RainfallPredictor, self).__init__()
        self.logger = logging.getLogger(__name__)
        
        # 从配置中获取模型参数
        self.input_size = config['input_size']
        hidden_layers = config['hidden_layers']
        self.dropout_rate = config['dropout_rate']
        self.use_batch_norm = config.get('batch_norm', True)
        
        # 构建网络层
        layers = []
        prev_size = self.input_size
        
        # 添加隐藏层
        for layer in hidden_layers:
            # 添加线性层
            layers.append(nn.Linear(prev_size, layer['size']))
            
            # 添加批归一化层
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(layer['size']))
            
            # 添加激活函数
            if layer['activation'].lower() == 'relu':
                layers.append(nn.ReLU())
            elif layer['activation'].lower() == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            
            # 添加Dropout
            layers.append(nn.Dropout(self.dropout_rate))
            
            prev_size = layer['size']
        
        # 输出层（线性输出，用于回归）
        layers.append(nn.Linear(prev_size, 1))
        
        self.model = nn.Sequential(*layers)
        
        # 初始化参数
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化模型参数"""
        for m in self.model:
            if isinstance(m, nn.Linear):
                # 使用He初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        return self.model(x)

    def predict(self, x):
        """预测函数"""
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)
        self.train()
        return predictions
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        model_info = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'input_size': self.input_size,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.use_batch_norm,
            'model_structure': str(self.model)
        }
        
        return model_info

    def save_model(self, path: Path):
        """保存模型"""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                'model_state_dict': self.state_dict(),
                'model_config': {
                    'input_size': self.input_size,
                    'dropout_rate': self.dropout_rate,
                    'batch_norm': self.use_batch_norm
                }
            }
            torch.save(checkpoint, path)
            self.logger.info(f"模型已保存到: {path}")
            return True
        except Exception as e:
            self.logger.error(f"保存模型失败: {str(e)}")
            return False

    def load_model(self, path: Path):
        """加载模型"""
        try:
            if not path.exists():
                raise FileNotFoundError(f"模型文件不存在: {path}")
            
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint['model_state_dict'])
            
            # 验证配置
            config = checkpoint['model_config']
            assert self.input_size == config['input_size'], "输入维度不匹配"
            assert self.dropout_rate == config['dropout_rate'], "Dropout率不匹配"
            assert self.use_batch_norm == config['batch_norm'], "批归一化配置不匹配"
            
            self.logger.info(f"模型已从 {path} 加载")
            return True
        except Exception as e:
            self.logger.error(f"加载模型失败: {str(e)}")
            return False