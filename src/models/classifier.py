import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class BinaryClassifier(nn.Module):
    """二分类神经网络模型"""
    def __init__(self, config):
        super(BinaryClassifier, self).__init__()
        self.logger = logging.getLogger(__name__)
        
        # 从配置中获取模型参数
        self.input_size = config['input_size']
        hidden_layers = config['hidden_layers']
        self.dropout_rate = config['dropout_rate']
        
        # 构建网络层
        layers = []
        prev_size = self.input_size
        
        # 添加隐藏层
        for layer in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, layer['size']),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev_size = layer['size']
        
        # 输出层
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
        # 初始化参数
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化模型参数"""
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        return self.model(x)

    def predict(self, x):
        """预测函数"""
        self.eval()  # 设置为评估模式
        with torch.no_grad():
            outputs = self.forward(x)
            predictions = (outputs > 0.5).float()
        self.train()  # 恢复训练模式
        return predictions
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        model_info = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'input_size': self.input_size,
            'model_structure': str(self.model)
        }
        
        return model_info

    def save_model(self, path):
        """保存模型"""
        try:
            torch.save({
                'model_state_dict': self.state_dict(),
                'model_config': {
                    'input_size': self.input_size,
                    'dropout_rate': self.dropout_rate
                }
            }, path)
            self.logger.info(f"模型已保存到: {path}")
            return True
        except Exception as e:
            self.logger.error(f"保存模型失败: {str(e)}")
            return False

    def load_model(self, path):
        """加载模型"""
        try:
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"模型已从 {path} 加载")
            return True
        except Exception as e:
            self.logger.error(f"加载模型失败: {str(e)}")
            return False