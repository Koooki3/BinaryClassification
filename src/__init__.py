"""
PyTorch 二分类项目
~~~~~~~~~~~~~~~~

这个包提供了一个完整的二分类深度学习项目框架，包含以下主要组件：

- data: 数据处理和加载
- models: 模型定义和管理
- utils: 工具函数和辅助功能
- trainer: 训练和评估功能
"""

from .data import DataProcessor
from .models import BinaryClassifier
from .utils import setup_logger, save_metrics, load_metrics, plot_training_curves
from .trainer import Trainer

__version__ = '1.0.0'
__author__ = 'kooki3'

__all__ = [
    'DataProcessor',
    'BinaryClassifier',
    'Trainer',
    'setup_logger',
    'save_metrics',
    'load_metrics',
    'plot_training_curves'
]