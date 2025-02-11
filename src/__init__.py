"""
降水量预测项目
~~~~~~~~~~~~

这个包提供了一个完整的降水量预测深度学习项目框架，包含以下主要组件：

- data: 数据处理和加载
- models: 模型定义和管理
- utils: 工具函数和辅助功能
- trainer: 训练和评估功能
"""

from .data import DataProcessor, RainfallDataset
from .models import RainfallPredictor
from .utils import (
    setup_logger,
    save_metrics,
    load_metrics,
    plot_training_curves,
    plot_prediction_scatter,
    plot_residuals,
    evaluate_regression_model
)
from .trainer import Trainer

__version__ = '2.0.0'
__author__ = 'kooki3'

__all__ = [
    'DataProcessor',
    'RainfallDataset',
    'RainfallPredictor',
    'Trainer',
    'setup_logger',
    'save_metrics',
    'load_metrics',
    'plot_training_curves',
    'plot_prediction_scatter',
    'plot_residuals',
    'evaluate_regression_model'
]