import os
import yaml
import logging
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from src.data.dataset import DataProcessor
from src.models.classifier import RainfallPredictor
from src.trainer import Trainer
from src.utils import (
    setup_logger,
    save_metrics,
    plot_training_curves,
    plot_prediction_scatter,
    plot_residuals,
    evaluate_regression_model
)

def setup_logging(config):
    """配置日志系统"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{timestamp}.log'
    
    logging.basicConfig(
        level=config['logging']['log_level'],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config():
    """加载配置文件"""
    try:
        config_path = Path("config/config.yaml")
        with open(config_path, 'r', encoding='utf-8') as config_file:
            config = yaml.safe_load(config_file)
            config['data']['csv_file'] = 'data/datasetMeteostat_广州.csv'
            return config
    except Exception as e:
        raise RuntimeError(f"无法加载配置文件: {str(e)}")

def setup_seed(seed):
    """设置随机种子以确保实验可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_user_input():
    """获取用户输入的预测参数"""
    print("\n=== 降水量预测指标输入 ===")
    print("\n请依次输入以下6个指标值（用空格分隔）：")
    print("\n1. 平均温度")
    print("   - 范围: 5-35°C")
    print("   - 说明: 当日平均气温")
    
    print("\n2. 最低温度")
    print("   - 范围: 0-30°C")
    print("   - 说明: 当日最低气温")
    
    print("\n3. 最高温度")
    print("   - 范围: 10-40°C")
    print("   - 说明: 当日最高气温")
    
    print("\n4. 风向")
    print("   - 范围: 0-360度")
    print("   - 说明: 风向角度（0表示北风）")
    
    print("\n5. 风速")
    print("   - 范围: 0-50 km/h")
    print("   - 说明: 平均风速")
    
    print("\n6. 气压")
    print("   - 范围: 990-1020 hPa")
    print("   - 说明: 海平面气压")
    
    print("\n示例输入：20 15 25 180 10 1013")
    
    ranges = [(5, 35), (0, 30), (10, 40), (0, 360),
              (0, 50), (990, 1020)]
    
    while True:
        try:
            values = input("\n请输入这6个指标值 > ").strip().split()
            
            if len(values) != 6:
                print("错误：必须输入6个数值！请重试。")
                continue
            
            features = []
            for i, (value, (min_val, max_val)) in enumerate(zip(values, ranges)):
                val = float(value)
                if not min_val <= val <= max_val:
                    print(f"错误：第{i+1}个值 {val} 超出范围 [{min_val}, {max_val}]")
                    break
                features.append(val)
            else:
                return features
            
        except ValueError:
            print("错误：请输入有效的数值！格式参考示例。")

def predict_sample(model, features, device, scaler=None):
    """预测单个样本"""
    if scaler is not None:
        features = scaler.transform([features])
    else:
        features = np.array([features])
    
    features_tensor = torch.FloatTensor(features).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(features_tensor)
        prediction = output.item()
        
    return prediction

def main():
    try:
        # 加载配置
        config = load_config()
        
        # 设置随机种子
        setup_seed(config['data']['random_seed'])
        
        # 设置日志
        logger = setup_logging(config)
        logger.info("开始训练广州降水量预测模型")
        
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {device}")
        
        # 初始化数据处理器
        data_processor = DataProcessor(config)
        
        # 加载并预处理数据
        if not data_processor.load_data():
            logger.error("气象数据集加载失败")
            return
        
        if not data_processor.preprocess_data():
            logger.error("数据预处理失败")
            return
        
        # 创建数据加载器
        train_loader, valid_loader, test_loader = data_processor.create_data_loaders()
        if train_loader is None:
            logger.error("创建数据加载器失败")
            return

        # 创建必要的目录
        dirs_to_create = {
            'models_dir': Path('models'),
            'results_dir': Path('results'),
            'plots_dir': Path('plots'),
            'logs_dir': Path('logs')
        }

        for dir_path in dirs_to_create.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        # 更新配置中的路径
        if 'paths' not in config:
            config['paths'] = {}
        config['paths'].update({
            'models_dir': str(dirs_to_create['models_dir']),
            'results_dir': str(dirs_to_create['results_dir']),
            'plots_dir': str(dirs_to_create['plots_dir']),
            'logs_dir': str(dirs_to_create['logs_dir'])
        })
        
        # 初始化模型
        model = RainfallPredictor(config['model']).to(device)
        logger.info(f"模型结构:\n{model}")
        
        # 初始化训练器
        trainer = Trainer(
            model=model,
            training_config=config['training'],
            full_config=config,  # 传递完整配置
            device=device,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader
        )
        
        # 开始训练
        best_model = trainer.train()
        
        # 保存最佳模型
        if best_model is not None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            models_dir = Path(config['paths']['models_dir'])
            
            # 准备保存的模型数据
            model_data = {
                'model_state_dict': best_model.state_dict(),
                'config': config,
                'scaler_X': data_processor.scaler_X,
                'scaler_y': data_processor.scaler_y
            }
            
            # 保存带时间戳的版本
            timestamp_path = models_dir / f'rainfall_model_{timestamp}.pth'
            torch.save(model_data, timestamp_path)
            logger.info(f"带时间戳的模型已保存至: {timestamp_path}")
            
            # 保存最新版本
            latest_path = models_dir / 'rainfall_model_latest.pth'
            torch.save(model_data, latest_path)
            logger.info(f"最新模型已保存至: {latest_path}")
        
        # 绘制训练曲线
        plots_path = Path(config['paths']['plots_dir']) / f'training_curves_{timestamp}.png'
        plot_training_curves(trainer.metrics, save_path=plots_path)
        
        # 保存训练指标
        metrics_path = Path(config['paths']['results_dir']) / f'training_metrics_{timestamp}.json'
        save_metrics(trainer.metrics, metrics_path)
        
        logger.info("训练程序完成")
        
        # 添加交互式预测功能
        print("\n=== 降水量预测功能 ===")
        while input("\n是否进行预测？(y/n) > ").strip().lower() == 'y':
            features = get_user_input()
            predicted_rainfall = predict_sample(
                best_model, 
                features, 
                device, 
                data_processor.scaler_X
            )
            
            # 将预测值转换回原始刻度
            predicted_rainfall = data_processor.scaler_y.inverse_transform(
                [[predicted_rainfall]]
            )[0][0]
            
            if config['data']['scaler']['target_transform'] == 'log1p':
                predicted_rainfall = np.expm1(predicted_rainfall)
            
            print("\n===== 预测结果 =====")
            print(f"预测降水量：{predicted_rainfall:.2f} mm")
            
            # 根据降水量判断天气状况
            if predicted_rainfall < 0.1:
                print("天气状况：晴朗")
                print("建议：适合户外活动")
            elif predicted_rainfall < 10:
                print("天气状况：小雨")
                print("建议：外出请携带雨伞")
            elif predicted_rainfall < 25:
                print("天气状况：中雨")
                print("建议：尽量避免户外活动")
            elif predicted_rainfall < 50:
                print("天气状况：大雨")
                print("建议：注意防范积水和交通安全")
            else:
                print("天气状况：暴雨")
                print("建议：密切关注天气预警，做好防汛准备")
        
        print("\n感谢使用广州降水量预测系统！")
        
    except Exception as e:
        logger.exception(f"程序运行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()