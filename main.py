import os
import yaml
import logging
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from src.data.dataset import DataProcessor
from src.models.classifier import BinaryClassifier
from src.trainer import Trainer
from src.utils import setup_logger, plot_training_curves

def setup_logging(config):
    """配置日志系统"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'diabetes_training_{timestamp}.log'
    
    logging.basicConfig(
        level=config['logging']['log_level'],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
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
            
            # 确保配置文件中包含数据集的路径
            config['data']['csv_file'] = 'data/dataset.csv'
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
    print("\n=== 糖尿病预测指标输入 ===")
    print("\n请依次输入以下8个指标值（用空格分隔）：")
    print("\n1. 怀孕次数")
    print("   - 范围: 0-17次")
    print("   - 说明: 患者怀孕的次数")
    
    print("\n2. 口服葡萄糖耐量试验中血浆葡萄糖浓度")
    print("   - 范围: 0-199 mg/dL")
    print("   - 说明: 葡萄糖耐量试验2小时后的血糖水平")
    
    print("\n3. 血压")
    print("   - 范围: 0-122 mmHg")
    print("   - 说明: 舒张压（确保数值准确）")
    
    print("\n4. 肱三头肌皮褶厚度")
    print("   - 范围: 0-99 mm")
    print("   - 说明: 反映体内脂肪含量的指标")
    
    print("\n5. 血清胰岛素")
    print("   - 范围: 0-846 mu U/ml")
    print("   - 说明: 2小时血清胰岛素水平")
    
    print("\n6. 体重指数(BMI)")
    print("   - 范围: 0-67.1 kg/m²")
    print("   - 说明: 体重(kg)/身高(m)的平方")
    
    print("\n7. 糖尿病家族遗传系数")
    print("   - 范围: 0.078-2.42")
    print("   - 说明: 基于家族史计算的遗传影响系数")
    
    print("\n8. 年龄")
    print("   - 范围: 21-81岁")
    print("   - 说明: 患者的实际年龄")
    
    print("\n示例输入：6 148 72 35 0 33.6 0.627 50")
    
    ranges = [(0, 17), (0, 199), (0, 122), (0, 99),
              (0, 846), (0, 67.1), (0.078, 2.42), (21, 81)]
    
    while True:
        try:
            values = input("\n请输入这8个指标值 > ").strip().split()
            
            if len(values) != 8:
                print("错误：必须输入8个数值！请重试。")
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
        probability = output.item()
        prediction = 1 if probability > 0.5 else 0
        
    return prediction, probability

def main():
    try:
        # 加载配置
        config = load_config()
        
        # 设置随机种子
        setup_seed(config['data']['random_seed'])
        
        # 设置日志
        logger = setup_logging(config)
        logger.info("开始训练diabetes数据集分类模型")
        
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {device}")
        
        # 初始化数据处理器
        data_processor = DataProcessor(config)
        
        # 加载并预处理数据
        if not data_processor.load_data():
            logger.error("diabetes数据集加载失败")
            return
        
        if not data_processor.preprocess_data():
            logger.error("数据预处理失败")
            return
        
        # 创建数据加载器
        train_loader, test_loader = data_processor.create_data_loaders()
        if train_loader is None:
            logger.error("创建数据加载器失败")
            return
        
        # 初始化模型
        model = BinaryClassifier(config['model']).to(device)
        logger.info(f"模型结构:\n{model}")
        
        # 获取并记录模型信息
        model_info = model.get_model_info()
        logger.info(f"模型总参数量: {model_info['total_params']}")
        logger.info(f"可训练参数量: {model_info['trainable_params']}")
        
        # 初始化训练器
        trainer = Trainer(
            model=model,
            config=config['training'],
            device=device,
            train_loader=train_loader,
            test_loader=test_loader
        )
        
        # 开始训练
        best_model = trainer.train()
        
        # 保存最佳模型
        if best_model is not None:
            save_dir = Path('models')
            save_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = save_dir / f'diabetes_model_{timestamp}.pth'
            
            # 保存模型
            if best_model.save_model(model_path):
                logger.info(f"最佳模型已保存至: {model_path}")
            else:
                logger.error("模型保存失败")
        
        # 绘制训练曲线
        plot_training_curves(
            trainer.metrics,
            save_path=f'logs/diabetes_training_curves_{timestamp}.png'
        )
        
        logger.info("训练程序完成")
        
        # 添加交互式预测功能
        print("\n=== 模型预测功能 ===")
        print("训练完成！是否要进行预测？")
        
        while input("\n是否进行预测？(y/n) > ").strip().lower() == 'y':
            features = get_user_input()
            prediction, probability = predict_sample(
                best_model, 
                features, 
                device, 
                data_processor.scaler
            )
            
            print("\n===== 预测结果 =====")
            if prediction == 1:
                print("预测结果：可能患有糖尿病")
                print(f"患病概率：{probability:.2%}")
            else:
                print("预测结果：健康")
                print(f"健康概率：{(1-probability):.2%}")
        
        print("\n感谢使用糖尿病预测系统！")
        
    except Exception as e:
        logger.exception(f"程序运行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()