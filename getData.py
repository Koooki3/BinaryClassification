import pandas as pd
import numpy as np
from datetime import datetime
from meteostat import Point, Daily
from pathlib import Path
import logging
from typing import Tuple, Dict, List

# 支持的中国主要城市配置
CHINA_CITIES: Dict[str, Dict[str, any]] = {
    '广州': {
        'latitude': 23.1291,
        'longitude': 113.2644,
        'timezone': 'Asia/Shanghai'
    },
    '北京': {
        'latitude': 39.9042,
        'longitude': 116.4074,
        'timezone': 'Asia/Shanghai'
    },
    '上海': {
        'latitude': 31.2304,
        'longitude': 121.4737,
        'timezone': 'Asia/Shanghai'
    },
    '深圳': {
        'latitude': 22.5431,
        'longitude': 114.0579,
        'timezone': 'Asia/Shanghai'
    }
}

# 数据范围配置（基于中国气候特征）
DATA_RANGES: Dict[str, Tuple[float, float]] = {
    'tavg': (-10, 40),    # 平均温度范围（°C）
    'tmin': (-20, 35),    # 最低温度范围
    'tmax': (-5, 45),     # 最高温度范围
    'wdir': (0, 360),     # 风向范围（度）
    'wspd': (0, 50),      # 风速范围（km/h）
    'pres': (980, 1030),  # 气压范围（hPa）
    'prcp': (0, 300)      # 降水量范围（mm）
}

def setup_logging() -> logging.Logger:
    """配置日志系统"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'weatherData_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_user_input() -> Tuple[str, datetime, datetime]:
    """获取用户输入的城市和时间范围"""
    print("\n=== 气象数据获取系统 ===")
    print("\n支持的城市：")
    for city in CHINA_CITIES.keys():
        print(f"- {city}")
    
    while True:
        city = input("\n请输入城市名称（如：广州）：").strip()
        if city in CHINA_CITIES:
            break
        print("不支持的城市，请重新输入！")
    
    while True:
        try:
            start_str = input("\n请输入起始日期（格式：YYYY-MM-DD）：").strip()
            end_str = input("请输入结束日期（格式：YYYY-MM-DD）：").strip()
            
            start_date = datetime.strptime(start_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_str, '%Y-%m-%d')
            
            if start_date > end_date:
                print("起始日期不能晚于结束日期！")
                continue
                
            if end_date > datetime.now():
                print("结束日期不能晚于今天！")
                continue
                
            break
        except ValueError:
            print("日期格式错误，请使用YYYY-MM-DD格式！")
    
    return city, start_date, end_date

def fetch_weather_data(city: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """获取指定城市的天气数据"""
    try:
        # 创建观测点
        city_config = CHINA_CITIES[city]
        location = Point(
            city_config['latitude'],
            city_config['longitude']
        )
        
        # 获取日常天气数据
        data = Daily(location, start_date, end_date)
        df = data.fetch()
        
        if df.empty:
            raise ValueError(f"未获取到{city}的数据")
            
        return df
        
    except Exception as e:
        raise RuntimeError(f"获取天气数据失败: {str(e)}")

def validate_weather_data(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """验证并清理天气数据"""
    for column, (min_val, max_val) in DATA_RANGES.items():
        if column not in df.columns:
            continue
            
        # 检查异常值
        mask = (df[column] < min_val) | (df[column] > max_val)
        invalid_count = mask.sum()
        
        if invalid_count > 0:
            logger.warning(f"{column} 列发现 {invalid_count} 个异常值")
            
            # 使用3天移动平均替换异常值
            df.loc[mask, column] = df[column].rolling(
                window=3,
                center=True,
                min_periods=1
            ).mean()
            
    return df

def process_weather_data(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """处理天气数据"""
    try:
        # 选择需要的特征
        features = ['tavg', 'tmin', 'tmax', 'wdir', 'wspd', 'pres', 'prcp']
        df = df[features].copy()
        
        # 统计缺失值
        missing_stats = df.isnull().sum()
        logger.info("缺失值统计：\n%s", missing_stats)
        
        # 处理缺失值
        df = df.interpolate(method='linear', limit_direction='both')
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        # 验证数据
        df = validate_weather_data(df, logger)
        
        # 添加时间特征
        df['month'] = df.index.month
        df['season'] = pd.cut(
            df['month'],
            bins=[0, 3, 6, 9, 12],
            labels=['冬', '春', '夏', '秋'],
            right=False
        )
        
        # 输出统计信息
        logger.info("\n数据统计摘要：\n%s", df[features].describe())
        
        return df[features]  # 只返回预测所需的特征

    except Exception as e:
        logger.error(f"数据处理失败: {str(e)}")
        raise

def save_weather_data(df: pd.DataFrame, output_file: Path, logger: logging.Logger) -> None:
    """保存天气数据"""
    try:
        output_file.parent.mkdir(exist_ok=True)
        
        # 保存为CSV文件
        df.to_csv(output_file, float_format='%.2f')
        logger.info(f"数据已保存到: {output_file}")
        logger.info(f"数据集大小: {df.shape}")
        
    except Exception as e:
        logger.error(f"保存数据失败: {str(e)}")
        raise

def main() -> None:
    """主函数"""
    logger = setup_logging()
    
    try:
        # 获取用户输入
        city, start_date, end_date = get_user_input()
        
        logger.info(f"开始获取{city}天气数据...")
        logger.info(f"时间范围: {start_date.date()} 到 {end_date.date()}")
        
        # 获取数据
        weather_data = fetch_weather_data(city, start_date, end_date)
        logger.info(f"成功获取 {len(weather_data)} 条天气记录")
        
        # 处理数据
        processed_data = process_weather_data(weather_data, logger)
        
        # 保存数据
        output_file = Path(f'data/datasetMeteostat_{city}.csv')
        save_weather_data(processed_data, output_file, logger)
        
        logger.info("数据处理完成!")
        
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()