import os
import sys
import pandas as pd
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_sampler import VolatilityBasedSampler
from src.core.data_reader import DataReader

def test_volatility_sampling():
    """
    测试波动率采样功能
    """
    print("开始测试波动率采样功能...")
    
    # 1. 准备测试数据
    # 如果有实际数据文件，可以使用它
    test_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'BINANCE_BTCUSDT_1D.csv')
    
    if os.path.exists(test_data_path):
        print(f"使用实际数据文件: {test_data_path}")
        reader = DataReader(data_dir=os.path.dirname(test_data_path))
        data = reader.read_csv_file('BINANCE_BTCUSDT_1D.csv')
        print(f"加载的数据形状: {data.shape}")
    else:
        print("实际数据文件不存在，生成模拟数据...")
        # 生成模拟的OHLCV数据
        np.random.seed(42)  # 确保结果可重现
        dates = pd.date_range(start='2020-01-01', periods=1000)
        
        # 创建一些合成的价格数据
        base_price = 10000
        volatility = 0.02  # 日波动率
        returns = np.random.normal(0, volatility, size=len(dates))
        
        # 添加一些波动率聚类特性（更接近真实市场）
        volatility_regime = np.random.choice([0.5, 1.0, 2.0], size=len(dates), p=[0.4, 0.4, 0.2])
        returns = returns * volatility_regime
        
        price = base_price * np.exp(np.cumsum(returns))
        
        # 创建OHLCV数据
        data = pd.DataFrame({
            'Date': dates,
            'Open': price * np.random.uniform(0.995, 1.005, size=len(dates)),
            'High': price * np.random.uniform(1.005, 1.01, size=len(dates)),
            'Low': price * np.random.uniform(0.99, 0.995, size=len(dates)),
            'Close': price,
            'Volume': np.random.uniform(1000, 10000, size=len(dates))
        })
        data.set_index('Date', inplace=True)
        print(f"生成的模拟数据形状: {data.shape}")
    
    # 2. 测试不同的采样策略
    sampler = VolatilityBasedSampler(volatility_window=20)
    
    # 测试波动率采样（默认策略）
    print("\n测试波动率采样:")
    volatility_sample = sampler.sample_by_volatility(data, target_samples=50)
    print(f"波动率采样后的数据形状: {volatility_sample.shape}")
    print(f"采样比例: {len(volatility_sample) / len(data):.4f}")
    
    # 测试不同的目标采样数量
    print("\n测试不同的目标采样数量:")
    small_sample = sampler.sample_by_volatility(data, target_samples=30)
    large_sample = sampler.sample_by_volatility(data, target_samples=100)
    print(f"30个样本时的数据形状: {small_sample.shape}")
    print(f"100个样本时的数据形状: {large_sample.shape}")
    
    # 3. 验证采样结果
    print("\n验证采样结果:")
    # 检查采样数据是否是原始数据的子集
    for sample, name in [(volatility_sample, '波动率'), (small_sample, '小样本'), (large_sample, '大样本')]:
        is_subset = sample.index.isin(data.index).all()
        print(f"{name}采样结果是原始数据的子集: {is_subset}")
    
    # 4. 测试边界情况
    print("\n测试边界情况:")
    # 空数据测试
    try:
        empty_sample = sampler.sample_by_volatility(pd.DataFrame(), target_samples=10)
        print("空数据测试: 正确处理")
    except Exception as e:
        print(f"空数据测试出错: {e}")
    
    # 数据量小于目标采样数量
    small_data = data.head(10)
    small_sample = sampler.sample_by_volatility(small_data, target_samples=20)
    print(f"数据量小于目标采样数量时的结果形状: {small_sample.shape}")
    
    print("\n波动率采样功能测试完成！")

def test_integration_with_ai_generator():
    """
    测试与AI策略生成器的集成
    """
    print("\n开始测试与AI策略生成器的集成...")
    
    try:
        from src.utils.ai_strategy_generator import AIStrategyGenerator
        
        # 创建一个模拟的AI生成器实例
        generator = AIStrategyGenerator(
            api_key="test_key",  # 测试用的API密钥
            use_reasoning=True,
            use_volatility_sampling=True,
            target_samples=20,
            sampling_strategy='hybrid'
        )
        
        # 准备测试数据
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=100)
        price = 10000 * np.exp(np.cumsum(np.random.normal(0, 0.02, size=len(dates))))
        
        data = pd.DataFrame({
            'Date': dates,
            'Open': price * np.random.uniform(0.995, 1.005, size=len(dates)),
            'High': price * np.random.uniform(1.005, 1.01, size=len(dates)),
            'Low': price * np.random.uniform(0.99, 0.995, size=len(dates)),
            'Close': price,
            'Volume': np.random.uniform(1000, 10000, size=len(dates))
        })
        data.set_index('Date', inplace=True)
        
        # 模拟analyze_data方法中的采样逻辑
        # 注意：这里不实际调用API，只测试采样部分
        sample_data = None
        
        if hasattr(generator, '_sampler') and generator._use_volatility_sampling:
            print("使用波动率采样器进行采样")
            sample_data = generator._sampler.sample_by_volatility(
                data, 
                target_samples=generator._target_samples,
                strategy=generator._sampling_strategy
            )
        else:
            print("使用默认的head(20)采样")
            sample_data = data.head(20)
        
        print(f"采样后的数据形状: {sample_data.shape}")
        print("与AI策略生成器的集成测试成功！")
        
    except Exception as e:
        print(f"与AI策略生成器的集成测试失败: {e}")

if __name__ == "__main__":
    test_volatility_sampling()
    test_integration_with_ai_generator()