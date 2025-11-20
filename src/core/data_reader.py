import pandas as pd
import numpy as np
from datetime import datetime
import os

class DataReader:
    """
    数据读取器，用于加载和预处理K线数据
    """
    
    def __init__(self, data_dir=None):
        """
        初始化数据读取器
        
        Args:
            data_dir: 数据文件目录，默认为当前目录
        """
        self.data_dir = data_dir or os.getcwd()
    
    def read_csv_file(self, file_name):
        """
        读取CSV格式的K线数据文件
        
        Args:
            file_name: 文件名或文件路径
            
        Returns:
            pandas.DataFrame: 包含K线数据的DataFrame
        """
        # 构建完整文件路径
        file_path = file_name if os.path.isabs(file_name) else os.path.join(self.data_dir, file_name)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 检查必需的列是否存在
        required_columns = ['time', 'open', 'high', 'low', 'close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"文件缺少必需的列: {col}")
        
        # 处理时间戳
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('datetime', inplace=True)
        
        # 重命名列名，使其更标准化
        df.rename(columns={'Volume': 'volume'}, inplace=True)
        
        # 排序确保数据按时间顺序排列
        df.sort_index(inplace=True)
        
        return df
    
    def get_all_available_data(self):
        """
        获取目录中所有可用的K线数据文件
        
        Returns:
            dict: {文件名: 数据描述}
        """
        available_data = {}
        
        # 遍历目录中的文件
        for file in os.listdir(self.data_dir):
            if file.endswith('.csv') and ('1D' in file or '1d' in file):  # 筛选日线数据
                try:
                    # 读取文件获取基本信息
                    df = self.read_csv_file(file)
                    # 提取交易对信息
                    if 'BTC' in file:
                        symbol = 'BTCUSDT'
                        exchange = 'BINANCE'
                    elif 'MSTR' in file:
                        symbol = 'MSTR'
                        exchange = 'BATS'
                    else:
                        symbol = file.split('_')[1] if '_' in file else file.split('.')[0]
                        exchange = file.split('_')[0] if '_' in file else 'UNKNOWN'
                    
                    available_data[file] = {
                        'symbol': symbol,
                        'exchange': exchange,
                        'timeframe': '1D',
                        'start_date': df.index.min().strftime('%Y-%m-%d'),
                        'end_date': df.index.max().strftime('%Y-%m-%d'),
                        'total_rows': len(df)
                    }
                except Exception as e:
                    print(f"处理文件 {file} 时出错: {str(e)}")
                    continue
        
        return available_data
    
    def align_dataframes(self, dataframes, fill_method='ffill', dropna=True):
        """
        对齐多个时间序列DataFrame，解决不同市场时间戳不一致的问题
        
        Args:
            dataframes: 字典，键为数据集名称，值为DataFrame
            fill_method: 缺失值填充方法，'ffill'为前向填充，'bfill'为后向填充，None为不填充
            dropna: 是否删除仍然包含NaN的行
            
        Returns:
            字典: 对齐后的DataFrame
            pandas.DatetimeIndex: 共同的时间索引
        """
        # 确保所有DataFrame都有DatetimeIndex
        for name, df in dataframes.items():
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(f"DataFrame {name} 必须有DatetimeIndex")
        
        # 找出所有DataFrame的时间范围交集
        all_indices = [df.index for df in dataframes.values()]
        common_start = max([idx.min() for idx in all_indices])
        common_end = min([idx.max() for idx in all_indices])
        
        # 创建一个完整的日期范围（考虑到交易日）
        # 先获取所有唯一的日期点，然后筛选出在共同时间范围内的
        all_dates = pd.DatetimeIndex([])
        for idx in all_indices:
            all_dates = all_dates.union(idx)
        
        # 筛选出在共同时间范围内的日期
        common_dates = all_dates[(all_dates >= common_start) & (all_dates <= common_end)]
        
        # 对每个DataFrame进行重索引和填充
        aligned_dfs = {}
        for name, df in dataframes.items():
            # 重索引到共同日期范围
            aligned_df = df.reindex(common_dates)
            
            # 填充缺失值
            if fill_method == 'ffill':
                aligned_df = aligned_df.ffill()
            elif fill_method == 'bfill':
                aligned_df = aligned_df.bfill()
            
            # 删除仍然包含NaN的行
            if dropna:
                aligned_df = aligned_df.dropna()
            
            aligned_dfs[name] = aligned_df
        
        return aligned_dfs, common_dates
    
    def prepare_data_for_analysis(self, df, add_indicators=True):
        """
        准备数据用于分析，添加常用指标
        
        Args:
            df: K线数据DataFrame
            add_indicators: 是否添加技术指标
            
        Returns:
            pandas.DataFrame: 处理后的数据
        """
        # 创建副本以避免修改原始数据
        processed_df = df.copy()
        
        # 计算基本指标
        if add_indicators:
            # 计算收益率
            processed_df['returns'] = processed_df['close'].pct_change()
            
            # 计算波动率（20日）
            processed_df['volatility'] = processed_df['returns'].rolling(window=20).std() * np.sqrt(20)
            
            # 计算简单移动平均线
            processed_df['SMA20'] = processed_df['close'].rolling(window=20).mean()
            processed_df['SMA50'] = processed_df['close'].rolling(window=50).mean()
            
            # 计算相对强弱指标（RSI）
            delta = processed_df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            processed_df['RSI'] = 100 - (100 / (1 + rs))
            
            # 计算MACD
            exp1 = processed_df['close'].ewm(span=12, adjust=False).mean()
            exp2 = processed_df['close'].ewm(span=26, adjust=False).mean()
            processed_df['MACD'] = exp1 - exp2
            processed_df['signal'] = processed_df['MACD'].ewm(span=9, adjust=False).mean()
        
        # 删除NaN值
        processed_df.dropna(inplace=True)
        
        return processed_df
        
    def prepare_data(self, df, add_indicators=True):
        """
        准备数据供回测使用（与prepare_data_for_analysis功能相同）
        
        Args:
            df: K线数据DataFrame
            add_indicators: 是否添加技术指标
            
        Returns:
            pandas.DataFrame: 处理后的数据
        """
        return self.prepare_data_for_analysis(df, add_indicators)
    
    def sample_data_for_llm(self, df, sample_size=100):
        """
        为大语言模型分析提供数据样本
        
        Args:
            df: K线数据DataFrame
            sample_size: 样本大小
            
        Returns:
            str: 格式化的数据样本字符串
        """
        # 如果数据量小于样本大小，使用全部数据
        if len(df) <= sample_size:
            sample_df = df
        else:
            # 均匀采样
            step = len(df) // sample_size
            sample_df = df.iloc[::step].head(sample_size)
        
        # 格式化数据字符串
        data_str = """
# K线数据样本 (最多100条)
# 格式: 日期,开盘价,最高价,最低价,收盘价,成交量
"""
        
        for idx, row in sample_df.iterrows():
            date_str = idx.strftime('%Y-%m-%d')
            data_str += f"{date_str},{row['open']:.2f},{row['high']:.2f},{row['low']:.2f},{row['close']:.2f},{row['volume']:.2f}\n"
        
        return data_str
        
    def get_data_summary(self, df):
        """
        获取数据摘要信息
        
        Args:
            df: K线数据DataFrame
            
        Returns:
            str: 数据摘要信息
        """
        summary = f"""
# 数据摘要
- 数据时间范围: {df.index.min().strftime('%Y-%m-%d')} 到 {df.index.max().strftime('%Y-%m-%d')}
- 总记录数: {len(df)}
- 价格范围: ${df['low'].min():.2f} - ${df['high'].max():.2f}
- 平均成交量: {df['volume'].mean():.0f}
"""
        return summary
        
    def calculate_indicators(self, df):
        """
        计算技术指标
        
        Args:
            df: K线数据DataFrame
            
        Returns:
            pandas.DataFrame: 添加了指标的数据
        """
        # 创建副本以避免修改原始数据
        df_copy = df.copy()
        
        # 计算收益率
        df_copy['returns'] = df_copy['close'].pct_change()
        
        # 计算波动率（20日）
        df_copy['volatility'] = df_copy['returns'].rolling(window=20).std() * np.sqrt(20)
        
        # 计算简单移动平均线
        df_copy['SMA20'] = df_copy['close'].rolling(window=20).mean()
        df_copy['SMA50'] = df_copy['close'].rolling(window=50).mean()
        
        # 计算相对强弱指标（RSI）
        delta = df_copy['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_copy['RSI'] = 100 - (100 / (1 + rs))
        
        # 计算MACD
        exp1 = df_copy['close'].ewm(span=12, adjust=False).mean()
        exp2 = df_copy['close'].ewm(span=26, adjust=False).mean()
        df_copy['MACD'] = exp1 - exp2
        df_copy['signal'] = df_copy['MACD'].ewm(span=9, adjust=False).mean()
        
        return df_copy
        
    def get_data_sample(self, df, sample_size=5):
        """
        获取数据样本
        
        Args:
            df: K线数据DataFrame
            sample_size: 样本大小
            
        Returns:
            pandas.DataFrame: 数据样本
        """
        # 如果数据量小于样本大小，使用全部数据
        if len(df) <= sample_size:
            return df
        else:
            # 均匀采样
            step = len(df) // sample_size
            return df.iloc[::step].head(sample_size)

# 示例用法
if __name__ == "__main__":
    reader = DataReader()
    
    # 显示可用数据
    available_data = reader.get_all_available_data()
    print("可用的K线数据文件:")
    for file_name, info in available_data.items():
        print(f"- {file_name}: {info['symbol']} ({info['exchange']}), 时间范围: {info['start_date']} 到 {info['end_date']}, 共{info['total_rows']}条数据")
    
    # 读取并处理BTC数据作为示例
    try:
        btc_file = "BINANCE_BTCUSDT, 1D_filtered.csv"
        print(f"\n读取 {btc_file} 数据...")
        df = reader.read_csv_file(btc_file)
        print(f"原始数据形状: {df.shape}")
        
        # 添加指标
        processed_df = reader.prepare_data_for_analysis(df)
        print(f"处理后的数据形状: {processed_df.shape}")
        
        # 显示数据样本
        sample_str = reader.sample_data_for_llm(processed_df)
        print("\n数据样本:")
        print(sample_str[:1000] + "..." if len(sample_str) > 1000 else sample_str)
        
    except Exception as e:
        print(f"错误: {str(e)}")