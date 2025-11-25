import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime
from src.utils.logger import Logger, log, info, warning, error, debug, critical

class DataSampler:
    """
    数据采样器类，用于从大型数据集中采样数据
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化数据采样器
        
        Args:
            config: 配置参数
        """
        # 设置默认配置
        self.default_config = {
            'target_samples': 500,  # 默认目标采样数量
            'min_samples': 300,     # 最小采样数量
            'max_samples': 800,     # 最大采样数量
            'volatility_window': 12, # 计算波动率的窗口大小（天）
            'volatility_field': 'close', # 用于计算波动率的价格字段
            'segment_sampling_ratio': 0.8 # 分段采样比例
        }
        
        # 更新配置
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
        
        # 初始化参数
        self.target_samples = self.config['target_samples']
        self.min_samples = self.config['min_samples']
        self.max_samples = self.config['max_samples']
        self.volatility_window = self.config['volatility_window']
        self.volatility_field = self.config['volatility_field']
        self.segment_sampling_ratio = self.config['segment_sampling_ratio']
        
    def sample_by_volatility(self, data: pd.DataFrame, target_samples: int = None) -> pd.DataFrame:
        """
        基于波动率进行数据采样，优先选择高波动率时段
        
        Args:
            data: 输入的数据框
            target_samples: 目标采样数量，如果为None则使用配置中的值
            
        Returns:
            采样后的数据框
        """
        try:
            info(f"[波动率采样] 开始处理数据: {len(data)}行")
            
            # 检查数据
            if len(data) <= self.min_samples:
                info(f"[波动率采样] 数据量({len(data)})小于最小采样数({self.min_samples})，直接返回全部数据")
                return data.copy()
            
            # 确定目标采样数
            if target_samples is None:
                target_samples = self.target_samples
            target_samples = min(max(target_samples, self.min_samples), self.max_samples)
            target_samples = min(target_samples, len(data))
            info(f"[波动率采样] 目标采样数: {target_samples}")
            
            # 计算波动率
            info(f"[波动率采样] 使用窗口大小{self.volatility_window}计算波动率...")
            
            # 检查字段是否存在
            if self.volatility_field not in data.columns:
                error(f"[波动率采样] 数据中不存在字段: {self.volatility_field}")
                # 使用第一列作为备选
                first_price_field = next((col for col in ['close', 'Close', 'price', 'Price'] if col in data.columns), None)
                if first_price_field:
                    self.volatility_field = first_price_field
                    warning(f"[波动率采样] 切换到可用字段: {self.volatility_field}")
                else:
                    self.volatility_field = data.columns[0]
                    warning(f"[波动率采样] 字段不可用，使用第一列: {self.volatility_field}")
            
            # 计算对数收益率
            try:
                # 尝试直接计算对数收益率
                log_returns = np.log(data[self.volatility_field] / data[self.volatility_field].shift(1))
                # 计算滚动波动率（标准差的年化值）
                volatility = log_returns.rolling(window=self.volatility_window).std() * np.sqrt(252)
            except Exception as e:
                # 如果失败，尝试使用简单的百分比变化
                warning(f"[波动率采样] 对数收益率计算失败，使用简单收益率: {str(e)}")
                returns = data[self.volatility_field].pct_change()
                volatility = returns.rolling(window=self.volatility_window).std() * np.sqrt(252)
            
            # 移除NaN值
            volatility = volatility.dropna()
            data_subset = data.iloc[len(data) - len(volatility):].copy()
            data_subset['volatility'] = volatility.values
            
            # 统计波动率
            vol_min = data_subset['volatility'].min()
            vol_max = data_subset['volatility'].max()
            info(f"[波动率采样] 波动率统计: 最小={vol_min:.6f}, 最大={vol_max:.6f}")
            
            # 如果所有波动率都相同，使用均匀采样
            if vol_min == vol_max:
                info(f"[波动率采样] 所有数据波动率相同，使用均匀采样")
                sample_indices = np.random.choice(data.index, size=target_samples, replace=False)
                sampled_data = data.loc[sample_indices].sort_index()
                info(f"[波动率采样] 均匀采样完成: {len(sampled_data)}行数据")
                return sampled_data
            
            # 将数据分成多个段
            num_segments = max(10, target_samples // 50)  # 每段约50个样本
            segment_size = len(data_subset) // num_segments
            info(f"[波动率采样] 将数据分为{num_segments}段，每段约{segment_size}行")
            
            # 在每段内根据波动率进行采样
            unique_indices = set()
            
            for i in range(num_segments):
                # 计算段的边界
                start_idx = i * segment_size
                if i == num_segments - 1:
                    end_idx = len(data_subset)  # 最后一段包含剩余所有数据
                else:
                    end_idx = (i + 1) * segment_size
                
                # 获取段数据
                segment_data = data_subset.iloc[start_idx:end_idx]
                
                # 计算该段的采样数量（与段长度成比例）
                segment_ratio = len(segment_data) / len(data_subset)
                segment_samples = max(1, int(target_samples * segment_ratio * self.segment_sampling_ratio))
                
                info(f"[波动率采样] 段{i+1}: {len(segment_data)}行数据，计划采样{segment_samples}行")
                
                # 根据波动率采样（高波动率优先）
                if len(segment_data) > 0:
                    # 计算权重（波动率归一化）
                    seg_vol_min = segment_data['volatility'].min()
                    seg_vol_max = segment_data['volatility'].max()
                    
                    if seg_vol_min != seg_vol_max:
                        # 波动率归一化到0-1区间
                        weights = (segment_data['volatility'] - seg_vol_min) / (seg_vol_max - seg_vol_min)
                    else:
                        # 如果段内波动率都相同，使用均匀权重
                        weights = np.ones(len(segment_data)) / len(segment_data)
                    
                    # 确保权重和为1
                    weights = weights / weights.sum()
                    
                    # 采样（有放回，但我们会在后面去重）
                    try:
                        sampled_indices = np.random.choice(
                            segment_data.index, 
                            size=segment_samples, 
                            replace=False,  # 改为无放回采样
                            p=weights
                        )
                        unique_indices.update(sampled_indices)
                    except ValueError:
                        # 如果权重问题导致采样失败，使用均匀采样
                        warning(f"[波动率采样] 段{i+1}权重采样失败，使用均匀采样")
                        sampled_indices = np.random.choice(
                            segment_data.index, 
                            size=min(segment_samples, len(segment_data)), 
                            replace=False
                        )
                        unique_indices.update(sampled_indices)
            
            info(f"[波动率采样] 初步采样: {len(unique_indices)}行数据")
            
            # 调整采样数量到目标
            # 如果采样数量不足，从剩余数据中补充
            if len(unique_indices) < target_samples:
                additional_samples = target_samples - len(unique_indices)
                info(f"[波动率采样] 采样数量不足，需要补充{additional_samples}行")
                
                # 从剩余数据中选择
                remaining_indices = set(data.index) - unique_indices
                if remaining_indices:
                    # 计算剩余数据的波动率
                    remaining_data = data.loc[list(remaining_indices)].copy()
                    
                    # 尝试计算剩余数据的波动率
                    try:
                        # 确保索引连续
                        remaining_data = remaining_data.sort_index()
                        # 计算收益率和波动率
                        try:
                            remaining_returns = np.log(remaining_data[self.volatility_field] / 
                                                     remaining_data[self.volatility_field].shift(1))
                        except Exception:
                            remaining_returns = remaining_data[self.volatility_field].pct_change()
                        
                        # 使用滚动窗口的最后一个值作为每个数据点的波动率
                        remaining_vol = remaining_returns.rolling(window=self.volatility_window).std() * np.sqrt(252)
                        remaining_data['volatility'] = remaining_vol
                        
                        # 删除NaN值
                        valid_data = remaining_data.dropna(subset=['volatility'])
                        
                        if len(valid_data) > 0:
                            # 基于波动率权重采样
                            weights = valid_data['volatility'] / valid_data['volatility'].sum()
                            additional_indices = np.random.choice(
                                valid_data.index, 
                                size=min(additional_samples, len(valid_data)), 
                                replace=False, 
                                p=weights
                            )
                        else:
                            # 如果没有有效数据，使用均匀采样
                            additional_indices = np.random.choice(
                                list(remaining_indices), 
                                size=min(additional_samples, len(remaining_indices)), 
                                replace=False
                            )
                    except Exception as e:
                        # 如果计算失败，使用均匀采样
                        warning(f"[波动率采样] 补充采样计算失败，使用均匀采样: {str(e)}")
                        additional_indices = np.random.choice(
                            list(remaining_indices), 
                            size=min(additional_samples, len(remaining_indices)), 
                            replace=False
                        )
                    
                    unique_indices.update(additional_indices)
                    info(f"[波动率采样] 补充采样完成，当前样本数: {len(unique_indices)}")
            
            # 如果采样数量超过目标，随机删除一些
            elif len(unique_indices) > target_samples:
                excess = len(unique_indices) - target_samples
                info(f"[波动率采样] 采样数量超出目标，需要删除{excess}行")
                
                # 转换为列表并随机排序
                indices_list = list(unique_indices)
                np.random.shuffle(indices_list)
                
                # 保留前target_samples个
                unique_indices = set(indices_list[:target_samples])
                info(f"[波动率采样] 调整后样本数: {len(unique_indices)}")
            
            # 获取采样数据并按时间排序
            sampled_data = data.loc[list(unique_indices)].sort_index()
            
            # 计算采样数据的波动率均值
            sample_vol_mean = data_subset.loc[list(set(unique_indices) & set(data_subset.index)), 'volatility'].mean()
            original_vol_mean = data_subset['volatility'].mean()
            
            info(f"[波动率采样] 采样完成！最终采样数据量: {len(sampled_data)}行")
            info(f"[波动率采样] 采样数据波动率均值: {sample_vol_mean:.6f}")
            info(f"[波动率采样] 原始数据波动率均值: {original_vol_mean:.6f}")
            info(f"[波动率采样] 波动率强化因子: {sample_vol_mean / original_vol_mean:.2f}倍")
            
            return sampled_data
            
        except Exception as e:
            error(f"[波动率采样] 采样过程发生错误: {str(e)}")
            # 返回原始数据的一个子集
            return data.sample(min(self.min_samples, len(data))).sort_index()
    
    def sample_with_strategy(self, data: pd.DataFrame, strategy: str, target_samples: int = None) -> pd.DataFrame:
        """
        根据指定的策略进行数据采样
        
        Args:
            data: 输入的数据框
            strategy: 采样策略名称
            target_samples: 目标采样数量，如果为None则使用配置中的值
            
        Returns:
            采样后的数据框
        """
        try:
            info(f"[策略采样] 开始使用{strategy}策略采样数据")
            
            # 如果没有指定采样数量，使用默认值
            if target_samples is None:
                target_samples = self.target_samples
            target_samples = min(max(target_samples, self.min_samples), self.max_samples)
            target_samples = min(target_samples, len(data))
            
            # 根据不同策略执行采样
            if strategy == "volatility":
                # 使用波动率采样（与sample_by_volatility方法相同）
                return self.sample_by_volatility(data, target_samples)
            elif strategy == "random":
                # 随机采样
                info(f"[策略采样] 使用随机采样，目标{target_samples}行")
                return data.sample(target_samples, random_state=42).sort_index()
            elif strategy == "recent":
                # 最近数据采样
                info(f"[策略采样] 使用最近数据采样，目标{target_samples}行")
                return data.tail(target_samples)
            elif strategy == "first":
                # 最早数据采样
                info(f"[策略采样] 使用最早数据采样，目标{target_samples}行")
                return data.head(target_samples)
            else:
                # 默认使用波动率采样
                warning(f"[策略采样] 未知策略'{strategy}'，默认使用波动率采样")
                return self.sample_by_volatility(data, target_samples)
                
        except Exception as e:
            error(f"[策略采样] 采样过程发生错误: {str(e)}")
            # 返回原始数据的一个子集
            return data.sample(min(self.min_samples, len(data))).sort_index()