import os
import sys
import time
import importlib.util
import tempfile
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# 导入我们创建的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from src.core.data_reader import DataReader
from src.core.backtester import Backtest, Strategy
from src.core.backtest_analyzer import BacktestAnalyzer
from src.utils.ai_strategy_generator import AIStrategyGenerator

class AITradingWorkflow:
    """
    AI交易策略工作流，串联数据读取、策略生成、回测和优化过程
    精简版：保留多文件输入功能，将具体处理交给大模型
    """
    
    def __init__(self, config: Optional[Dict] = None, config_file: Optional[str] = None):
        """
        初始化工作流
        
        Args:
            config: 工作流配置参数字典
            config_file: 配置文件路径（JSON格式）
        """
        # 默认配置
        default_config = {
            'main_data_file': 'BINANCE_BTCUSDT_1D.csv',  # 主回测数据文件
            'main_data_description': '比特币日线数据',  # 主数据描述
            'additional_data_files': [],  # 额外数据文件列表
            'additional_data_descriptions': [],  # 额外数据描述列表
            'initial_capital': 10000.0,  # 初始资金
            'commission_rate': 0.001,  # 佣金率
            'max_optimization_rounds': 3,  # 最大优化轮数
            'use_reasoning': True,  # 使用思考模式
            'api_key': None,  # DeepSeek API密钥
            'output_dir': f'output_{datetime.now().strftime("%Y%m%d_%H%M%S")}',  # 自动生成带时间戳的输出目录
            'data_directory': '',  # 数据文件目录
            'run_all_steps': True,  # 是否运行所有步骤
            'steps_to_run': ['load_data', 'analyze_data', 'generate_initial_strategy', 'run_optimization_cycle']  # 要运行的步骤列表
        }
        
        # 合并配置
        self.config = default_config.copy()
        
        # 1. 从配置文件加载配置（如果提供）
        if config_file:
            file_config = self._load_config_file(config_file)
            if file_config:
                print(f"[配置] 从配置文件 {config_file} 加载配置")
                # 忽略不再使用的配置项
                # if 'correlation_analysis' in file_config:
                #     print("[配置] 注意: correlation_analysis配置项已不再使用")
                #     file_config.pop('correlation_analysis')
                self.config.update(file_config)
        
        # 2. 从传入的字典更新配置（优先级高于文件配置）
        if config:
            print("[配置] 从传入的配置字典更新配置")
            self.config.update(config)
        
        # 不再需要处理旧配置的向后兼容性
        
        # 4. 加载API密钥
        self._load_api_key()
        
        # 5. 验证必需的配置项
        self._validate_config()
        
        # 6. 标准化配置项
        self._normalize_config()
        
        # 确保输出目录存在
        self.output_dir = self.config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置日志文件路径
        self.log_file = os.path.join(self.output_dir, f"workflow_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        # 工作流状态
        self.data = None  # 单一数据集，用于回测
        self.data_sets = {}  # 多数据集字典
        self.analysis_result = None
        self.current_strategy_code = None
        self.current_strategy_description = None
        self.backtest_results = []
        self.optimization_history = []
        
        # 初始化数据读取器和AI生成器
        self.reader = DataReader(data_dir=self.config.get('data_directory', ''))
        self.generator = AIStrategyGenerator(
            api_key=self.config['api_key'],
            use_reasoning=self.config['use_reasoning']
        )
        
        # 记录日志
        self._log("工作流初始化完成")
    
    def _log(self, message: str):
        """
        记录日志
        
        Args:
            message: 日志消息
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        # 写入日志文件
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_message + '\n')
        except Exception as e:
            # 如果无法写入日志文件，只打印到控制台
            print(f"无法写入日志文件: {e}")
    
    def _load_config_file(self, config_file: str) -> Optional[Dict]:
        """
        从文件加载配置
        
        Args:
            config_file: 配置文件路径
            
        Returns:
            加载的配置字典，如果失败返回None
        """
        try:
            # 如果是相对路径，相对于当前文件所在目录
            if not os.path.isabs(config_file):
                config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_file)
            
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"警告: 无法读取配置文件 {config_file}: {e}")
            return None
    
    def _load_api_key(self):
        """
        加载API密钥，支持多种方式
        """
        if not self.config['api_key']:
            try:
                # 1. 尝试从config目录读取API密钥
                api_key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'api_key.json')
                if os.path.exists(api_key_path):
                    with open(api_key_path, 'r', encoding='utf-8') as f:
                        key_config = json.load(f)
                        # 支持多种可能的键名
                        self.config['api_key'] = key_config.get('deepseek_api_key') or \
                                              key_config.get('deepseek') or \
                                              key_config.get('api_key')
                
                # 2. 如果配置中指定了API密钥文件，尝试从那里读取
                elif 'api_key_file' in self.config:
                    key_file = self.config['api_key_file']
                    if not os.path.isabs(key_file):
                        key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), key_file)
                    if os.path.exists(key_file):
                        with open(key_file, 'r', encoding='utf-8') as f:
                            key_config = json.load(f)
                            self.config['api_key'] = key_config.get('deepseek_api_key') or \
                                                  key_config.get('deepseek') or \
                                                  key_config.get('api_key')
            except (FileNotFoundError, json.JSONDecodeError) as e:
                self._log(f"警告: 无法读取API密钥文件: {e}")
    
    def _validate_config(self):
        """
        验证配置的有效性
        """
        # 确保额外数据描述与额外数据文件数量一致
        if len(self.config.get('additional_data_files', [])) != len(self.config.get('additional_data_descriptions', [])):
            self._log("警告: 额外数据描述数量与额外数据文件数量不一致，自动补充描述")
            descriptions = self.config.get('additional_data_descriptions', [])
            while len(descriptions) < len(self.config.get('additional_data_files', [])):
                descriptions.append(f"额外数据集_{len(descriptions) + 1}")
            self.config['additional_data_descriptions'] = descriptions[:len(self.config.get('additional_data_files', []))]
    
    def _normalize_config(self):
        """
        标准化配置项
        """
        # 标准化输出目录路径
        if not os.path.isabs(self.config['output_dir']):
            self.config['output_dir'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.config['output_dir'])
        
        # 标准化数据目录路径
        if self.config.get('data_directory') and not os.path.isabs(self.config['data_directory']):
            self.config['data_directory'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.config['data_directory'])
        
        # 确保steps_to_run是列表
        if isinstance(self.config.get('steps_to_run'), str):
            self.config['steps_to_run'] = [step.strip() for step in self.config['steps_to_run'].split(',')]
    
    def load_data(self) -> bool:
        """
        加载和准备数据，支持主数据集和额外数据集
        
        Returns:
            bool: 是否成功
        """
        try:
            # 加载主回测数据集
            main_data_file = self.config['main_data_file']
            main_data_description = self.config['main_data_description']
            
            self._log(f"开始加载主回测数据集: {main_data_file} ({main_data_description})")
            
            # 读取主数据集
            df = self.reader.read_csv_file(main_data_file)
            
            # 准备数据
            df = self.reader.prepare_data(df)
            
            # 存储主数据集
            main_symbol = main_data_file.split('.')[0].strip()  # 提取股票/加密货币代码
            self.data_sets[main_symbol] = {
                'data': df,
                'file': main_data_file,
                'description': main_data_description
            }
            
            self._log(f"  - 主数据集加载完成，共 {len(df)} 条记录")
            self._log(f"  - 数据时间范围: {df.index[0]} 到 {df.index[-1]}")
            
            # 设置回测数据集为主数据集
            self.data = df
            
            # 加载额外数据集
            additional_data_files = self.config.get('additional_data_files', [])
            additional_data_descriptions = self.config.get('additional_data_descriptions', [])
            
            if additional_data_files:
                self._log(f"开始加载 {len(additional_data_files)} 个额外数据集")
                
                # 确保描述列表与文件列表长度一致
                if len(additional_data_descriptions) < len(additional_data_files):
                    # 用文件名填充缺少的描述
                    for i in range(len(additional_data_descriptions), len(additional_data_files)):
                        additional_data_descriptions.append(f"额外数据集 {i+1}")
                    self.config['additional_data_descriptions'] = additional_data_descriptions
                
                # 读取所有额外数据集
                for i, (data_file, data_desc) in enumerate(zip(additional_data_files, additional_data_descriptions)):
                    self._log(f"加载额外数据文件 {i+1}/{len(additional_data_files)}: {data_file} ({data_desc})")
                    
                    # 读取数据
                    df = self.reader.read_csv_file(data_file)
                    
                    # 准备数据
                    df = self.reader.prepare_data(df)
                    
                    # 存储额外数据集
                    symbol = data_file.split('.')[0].strip()  # 提取股票/加密货币代码
                    # 避免与主数据集冲突
                    if symbol in self.data_sets:
                        symbol = f"{symbol}_additional_{i}"
                    
                    self.data_sets[symbol] = {
                        'data': df,
                        'file': data_file,
                        'description': data_desc
                    }
                    
                    self._log(f"  - 加载完成，共 {len(df)} 条记录")
                    self._log(f"  - 数据时间范围: {df.index[0]} 到 {df.index[-1]}")
            
            # 保存数据摘要
            self._save_data_summaries()
            
            return True
        except Exception as e:
            self._log(f"数据加载失败: {e}")
            import traceback
            self._log(f"错误详情: {traceback.format_exc()}")
            return False
            
    def _save_data_summaries(self):
        """
        保存所有数据集的摘要信息
        """
        summary_file = os.path.join(self.output_dir, 'data_summary.txt')
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"数据集摘要 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for symbol, dataset in self.data_sets.items():
                f.write(f"## {symbol}: {dataset['description']}\n")
                f.write(f"文件: {dataset['file']}\n")
                f.write(f"记录数: {len(dataset['data'])} 条\n")
                f.write(f"时间范围: {dataset['data'].index[0]} 到 {dataset['data'].index[-1]}\n")
                
                # 添加数据摘要
                data_summary = self.reader.get_data_summary(dataset['data'])
                f.write("\n" + data_summary + "\n\n")
                f.write("-" * 50 + "\n\n")
        
        self._log(f"数据摘要已保存到 {summary_file}")
    
    def analyze_data(self) -> bool:
        """
        使用AI分析数据，支持多数据集信息汇总
        
        Returns:
            bool: 是否成功
        """
        try:
            if not self.data_sets:
                self._log("错误: 数据尚未加载")
                return False
            
            # 首先进行单一数据集分析（使用默认回测数据集）
            self._log("开始使用AI分析默认数据集...")
            self.analysis_result = self.generator.analyze_data(
                self.data,
                self.config['main_data_description']
            )
            
            # 保存单一数据集分析结果
            analysis_file = os.path.join(self.output_dir, 'data_analysis_result.txt')
            with open(analysis_file, 'w', encoding='utf-8') as f:
                f.write(f"数据分析结果 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(self.analysis_result)
            
            # 保存整体数据摘要
            all_summaries = """
数据汇总信息 - """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n"
            for symbol, dataset in self.data_sets.items():
                all_summaries += f"\n=== {symbol} 数据集信息 ===\n"
                all_summaries += f"文件: {dataset['file']}\n"
                all_summaries += f"描述: {dataset['description']}\n"
                all_summaries += f"数据点数量: {len(dataset['data'])} 条\n"
                all_summaries += f"时间范围: {dataset['data'].index.min()} 到 {dataset['data'].index.max()}\n\n"
            
            summary_path = os.path.join(self.output_dir, "data_summary.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(all_summaries)
            
            # 如果有多个数据集，创建多数据集信息汇总
            if len(self.data_sets) > 1:
                self._log("生成多数据集信息汇总...")
                multi_data_summary = self._generate_multi_data_summary()
                
                # 保存多数据集信息
                multi_data_file = os.path.join(self.output_dir, 'multi_data_summary.txt')
                with open(multi_data_file, 'w', encoding='utf-8') as f:
                    f.write(f"多数据集信息汇总 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(multi_data_summary)
            
            self._log("数据AI分析完成")
            return True
        except Exception as e:
            self._log(f"数据分析失败: {e}")
            import traceback
            self._log(f"错误详情: {traceback.format_exc()}")
            return False
    
    def _generate_multi_data_summary(self) -> str:
        """
        生成多数据集信息汇总
        
        Returns:
            str: 多数据集信息汇总文本
        """
        # 准备所有数据集的基本信息
        symbols = list(self.data_sets.keys())
        
        result = []
        result.append("多数据集信息汇总")
        result.append(f"参与分析的数据集数量: {len(symbols)}")
        result.append(f"参与分析的数据集: {', '.join(symbols)}")
        result.append("")
        
        # 收集每个数据集的基本统计信息
        for symbol, dataset in self.data_sets.items():
            result.append(f"\n{symbol} 数据集详情:")
            result.append(f"  - 描述: {dataset['description']}")
            result.append(f"  - 文件: {dataset['file']}")
            result.append(f"  - 数据点数量: {len(dataset['data'])}")
            result.append(f"  - 时间范围: {dataset['data'].index.min()} 到 {dataset['data'].index.max()}")
            
            # 计算基本统计
            df = dataset['data']
            result.append(f"  - 价格范围: {df['close'].min():.2f} 到 {df['close'].max():.2f}")
            result.append(f"  - 平均交易量: {df['volume'].mean():.2f}")
        
        return "\n".join(result)
    
    def generate_initial_strategy(self) -> bool:
        """
        生成初始策略
        
        Returns:
            bool: 是否成功
        """
        try:
            if self.data is None:
                self._log("错误: 数据尚未加载")
                return False
            
            self._log("开始生成初始交易策略...")
            
            # 确保分析结果有效
            analysis_result = self.analysis_result if hasattr(self, 'analysis_result') and self.analysis_result else "暂无详细分析"
            
            # 获取当前文件所在目录的绝对路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 添加路径提示到analysis_result，确保生成的策略有正确的导入路径
            path_hint = f"\n\n重要提示: 生成的策略代码必须包含以下导入设置，以确保正确找到backtester模块：\n\n```python\nimport sys\nimport os\n# 添加项目根目录和src目录到Python路径\nsys.path.append('{current_dir}')\nsys.path.append(os.path.join('{current_dir}', 'src'))\n\n# 正确导入Strategy类\nfrom src.core.backtester import Strategy\n```\n\n请确保在策略代码的开头包含这些导入语句。"
            
            # 准备多数据集信息（如果有）
            multi_data_info = ""
            if len(self.data_sets) > 1:
                multi_data_info = "\n\n可用的多数据集信息:\n"
                for symbol, dataset in self.data_sets.items():
                    multi_data_info += f"- {symbol}: {dataset['description']}（已加载）\n"
                multi_data_info += "\n注意：默认回测将使用第一个数据集，但您可以在策略中引用其他数据集进行多资产分析。"
            
            self.current_strategy_code, self.current_strategy_description = self.generator.generate_strategy(
                self.data,
                self.config['main_data_description'],
                analysis_result + path_hint + multi_data_info
            )
            
            # 验证策略代码
            if not self.generator.validate_strategy_code(self.current_strategy_code):
                self._log("策略代码验证失败")
                # 提供一个备用策略作为安全措施
                self.current_strategy_code = self._get_fallback_strategy()
                self.current_strategy_description = "备用基础策略"
            
            # 保存策略
            strategy_file = os.path.join(self.output_dir, 'initial_strategy.py')
            self.generator.save_strategy(
                self.current_strategy_code,
                self.current_strategy_description,
                strategy_file
            )
            
            self._log("初始策略生成完成")
            return True
        except Exception as e:
            self._log(f"策略生成失败: {e}")
            # 设置一个备用策略
            self.current_strategy_code = self._get_fallback_strategy()
            self.current_strategy_description = "备用基础策略"
            strategy_file = os.path.join(self.output_dir, 'initial_strategy.py')
            self.generator.save_strategy(
                self.current_strategy_code,
                self.current_strategy_description,
                strategy_file
            )
            return True
            
    def _get_fallback_strategy(self) -> str:
        """
        获取备用策略代码
        
        Returns:
            str: 备用策略代码
        """
        # 获取当前文件所在目录的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 构建策略代码
        strategy_code = f"""
import sys
import os
# 添加项目根目录和src目录到Python路径
sys.path.append('{current_dir}')
sys.path.append(os.path.join('{current_dir}', 'src'))

# 正确导入Strategy类
from src.core.backtester import Strategy
import pandas as pd
import numpy as np

class GeneratedStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self.position_size = 0.1
    
    def initialize(self, data):
        # 初始化方法，接受data参数
        pass
    
    def on_bar(self, index, row, data):
        # on_bar方法，接受index、row和data三个参数，返回空字符串（无交易信号）
        return ""
"""
        
        return strategy_code
    
    def backtest_strategy(self, strategy_code: str = None, round_num: int = 0) -> Dict:
        """
        回测策略
        
        Args:
            strategy_code: 策略代码，如果为None则使用当前策略
            round_num: 优化轮数
            
        Returns:
            Dict: 回测结果
        """
        try:
            if self.data is None:
                self._log("错误: 数据尚未加载")
                return None
            
            # 使用指定的策略代码或当前策略代码
            code_to_use = strategy_code or self.current_strategy_code
            if not code_to_use:
                self._log("错误: 没有可用的策略代码")
                return None
            
            self._log(f"开始回测策略 (轮数: {round_num})...")
            
            # 获取当前文件所在目录的绝对路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 确保策略代码开头有正确的导入语句
            # 移除所有可能存在的导入语句
            import re
            code_to_use = re.sub(r'import\s+sys\s+.*?(?=class|#|$)', '', code_to_use, flags=re.DOTALL)
            code_to_use = re.sub(r'from\s+backtester\s+import\s+Strategy[\s\S]*?(?=class|#|$)', '', code_to_use, flags=re.DOTALL)
            code_to_use = re.sub(r'from\s+src.core.backtester\s+import\s+Strategy[\s\S]*?(?=class|#|$)', '', code_to_use, flags=re.DOTALL)
            
            # 添加正确的导入语句到代码开头
            import_lines = f"import sys\nimport os\nimport numpy as np\nimport pandas as pd\n# 添加项目根目录和src目录到Python路径\nsys.path.append('{current_dir}')\nsys.path.append(os.path.join('{current_dir}', 'src'))\n\n# 正确导入Strategy类\nfrom src.core.backtester import Strategy\n\n"
            code_to_use = import_lines + code_to_use
            self._log("已为策略代码添加正确的导入语句")
            
            # 使用临时文件加载策略类
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as temp_file:
                temp_file.write(code_to_use)
                temp_file_path = temp_file.name
            
            try:
                # 确保Python路径设置正确
                sys.path.append(current_dir)
                sys.path.append(os.path.join(current_dir, 'src'))
                
                # 动态导入策略类
                spec = importlib.util.spec_from_file_location("strategy_module", temp_file_path)
                strategy_module = importlib.util.module_from_spec(spec)
                # 设置模块的__file__和__package__属性
                strategy_module.__file__ = temp_file_path
                strategy_module.__package__ = "strategy_module"
                # 将模块添加到sys.modules中
                sys.modules["strategy_module"] = strategy_module
                
                # 执行模块
                try:
                    spec.loader.exec_module(strategy_module)
                except ImportError as e:
                    self._log(f"导入错误: {e}")
                    self._log(f"当前Python路径: {sys.path}")
                    raise
                
                # 获取策略类
                strategy_class = strategy_module.GeneratedStrategy
                
                # 创建策略实例
                strategy = strategy_class()
                
                # 运行回测
                backtest = Backtest(
                    self.data,
                    strategy,
                    initial_capital=self.config['initial_capital'],
                    commission_rate=self.config['commission_rate']
                )
                
                results = backtest.run()
                
                # 获取回测结果摘要
                summary = backtest.get_results_summary()
                
                # 使用分析器进行更详细的分析
                analyzer = BacktestAnalyzer(results)
                detailed_report = analyzer.generate_detailed_report()
                
                # 保存回测结果
                backtest_dir = os.path.join(self.output_dir, f"backtest_round_{round_num}")
                os.makedirs(backtest_dir, exist_ok=True)
                
                # 保存回测摘要
                summary_file = os.path.join(backtest_dir, 'backtest_summary.txt')
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(summary)
                
                # 保存详细报告
                report_file = os.path.join(backtest_dir, 'detailed_analysis.txt')
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(detailed_report)
                
                # 保存分析指标JSON
                analyzer.to_json(os.path.join(backtest_dir, 'analysis_metrics.json'))
                
                # 保存回测使用的策略代码
                strategy_file = os.path.join(backtest_dir, 'strategy.py')
                with open(strategy_file, 'w', encoding='utf-8') as f:
                    f.write(code_to_use)
                
                # 记录回测结果
                backtest_info = {
                    'round': round_num,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'total_return': results.get('total_return', 0),
                    'sharpe_ratio': results.get('sharpe_ratio', 0),
                    'max_drawdown': results.get('max_drawdown', 0),
                    'win_rate': results.get('win_rate', 0),
                    'dir': backtest_dir
                }
                
                self.backtest_results.append(backtest_info)
                
                self._log(f"回测完成，收益率: {backtest_info['total_return']:.2%}, 夏普比率: {backtest_info['sharpe_ratio']:.2f}")
                
                # 返回回测摘要（用于AI优化）
                return {
                    'summary': summary,
                    'detailed_report': detailed_report,
                    'metrics': analyzer.calculate_comprehensive_metrics(),
                    'strategy_code': code_to_use
                }
                
            finally:
                # 清理临时文件
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            self._log(f"回测失败: {e}")
            import traceback
            self._log(f"错误详情: {traceback.format_exc()}")
            return None
    
    def optimize_strategy(self, backtest_result: Dict, round_num: int) -> Tuple[str, str]:
        """
        优化策略
        
        Args:
            backtest_result: 回测结果
            round_num: 优化轮数
            
        Returns:
            Tuple[str, str]: (优化后的策略代码, 优化分析)
        """
        try:
            self._log(f"开始优化策略 (轮数: {round_num})...")
            
            # 获取当前文件所在目录的绝对路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 添加路径提示到分析结果，确保生成的策略有正确的导入路径
            path_hint = f"\n\n重要提示: 优化后的策略代码必须包含以下导入设置，以确保正确找到backtester模块：\n\n```python\nimport sys\nimport os\n# 添加项目根目录和src目录到Python路径\nsys.path.append('{current_dir}')\nsys.path.append(os.path.join('{current_dir}', 'src'))\n\n# 正确导入Strategy类\nfrom src.core.backtester import Strategy\n```\n\n请确保在策略代码的开头包含这些导入语句。"
            
            # 调用AI优化策略
            optimized_code, optimization_analysis = self.generator.optimize_strategy(
                strategy_code=backtest_result['strategy_code'],
                strategy_description=self.current_strategy_description,
                backtest_results=backtest_result['summary'] + path_hint,
                data_description=self.config['main_data_description']
            )
            
            # 验证优化后的策略代码
            if not self.generator.validate_strategy_code(optimized_code):
                self._log("优化后的策略代码验证失败")
                return None, None
            
            # 保存优化分析和代码
            optimize_dir = os.path.join(self.output_dir, f"optimization_round_{round_num}")
            os.makedirs(optimize_dir, exist_ok=True)
            
            # 保存优化分析
            analysis_file = os.path.join(optimize_dir, 'optimization_analysis.txt')
            with open(analysis_file, 'w', encoding='utf-8') as f:
                f.write(optimization_analysis)
            
            # 保存优化后的策略代码
            strategy_file = os.path.join(optimize_dir, 'optimized_strategy.py')
            with open(strategy_file, 'w', encoding='utf-8') as f:
                f.write(optimized_code)
            
            # 记录优化历史
            self.optimization_history.append({
                'round': round_num,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'dir': optimize_dir
            })
            
            self._log("策略优化完成")
            return optimized_code, optimization_analysis
            
        except Exception as e:
            self._log(f"策略优化失败: {e}")
            return None, None
    
    def run_optimization_cycle(self, max_rounds: int = None) -> bool:
        """
        运行优化循环
        
        Args:
            max_rounds: 最大优化轮数，如果为None则使用配置中的值
            
        Returns:
            bool: 是否成功
        """
        try:
            rounds = max_rounds or self.config['max_optimization_rounds']
            
            self._log(f"开始优化循环，共 {rounds} 轮")
            
            # 第一轮：回测初始策略
            backtest_result = self.backtest_strategy(round_num=0)
            if not backtest_result:
                return False
            
            # 保存第一轮结果
            best_result = backtest_result
            best_code = self.current_strategy_code
            
            # 进行多轮优化
            for i in range(1, rounds + 1):
                self._log(f"===== 优化轮次 {i}/{rounds} =====")
                
                # 优化策略
                optimized_code, _ = self.optimize_strategy(
                    backtest_result=backtest_result,
                    round_num=i
                )
                
                if not optimized_code:
                    self._log(f"第 {i} 轮优化失败，跳过")
                    continue
                
                # 回测优化后的策略
                new_backtest_result = self.backtest_strategy(
                    strategy_code=optimized_code,
                    round_num=i
                )
                
                if not new_backtest_result:
                    self._log(f"第 {i} 轮回测失败，跳过")
                    continue
                
                # 比较结果，更新最优结果
                current_sharpe = new_backtest_result['metrics'].get('sharpe_ratio', 0)
                best_sharpe = best_result['metrics'].get('sharpe_ratio', 0)
                
                if current_sharpe > best_sharpe:
                    best_result = new_backtest_result
                    best_code = optimized_code
                    self._log(f"第 {i} 轮优化成功，夏普比率从 {best_sharpe:.2f} 提升到 {current_sharpe:.2f}")
                else:
                    self._log(f"第 {i} 轮优化未能提高性能，保持当前最优策略")
                
                # 更新回测结果用于下一轮优化
                backtest_result = new_backtest_result
                self.current_strategy_code = optimized_code
                
                # 短暂休息，避免API调用过于频繁
                time.sleep(5)
            
            # 保存最终最优策略
            final_strategy_file = os.path.join(self.output_dir, 'final_optimized_strategy.py')
            with open(final_strategy_file, 'w', encoding='utf-8') as f:
                f.write(best_code)
            
            self._log(f"优化循环完成，最优策略已保存到 {final_strategy_file}")
            
            # 生成优化总结报告
            self._generate_optimization_summary()
            
            return True
            
        except Exception as e:
            self._log(f"优化循环失败: {e}")
            return False
    
    def _generate_optimization_summary(self):
        """
        生成优化总结报告
        """
        try:
            summary_file = os.path.join(self.output_dir, 'optimization_summary.txt')
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("===== AI交易策略优化总结报告 =====\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # 数据信息
                f.write("## 数据信息\n")
                f.write(f"- 主数据文件: {self.config.get('main_data_file', '未知')}\n")
                f.write(f"- 主数据描述: {self.config.get('main_data_description', '未知')}\n")
                if self.data is not None:
                    f.write(f"- 数据量: {len(self.data)} 条记录\n")
                    f.write(f"- 时间范围: {self.data.index[0]} 到 {self.data.index[-1]}\n\n")
                
                # 多数据集信息（如果有）
                if len(self.data_sets) > 1:
                    f.write("## 多数据集信息\n")
                    f.write(f"- 使用的数据集数量: {len(self.data_sets)}\n")
                    for symbol, dataset in self.data_sets.items():
                        f.write(f"  - {symbol}: {dataset['description']}\n")
                    f.write("\n")
                
                # 回测结果对比
                if self.backtest_results:
                    f.write("## 回测结果对比\n")
                    f.write("轮次,时间,总收益率,夏普比率,最大回撤,胜率\n")
                    
                    for result in self.backtest_results:
                        f.write(f"{result['round']},")
                        f.write(f"{result['timestamp']},")
                        f.write(f"{result['total_return']:.2%},")
                        f.write(f"{result['sharpe_ratio']:.2f},")
                        f.write(f"{result['max_drawdown']:.2%},")
                        f.write(f"{result['win_rate']:.2%}\n")
                    
                    # 标记最优结果
                    if len(self.backtest_results) > 1:
                        best_result = max(self.backtest_results, key=lambda x: x['sharpe_ratio'])
                        f.write(f"\n最优结果 (轮次 {best_result['round']}):\n")
                        f.write(f"- 总收益率: {best_result['total_return']:.2%}\n")
                        f.write(f"- 夏普比率: {best_result['sharpe_ratio']:.2f}\n")
                        f.write(f"- 最大回撤: {best_result['max_drawdown']:.2%}\n")
                
                # 优化过程
                f.write("\n## 优化过程\n")
                f.write(f"- 总优化轮数: {self.config['max_optimization_rounds']}\n")
                f.write(f"- 成功优化轮数: {len(self.optimization_history)}\n\n")
                
                # 结论和建议
                f.write("## 结论与建议\n")
                if self.backtest_results:
                    best_result = max(self.backtest_results, key=lambda x: x['sharpe_ratio'])
                    
                    if best_result['sharpe_ratio'] > 1.0 and best_result['total_return'] > 0:
                        f.write("✅ 策略表现优秀，建议进一步验证后考虑实盘测试\n")
                    elif best_result['sharpe_ratio'] > 0.5 and best_result['total_return'] > 0:
                        f.write("✅ 策略表现良好，具有一定的实用性\n")
                    elif best_result['total_return'] > 0:
                        f.write("⚠️  策略能够盈利但风险较高，建议加强风险控制\n")
                    else:
                        f.write("❌ 策略未能产生正收益，建议重新设计\n")
                    
                    f.write("\n建议:\n")
                    f.write("1. 在不同时间段的数据上进行验证\n")
                    f.write("2. 考虑添加仓位管理和止损策略\n")
                    f.write("3. 监控实盘表现，及时调整参数\n")
            
            self._log(f"优化总结报告已保存到 {summary_file}")
            
        except Exception as e:
            self._log(f"生成优化总结失败: {e}")
    
    def run_full_workflow(self) -> bool:
        """
        运行完整的AI交易策略工作流
        根据配置选择要运行的步骤
        
        Returns:
            bool: 工作流是否成功运行
        """
        try:
            self._log("🚀 开始运行AI交易策略工作流...")
            
            # 打印配置摘要
            self._log(f"配置摘要: 主数据文件={self.config.get('main_data_file', '未知')}, 额外数据文件={len(self.config.get('additional_data_files', []))}个, 优化轮数={self.config['max_optimization_rounds']}")
            
            # 定义工作流步骤映射
            workflow_steps = {
                'load_data': {
                    'method': self.load_data,
                    'description': '加载数据'
                },
                'analyze_data': {
                    'method': self.analyze_data,
                    'description': '分析数据'
                },
                'generate_initial_strategy': {
                    'method': self.generate_initial_strategy,
                    'description': '生成初始策略'
                },
                'run_optimization_cycle': {
                    'method': self.run_optimization_cycle,
                    'description': '运行优化循环'
                }
            }
            
            # 确定要运行的步骤
            steps_to_run = self.config.get('steps_to_run', [])
            if not steps_to_run or self.config.get('run_all_steps', True):
                # 如果没有指定步骤或run_all_steps为True，则运行所有步骤
                steps_to_run = list(workflow_steps.keys())
            
            # 按定义的顺序运行步骤
            ordered_steps = ['load_data', 'analyze_data', 'generate_initial_strategy', 'run_optimization_cycle']
            filtered_steps = [step for step in ordered_steps if step in steps_to_run]
            
            self._log(f"将按以下顺序运行步骤: {', '.join(filtered_steps)}")
            
            # 执行每个步骤
            for step_name in filtered_steps:
                step_info = workflow_steps.get(step_name)
                if not step_info:
                    self._log(f"警告: 未知步骤 '{step_name}'，跳过")
                    continue
                
                self._log(f"\n📊 开始 {step_info['description']}...")
                start_time = time.time()
                
                try:
                    success = step_info['method']()
                    
                    if not success:
                        self._log(f"❌ {step_info['description']}失败")
                        # 如果步骤失败，是否继续取决于配置
                        if self.config.get('continue_on_failure', False):
                            self._log("配置允许继续执行，将尝试运行下一个步骤")
                        else:
                            return False
                    else:
                        duration = time.time() - start_time
                        self._log(f"✅ {step_info['description']}成功完成 (耗时: {duration:.2f}秒)")
                except Exception as e:
                    self._log(f"❌ {step_info['description']}执行异常: {e}")
                    import traceback
                    self._log(f"错误详情: {traceback.format_exc()}")
                    if not self.config.get('continue_on_failure', False):
                        return False
            
            # 生成最终的优化总结报告
            if 'run_optimization_cycle' in filtered_steps:
                self._generate_optimization_summary()
            
            self._log("🎉 AI交易策略工作流运行完成！")
            self._log(f"所有结果已保存到: {self.output_dir}")
            return True
            
        except Exception as e:
            self._log(f"工作流运行失败: {e}")
            import traceback
            self._log(f"错误详情: {traceback.format_exc()}")
            return False

def main():
    """
    主函数，支持从命令行参数读取配置文件路径
    
    用法示例:
        python ai_trading_workflow.py
        python ai_trading_workflow.py --config my_config.json
        python ai_trading_workflow.py --config /path/to/config.json
    """
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='AI交易策略工作流')
    parser.add_argument('--config', type=str, help='配置文件路径（JSON格式）')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 如果指定了配置文件，尝试加载
    config_file = args.config
    
    # 基础配置（优先级低于配置文件）
    base_config = {
        # 精简版：移除了不必要的配置项
    }
    
    print("🎯 AI交易策略工作流启动")
    
    # 创建并运行工作流
    workflow = AITradingWorkflow(config=base_config, config_file=config_file)
    
    # 如果没有指定配置文件，打印提示信息
    if not config_file:
        print("\nℹ️  未指定配置文件，使用默认配置")
        print("   您可以通过 --config 参数指定配置文件路径")
        print("   例如: python ai_trading_workflow.py --config my_config.json")
        print(f"\n📊 数据文件: {workflow.config['main_data_file']}")
        print(f"💰 初始资金: {workflow.config['initial_capital']}")
        print(f"🔄 优化轮数: {workflow.config['max_optimization_rounds']}")
        print(f"📁 输出目录: {workflow.output_dir}")
    else:
        print(f"\n📄 使用配置文件: {config_file}")
        print(f"📊 数据文件: {workflow.config['main_data_file']}")
        print(f"📁 输出目录: {workflow.output_dir}")
    
    # 运行工作流
    success = workflow.run_full_workflow()
    
    # 打印结果信息
    if success:
        print(f"\n✅ 工作流成功完成！所有结果已保存到: {workflow.output_dir}")
        print("\n🔍 下一步建议:")
        print("   1. 查看优化总结报告了解策略表现")
        print("   2. 检查生成的策略代码是否符合预期")
        print("   3. 通过配置文件自定义更多参数进行实验")
    else:
        print("\n❌ 工作流运行失败，请查看日志获取详细信息")

if __name__ == "__main__":
    main()