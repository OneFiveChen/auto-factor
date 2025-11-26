import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from src.utils.deepseek_client import DeepSeekClient
from src.utils.data_sampler import DataSampler
from src.utils.logger import Logger, log, info, warning, error, debug, critical, set_global_log_file

class AIStrategyGenerator:
    """
    AI策略生成器，用于与DeepSeek API交互，分析数据并生成交易策略代码
    """
    
    def __init__(self, api_key: Optional[str] = None, use_reasoning: bool = True,
                 use_volatility_sampling: bool = True, target_samples: int = 30,
                 sampling_strategy: str = 'volatility', volatility_window: int = 20):
        """
        初始化AI策略生成器
        
        Args:
            api_key: DeepSeek API密钥
            use_reasoning: 是否使用思考模式
            use_volatility_sampling: 是否使用基于波动率的采样
            target_samples: 目标采样点数
            sampling_strategy: 采样策略
            volatility_window: 波动率计算窗口
        """
        self.client = DeepSeekClient(api_key=api_key, use_reasoning=use_reasoning)
        self.use_reasoning = use_reasoning
        self.use_volatility_sampling = use_volatility_sampling
        self.target_samples = target_samples
        self.sampling_strategy = sampling_strategy
        self.volatility_window = volatility_window
        self.sampler = DataSampler(config={'volatility_window': volatility_window})
        # 初始化对话历史，用于多轮对话，设置系统提示
        self.client.start_conversation("""
        你是一位专业的量化交易和数据分析专家，精通金融市场分析、统计建模和算法开发。
        请提供专业、深入且实用的分析和建议。
        """)
    
    def send_prompt(self, prompt: str) -> str:
        """
        直接发送提示词给AI并获取回复
        
        Args:
            prompt: 要发送的提示词
            
        Returns:
            str: AI的回复内容
        """
        # 默认系统提示词
        system_prompt = """
        你是一位专业的量化交易和数据分析专家，精通金融市场分析、统计建模和算法开发。
        请提供专业、深入且实用的分析和建议。
        """
        
        # 使用多轮对话方式调用API
        response = self.client.send_message_round(
            prompt=prompt,
            use_reasoning=self.use_reasoning,
            stream=True
        )
        
        return response
    
    def analyze_data(self, data: pd.DataFrame, data_description: str = "",
                    use_volatility_sampling: Optional[bool] = None,
                    target_samples: Optional[int] = None,
                    sampling_strategy: str = "volatility") -> str:
        """
        让AI分析数据，理解市场特性和潜在的交易机会
        
        Args:
            data: 要分析的K线数据
            data_description: 数据的额外描述信息（如资产名称、时间周期等）
            use_volatility_sampling: 是否使用基于波动率的采样（覆盖实例配置）
            target_samples: 目标采样点数（覆盖实例配置）
            sampling_strategy: 采样策略，可选值：'volatility'(基于波动率), 'uniform'(均匀采样), 'head'(取前N个)
            
        Returns:
            str: AI的数据分析结果
        """
        info("=== 开始数据分析 ===")
        info(f"原始数据总行数: {len(data)}")
        
        # 确定是否使用波动率采样
        if use_volatility_sampling is None:
            use_volatility_sampling = self.use_volatility_sampling
            
        info(f"是否使用波动率采样: {use_volatility_sampling}")
        
        # 确定目标采样数
        if target_samples is None:
            target_samples = self.target_samples
        
        info(f"目标采样数: {target_samples}")
        info(f"采样策略: {sampling_strategy}")
        
        # 准备数据样本
        info("开始数据采样...")
        if use_volatility_sampling and sampling_strategy == "volatility":
            # 使用基于波动率的采样
            sample_df = self.sampler.sample_by_volatility(data, target_samples)
            sampling_info = f"基于波动率的采样（{len(sample_df)}行数据）"
        elif use_volatility_sampling:
            # 使用指定的采样策略
            sample_df = self.sampler.sample_with_strategy(data, sampling_strategy, target_samples)
            sampling_info = f"使用{sampling_strategy}策略采样（{len(sample_df)}行数据）"
        else:
            # 使用默认的head方式
            sample_df = data.head(target_samples)
            sampling_info = f"使用前{len(sample_df)}行数据"
        
        info(f"采样完成！采样数据量: {len(sample_df)}行")
        info(f"采样时间范围: {sample_df.index[0]} 到 {sample_df.index[-1]}")
        info("=== 开始准备AI分析 ===")
        
        sample_data = sample_df.to_string()
        data_summary = self._get_data_summary(data)
        
        # 构建提示词，使用format方法避免f-string空表达式问题
        data_desc_text = data_description if data_description else "暂无数据描述"
        prompt = """
# 数据分析任务

## 数据描述
{}

## 数据统计摘要
{}

## 数据样本（{}）
{}

## 任务要求
请你作为一位专业的量化交易分析师，对上述数据进行全面分析，并回答以下问题：
1. 数据的基本特性是什么？（波动率、趋势性、季节性等）
2. 基于这些特性，哪些类型的交易策略可能会有效？
3. 你观察到了哪些潜在的交易模式或信号？
4. 对于这种类型的数据，你建议使用哪些技术指标或特征？
5. 有哪些风险因素需要注意？

请提供详细、专业且有深度的分析，帮助后续的策略开发。
""".format(data_desc_text, data_summary, sampling_info, sample_data)
        
        # 系统提示词
        system_prompt = """
你是一位专业的量化交易分析师和策略研究员，精通金融市场、技术分析和量化交易算法。
你拥有丰富的交易经验和深厚的统计知识，能够从数据中发现有价值的模式和规律。
请基于提供的数据，给出客观、专业、深入的分析，避免过度简化或不切实际的结论。
"""
        
        # 使用多轮对话方式调用API
        response = self.client.send_message_round(
            prompt=prompt,
            use_reasoning=self.use_reasoning,
            stream=True  # 启用流式输出，提高用户体验
        )
        
        return response
    
    def generate_strategy(self, data: pd.DataFrame, data_description: str = "", 
                         analysis_result: Optional[str] = None,
                         use_volatility_sampling: Optional[bool] = None,
                         target_samples: Optional[int] = None,
                         sampling_strategy: str = "volatility") -> Tuple[str, str]:
        """
        生成交易策略代码
        
        Args:
            data: K线数据
            data_description: 数据描述
            analysis_result: 可选的数据分析结果
            use_volatility_sampling: 是否使用基于波动率的采样（覆盖实例配置）
            target_samples: 目标采样点数（覆盖实例配置）
            sampling_strategy: 采样策略，可选值：'volatility'(基于波动率), 'uniform'(均匀采样), 'head'(取前N个)
            
        Returns:
            Tuple[str, str]: (策略代码, 策略说明)
        """
        # 如果没有提供分析结果，则先执行分析
        if not analysis_result:
            analysis_result = self.analyze_data(data, data_description, 
                                              use_volatility_sampling=use_volatility_sampling,
                                              target_samples=target_samples,
                                              sampling_strategy=sampling_strategy)
        
        # 确定是否使用波动率采样
        if use_volatility_sampling is None:
            use_volatility_sampling = self.use_volatility_sampling
            
        # 确定目标采样数
        if target_samples is None:
            target_samples = self.target_samples
        
        # 准备数据样本和统计信息
        if use_volatility_sampling:
            # 使用基于波动率的采样
            sample_df = self.sampler.sample_with_strategy(data, sampling_strategy, min(target_samples, 20))
            sample_data = sample_df.to_string()
        else:
            # 使用默认的head方式
            sample_data = data.head(10).to_string()
        
        data_summary = self._get_data_summary(data)
        
        # 构建提示词，使用format方法避免f-string空表达式问题
        data_desc_text = data_description if data_description else "暂无数据描述"
        # 对所有参数进行花括号转义，避免被format方法错误解释
        escaped_data_desc = data_desc_text.replace('{', '{{').replace('}', '}}')
        debug(f"[DEBUG] 转义后的数据描述: {escaped_data_desc}")
        escaped_data_summary = data_summary.replace('{', '{{').replace('}', '}}')
        debug(f"[DEBUG] 转义后的数据摘要: {escaped_data_summary}")
        # escaped_sample_data = sample_data.replace('{', '{{').replace('}', '}}')
        # print(f"[DEBUG] 转义后的样本数据: {escaped_sample_data[:100]}...")  # 只打印前100个字符
        escaped_analysis_result = analysis_result.replace('{', '{{').replace('}', '}}')
        debug(f"[DEBUG] 转义后的分析结果: {escaped_analysis_result[:100]}...")  # 只打印前100个字符
        
        info("=================初始化策略请求prompt，请观察================")
        # 构建提示词，使用命名参数避免位置参数错误
        prompt = """
# 策略生成任务

## 数据描述
{data_desc}

## 数据统计摘要
{data_sum}

## 数据样本请参考之前给你发送的数据

## 数据分析结果
{analysis}

## 任务要求
请基于上述数据和分析，设计一个交易策略，并按照以下格式返回：

### 策略说明
[在这里详细描述你的策略逻辑、参数选择理由和预期表现]

### 策略代码
```python
# 策略类名必须为GeneratedStrategy，并且继承自Strategy基类
from backtester import Strategy
import pandas as pd
import numpy as np

class GeneratedStrategy(Strategy):
    '''
    [在这里添加策略的文档字符串]
    '''
    
    def __init__(self, params=None):
        super().__init__(params)
        # 在这里设置默认参数
        self.params = params 
        # 请为策略指定一个有意义的名称
        self.name = "[策略名称]"
    
    def initialize(self, data):
        # 在初始化方法中计算所需的技术指标或特征
        # 例如：
        # self.sma_fast = data['close'].rolling(window=20).mean()
        # self.sma_slow = data['close'].rolling(window=50).mean()
        pass
    
    def on_bar(self, index, row, data):
        # 在这里实现策略逻辑
        # 返回 'BUY', 'SELL' 或空字符串 ''（无操作）
        # 确保在返回信号前有足够的数据点进行计算
        # 例如：
        # if index < 50:  # 确保有足够的数据点
        #     return ''
        # 
        # if self.sma_fast.iloc[index] > self.sma_slow.iloc[index] and \
        #    self.sma_fast.iloc[index-1] <= self.sma_slow.iloc[index-1]:
        #     return 'BUY'
        # elif self.sma_fast.iloc[index] < self.sma_slow.iloc[index] and \
        #      self.sma_fast.iloc[index-1] >= self.sma_slow.iloc[index-1]:
        #     return 'SELL'
        # return ''
        pass
```

## 重要要求
1. 策略类必须命名为`GeneratedStrategy`，并且继承自`Strategy`基类
2. 必须实现`initialize`和`on_bar`方法
3. 策略逻辑必须合理且可解释
4. 避免过拟合，使用常见的技术指标和合理的参数
5. 确保代码可以直接运行，没有语法错误
6. 提供详细的注释和策略说明
7. 考虑风险控制机制

请输出完整的策略代码和详细的策略说明。
""".format(
            data_desc=escaped_data_desc,
            data_sum=escaped_data_summary,
            # sample=escaped_sample_data,
            analysis=escaped_analysis_result
        )
        
        debug(f"[DEBUG] 生成的提示词长度: {len(prompt)} 字符")
        debug(f"[DEBUG] 提示词前100个字符: {prompt}")
        # 系统提示词
        system_prompt = """
你是一位专业的量化交易策略专家，精通Python编程和技术分析。
请根据提供的数据和分析，设计一个合理、可实现且有潜力的交易策略。
生成的代码必须严格遵循要求的格式，可以直接运行，并且策略逻辑必须清晰且有理论依据。
"""
        info("=================初始化策略请求，请观察================")
        # 使用多轮对话方式调用API
        response = self.client.send_message_round(
            prompt=prompt,
            use_reasoning=self.use_reasoning,
            stream=True
        )
        info("=================初始化策略请求结束，请观察结果================")
        
        # 解析策略代码和说明
        debug(f"[DEBUG] 大模型初始化策略策略响应: {response}")
        strategy_code, strategy_description = self._parse_strategy_response(response)
        
        return strategy_code, strategy_description
    
    def optimize_strategy(self, strategy_code: str, strategy_description: str, 
                         backtest_results: str, data_description: str = "") -> Tuple[str, str]:
        """
        基于回测结果优化策略
        
        Args:
            strategy_code: 当前策略代码
            strategy_description: 当前策略说明
            backtest_results: 回测结果摘要
            data_description: 数据描述
            
        Returns:
            Tuple[str, str]: (优化后的策略代码, 优化说明)
        """
        # 构建提示词，使用format方法避免f-string空表达式问题
        data_desc_text = data_description if data_description else "暂无数据描述"
        # 转义所有参数中的花括号，避免被format方法错误解释
        escaped_data_desc = data_desc_text.replace('{', '{{').replace('}', '}}')
        debug(f"优化策略 - 转义后的数据描述: {escaped_data_desc}")
        
        escaped_strategy_desc = strategy_description.replace('{', '{{').replace('}', '}}')
        debug(f"优化策略 - 转义后的策略说明: {escaped_strategy_desc[:2000]}...")
        
        escaped_strategy_code = strategy_code.replace('{', '{{').replace('}', '}}')
        debug(f"优化策略 - 转义后的策略代码长度: {len(escaped_strategy_code)} 字符")
        
        escaped_backtest_results = backtest_results.replace('{', '{{').replace('}', '}}')
        debug(f"优化策略 - 转义后的回测结果: {escaped_backtest_results[:2000]}...")
        
        # 构建提示词，使用命名参数避免位置参数错误
        prompt = """
# 策略优化任务

## 数据描述
{data_desc}

## 当前策略说明
{strategy_desc}

## 当前策略代码
```python
{strategy_code}
```

## 回测结果
{backtest_results}

## 任务要求
请分析回测结果，找出当前策略的优缺点，并提出改进方案。然后根据这些改进，生成优化后的策略代码。

请按照以下格式返回：

### 优化分析
[在这里详细分析当前策略的优缺点，解释哪些方面需要改进，以及为什么]

### 优化后的策略代码
```python
# 优化后的策略代码，必须保持类名为GeneratedStrategy
# [优化后的完整代码]
```

## 重要要求
1. 优化必须基于回测结果的具体数据，而不是泛泛而谈
2. 保持策略的核心逻辑，只对关键部分进行优化
3. 代码必须严格遵循之前的格式要求
4. 提供详细的优化理由和预期效果
5. 确保优化后的代码可以直接运行

请输出优化分析和完整的优化后策略代码。
""".format(
            data_desc=escaped_data_desc,
            strategy_desc=escaped_strategy_desc,
            strategy_code=escaped_strategy_code,
            backtest_results=escaped_backtest_results
        )
        
        debug(f"[DEBUG] 优化策略 - 生成的提示词长度: {len(prompt)} 字符")
        debug(f"[DEBUG] 优化策略 - 提示词前100个字符: {prompt[:2000]}...")
        
        # 系统提示词
        system_prompt = """
你是一位专业的量化交易策略优化专家，擅长分析回测结果并针对性地改进策略。
请根据提供的回测数据和当前策略，找出具体的改进点，并生成优化后的策略代码。
优化应该有理有据，基于数据和交易理论，避免随意更改参数。
"""
        
        # 使用多轮对话方式调用API
        response = self.client.send_message_round(
            prompt=prompt,
            use_reasoning=self.use_reasoning,
            stream=True
        )
        
        # 解析优化后的策略代码和说明
        optimized_code, optimization_analysis = self._parse_optimization_response(response)
        
        return optimized_code, optimization_analysis
    
    def _get_data_summary(self, data: pd.DataFrame) -> str:
        """
        获取数据的统计摘要
        
        Args:
            data: K线数据
            
        Returns:
            str: 格式化的统计摘要
        """
        # 计算基本统计指标
        stats = data.describe()
        
        # 计算额外的统计指标
        n_days = len(data)
        start_date = data.index[0] if hasattr(data.index, 'name') and data.index.name == 'time' else 'Unknown'
        end_date = data.index[-1] if hasattr(data.index, 'name') and data.index.name == 'time' else 'Unknown'
        
        # 计算收益率和波动率
        if 'close' in data.columns:
            returns = data['close'].pct_change().dropna()
            annualized_vol = returns.std() * np.sqrt(365)
            total_return = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]
        else:
            annualized_vol = 0
            total_return = 0
        
        # 构建摘要
        summary = f"""
- 数据总条数: {n_days}
- 时间范围: {start_date} 到 {end_date}
- 年化波动率: {annualized_vol:.2%}
- 总收益率: {total_return:.2%}
        """
        
        return summary
    
    def _parse_strategy_response(self, response: str) -> Tuple[str, str]:
        """
        解析策略生成响应，提取策略代码和说明
        
        Args:
            response: API响应文本
            
        Returns:
            Tuple[str, str]: (策略代码, 策略说明)
        """
        # 提取策略说明
        desc_pattern = r'### 策略说明\s+(.*?)\s+### 策略代码'  # 匹配策略说明部分
        desc_match = re.search(desc_pattern, response, re.DOTALL)
        strategy_description = desc_match.group(1).strip() if desc_match else "无策略说明"
        
        # 提取策略代码
        code_pattern = r'```python\s+(.*?)\s+```'  # 匹配Python代码块
        code_match = re.search(code_pattern, response, re.DOTALL)
        
        if code_match:
            strategy_code = code_match.group(1).strip()
        else:
            # 如果没有找到代码块，尝试提取整个类定义
            class_pattern = r'class\s+GeneratedStrategy\s*\([^)]*\)\s*:\s+.*?'  # 匹配类定义开始
            class_match = re.search(class_pattern, response, re.DOTALL)
            if class_match:
                # 尝试提取完整的类定义
                start_idx = response.find(class_match.group(0))
                # 简单方法：从类定义开始，到下一个类定义或文件结束
                end_idx = response.find('class ', start_idx + 1)
                if end_idx == -1:
                    end_idx = len(response)
                strategy_code = response[start_idx:end_idx].strip()
            else:
                strategy_code = ""  # 如果还是找不到，返回空
        
        # 确保代码包含必要的导入和类结构
        if strategy_code:
            # 确保导入语句存在
            if 'from backtester import Strategy' not in strategy_code:
                strategy_code = "from backtester import Strategy\n" + strategy_code
            if 'import pandas as pd' not in strategy_code:
                strategy_code = "import pandas as pd\n" + strategy_code
            if 'import numpy as np' not in strategy_code:
                strategy_code = "import numpy as np\n" + strategy_code
            # 确保使用on_bar方法，而不是on_tick
            if 'def on_tick' in strategy_code and 'def on_bar' not in strategy_code:
                strategy_code = strategy_code.replace('def on_tick', 'def on_bar')
        
        return strategy_code, strategy_description
    
    def _parse_optimization_response(self, response: str) -> Tuple[str, str]:
        """
        解析策略优化响应，提取优化后的代码和分析
        
        Args:
            response: API响应文本
            
        Returns:
            Tuple[str, str]: (优化后的策略代码, 优化分析)
        """
        # 提取优化分析
        analysis_pattern = r'### 优化分析\s+(.*?)\s+### 优化后的策略代码'
        analysis_match = re.search(analysis_pattern, response, re.DOTALL)
        optimization_analysis = analysis_match.group(1).strip() if analysis_match else "无优化分析"
        
        # 提取优化后的代码
        code_pattern = r'```python\s+(.*?)\s+```'
        code_match = re.search(code_pattern, response, re.DOTALL)
        
        if code_match:
            optimized_code = code_match.group(1).strip()
        else:
            optimized_code = ""  # 如果找不到，返回空
        
        # 确保代码包含必要的导入和类结构
        if optimized_code:
            # 确保导入语句存在
            if 'from backtester import Strategy' not in optimized_code:
                optimized_code = "from backtester import Strategy\n" + optimized_code
            if 'import pandas as pd' not in optimized_code:
                optimized_code = "import pandas as pd\n" + optimized_code
            if 'import numpy as np' not in optimized_code:
                optimized_code = "import numpy as np\n" + optimized_code
            # 确保使用on_bar方法，而不是on_tick
            if 'def on_tick' in optimized_code and 'def on_bar' not in optimized_code:
                optimized_code = optimized_code.replace('def on_tick', 'def on_bar')
        
        return optimized_code, optimization_analysis
    
    def validate_strategy_code(self, code: str) -> bool:
        """
        验证策略代码是否有效
        
        Args:
            code: 策略代码
            
        Returns:
            bool: 代码是否有效
        """
        try:
            # 检查是否包含必要的类和方法
            if 'class GeneratedStrategy' not in code:
                error("策略代码必须包含名为GeneratedStrategy的类")
                return False
            
            if 'def initialize' not in code:
                error("策略代码必须包含initialize方法")
                return False
            
            if 'def on_bar' not in code:
                error("策略代码必须包含on_bar方法")
                return False
            
            # 尝试编译代码，检查语法错误
            compile(code, '<string>', 'exec')
            
            return True
        except Exception as e:
            error(f"策略代码验证失败: {e}")
            return False
    
    def save_strategy(self, code: str, description: str, filename: str = "generated_strategy.py"):
        """
        保存策略代码到文件
        
        Args:
            code: 策略代码
            description: 策略说明
            filename: 保存的文件名
        """
        # 先处理description中的换行符，再插入到字符串中
        strategy_name = filename.replace('.py', '')
        processed_description = description.replace('\n', ' ')
        
        # 在代码开头添加策略说明作为注释
        header = """
# 策略名称: {}
# 策略说明: {}

""".format(strategy_name, processed_description)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(header + code)
        
        info(f"策略已保存到 {filename}")

# 示例用法
if __name__ == "__main__":
    info("AI策略生成器已创建，使用方法:")
    info("""
    from data_reader import DataReader
    from ai_strategy_generator import AIStrategyGenerator
    
    # 读取数据
    reader = DataReader()
    df = reader.read_csv_file("BINANCE_BTCUSDT, 1D_filtered.csv")
    
    # 创建AI策略生成器
    generator = AIStrategyGenerator(use_reasoning=True)
    
    # 分析数据
    analysis = generator.analyze_data(df, "比特币日线数据")
    
    # 生成策略
    strategy_code, strategy_description = generator.generate_strategy(
        df, 
        "比特币日线数据", 
        analysis
    )
    
    # 验证策略代码
    if generator.validate_strategy_code(strategy_code):
        generator.save_strategy(strategy_code, strategy_description)
    """)