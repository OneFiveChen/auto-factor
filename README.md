

## 项目结构

项目采用模块化设计，主要包含以下核心组件：

```
├── ai_trading_workflow.py      # 主工作流入口文件
├── src/                        # 源代码目录
│   ├── core/                   # 核心功能模块
│   │   ├── backtester.py       # 回测引擎
│   │   ├── backtest_analyzer.py # 回测结果分析器
│   │   └── data_reader.py      # 数据读取器
│   └── utils/                  # 工具类模块
│       ├── ai_strategy_generator.py # AI策略生成器
│       └── deepseek_client.py  # DeepSeek API客户端
├── example_config.json         # 配置文件示例
└── README.md                   # 项目说明文档
```

## 核心模块说明

### 1. 主工作流 (ai_trading_workflow.py)

AITradingWorkflow类是整个系统的核心，整合了数据读取、策略生成、回测和优化的完整流程。支持通过配置文件灵活控制工作流行为。

### 2. 核心功能模块 (src/core/)

- **backtester.py**: 提供Strategy基类和回测框架，处理策略执行和资金管理
- **backtest_analyzer.py**: 计算关键性能指标（收益率、夏普比率、最大回撤等）
- **data_reader.py**: 处理CSV数据加载和格式标准化

### 3. AI工具模块 (src/utils/)

- **ai_strategy_generator.py**: 调用大模型API生成和优化交易策略代码
- **deepseek_client.py**: 管理与DeepSeek API的连接和请求

## 安装说明

1. 确保Python 3.8+已安装
2. 安装必要的依赖：

```bash
pip install pandas numpy
```

3. 配置API密钥：在`config/api_key.json`文件中添加您的DeepSeek API密钥：

```json
{
  "deepseek_api_key": "your_api_key_here"
}
```

## 使用方法

### 通过配置文件运行工作流

创建配置文件（如`example_config.json`）并执行以下命令：

```bash
python ai_trading_workflow.py --config example_config.json
```

### 配置文件示例

```json
{
    "main_data_file": "BINANCE_BTCUSDT_1D.csv",
    "main_data_description": "比特币日线K线数据",
    "additional_data_files": [],
    "additional_data_descriptions": [],
    "data_directory": "data",
    "initial_capital": 10000.0,
    "commission_rate": 0.001,
    "max_optimization_rounds": 3,
    "use_reasoning": true,
    "output_dir": "output_custom",
    "run_all_steps": true,
    "steps_to_run": ["load_data", "analyze_data", "generate_initial_strategy", "run_optimization_cycle"]
}
```


## 主要功能

1. **多数据源支持**: 可导入多个交易品种的历史数据进行分析
2. **AI策略生成**: 利用大模型自动生成交易策略代码
3. **策略优化**: 自动进行多轮策略优化和参数调优
4. **回测分析**: 提供完整的回测功能和性能指标计算
5. **配置灵活**: 通过JSON配置文件控制工作流行为

## 输出结果

运行完成后，工作流将在指定的输出目录中生成：
- `initial_strategy.py`: 初始生成的策略代码
- `final_optimized_strategy.py`: 优化后的最终策略代码
- `optimization_summary.txt`: 优化过程和结果摘要
- 各轮次的回测详细分析报告

## 注意事项

- 请确保API密钥配置正确，且有足够的API调用配额
- 数据文件需为CSV格式，包含日期和OHLCV数据
- 首次运行可能需要较长时间，取决于数据量和优化轮次
- 生成的策略仅供参考，实盘交易前请仔细验证