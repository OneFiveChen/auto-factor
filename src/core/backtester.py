import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable

class Strategy:
    """
    策略基类，所有交易策略都应继承此类
    """
    
    def __init__(self, params: Optional[Dict] = None):
        """
        初始化策略
        
        Args:
            params: 策略参数
        """
        self.params = params or {}
        self.name = self.__class__.__name__
    
    def initialize(self, data: pd.DataFrame):
        """
        策略初始化
        
        Args:
            data: K线数据
        """
        pass
    
    def on_bar(self, index: int, row: pd.Series, data: pd.DataFrame) -> str:
        """
        处理每根K线
        
        Args:
            index: 当前K线索引
            row: 当前K线数据
            data: 完整数据
            
        Returns:
            str: 交易信号 'BUY', 'SELL', 或 ''（无操作）
        """
        return ''
    
    def get_name(self) -> str:
        """
        获取策略名称
        """
        return self.name

class Backtest:
    """
    回测引擎
    """
    
    def __init__(self, data: pd.DataFrame, strategy: Strategy, 
                 initial_capital: float = 10000.0, 
                 commission_rate: float = 0.001):
        """
        初始化回测引擎
        
        Args:
            data: K线数据
            strategy: 交易策略
            initial_capital: 初始资金
            commission_rate: 佣金率
        """
        self.data = data.copy()
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        
        # 初始化回测结果容器
        self.trades = []  # 交易记录
        self.equity_curve = []  # 资金曲线
        self.positions = []  # 持仓记录
        
        # 当前状态
        self.current_capital = initial_capital
        self.current_position = 0  # 持仓数量
        self.entry_price = 0.0  # 入场价格
        
    def run(self) -> Dict:
        """
        运行回测
        
        Returns:
            Dict: 回测结果
        """
        # 初始化策略
        self.strategy.initialize(self.data)
        
        # 初始化资金曲线
        self.equity_curve = [self.initial_capital]
        self.positions = [0]
        
        # 遍历每根K线
        for i in range(len(self.data)):
            current_row = self.data.iloc[i]
            
            # 获取交易信号
            signal = self.strategy.on_bar(i, current_row, self.data)
            
            # 执行交易
            self._execute_trade(i, current_row, signal)
            
            # 计算当前资产价值
            current_value = self.current_capital + self.current_position * current_row['close']
            self.equity_curve.append(current_value)
            self.positions.append(self.current_position)
        
        # 计算回测指标
        results = self._calculate_metrics()
        
        return results
    
    def _execute_trade(self, index: int, row: pd.Series, signal: str):
        """
        执行交易
        
        Args:
            index: 当前索引
            row: 当前K线数据
            signal: 交易信号
        """
        # 不区分大小写判断买入信号
        if signal.upper() == 'BUY' and self.current_position == 0:
            # 买入信号，且当前空仓
            self.entry_price = row['close']
            # 全仓买入（简化处理）
            self.current_position = self.current_capital / row['close']
            # 扣除佣金
            commission = self.current_position * row['close'] * self.commission_rate
            self.current_capital = -commission  # 剩余资金为负的佣金
            
            # 记录交易
            self.trades.append({
                'timestamp': row.name,
                'type': 'BUY',
                'price': row['close'],
                'quantity': self.current_position,
                'commission': commission,
                'balance': self.current_capital
            })
        
        elif signal.upper() == 'SELL' and self.current_position > 0:
            # 卖出信号，且当前持仓
            sell_value = self.current_position * row['close']
            # 扣除佣金
            commission = sell_value * self.commission_rate
            self.current_capital = sell_value - commission
            
            # 记录交易
            self.trades.append({
                'timestamp': row.name,
                'type': 'SELL',
                'price': row['close'],
                'quantity': self.current_position,
                'commission': commission,
                'balance': self.current_capital,
                'profit': sell_value - self.current_position * self.entry_price - commission * 2  # 双向佣金
            })
            
            # 清空持仓
            self.current_position = 0
    
    def _calculate_metrics(self) -> Dict:
        """
        计算回测指标
        
        Returns:
            Dict: 回测指标
        """
        # 计算每日收益率
        equity_series = pd.Series(self.equity_curve[1:])  # 去掉初始值
        returns = equity_series.pct_change().fillna(0)
        
        # 计算基本指标
        total_return = (equity_series.iloc[-1] - self.initial_capital) / self.initial_capital
        
        # 计算年化收益率（假设日线数据）
        trading_days = len(equity_series)
        annual_return = (1 + total_return) ** (365 / trading_days) - 1
        
        # 计算最大回撤
        cum_max = equity_series.cummax()
        drawdown = (equity_series - cum_max) / cum_max
        max_drawdown = drawdown.min()
        
        # 计算夏普比率（假设无风险利率为0）
        sharpe_ratio = np.sqrt(365) * returns.mean() / returns.std() if returns.std() != 0 else 0
        
        # 计算胜率
        winning_trades = sum(1 for trade in self.trades if trade.get('profit', 0) > 0)
        total_trades = len(self.trades) // 2  # 每笔交易包含买入和卖出
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 计算平均盈亏比
        if total_trades > 0:
            profits = [trade.get('profit', 0) for trade in self.trades if 'profit' in trade]
            avg_profit = sum(p for p in profits if p > 0) / len([p for p in profits if p > 0]) if any(p > 0 for p in profits) else 0
            avg_loss = abs(sum(p for p in profits if p < 0) / len([p for p in profits if p < 0])) if any(p < 0 for p in profits) else 1
            profit_factor = avg_profit / avg_loss if avg_loss != 0 else float('inf')
        else:
            profit_factor = 0
        
        # 构建结果字典
        metrics = {
            'strategy_name': self.strategy.get_name(),
            'initial_capital': self.initial_capital,
            'final_capital': equity_series.iloc[-1],
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'equity_curve': equity_series,
            'returns': returns,
            'trades': self.trades,
            'positions': pd.Series(self.positions[1:], index=self.data.index)  # 去掉初始值
        }
        
        return metrics
    
    def get_results_summary(self) -> str:
        """
        获取回测结果摘要（用于LLM分析）
        
        Returns:
            str: 格式化的结果摘要
        """
        metrics = self._calculate_metrics()
        
        summary = """
# 回测结果摘要
## 基本信息
- 策略名称: {strategy_name}
- 初始资金: {initial_capital:.2f}
- 最终资金: {final_capital:.2f}
- 总收益率: {total_return:.2%}
- 年化收益率: {annual_return:.2%}

## 风险指标
- 最大回撤: {max_drawdown:.2%}
- 夏普比率: {sharpe_ratio:.2f}

## 交易统计
- 总交易次数: {total_trades}
- 胜率: {win_rate:.2%}
- 盈亏比: {profit_factor:.2f}

## 交易明细
{trade_details}

## 关键交易分析
{key_trades}
        """.format(
            strategy_name=metrics['strategy_name'],
            initial_capital=metrics['initial_capital'],
            final_capital=metrics['final_capital'],
            total_return=metrics['total_return'],
            annual_return=metrics['annual_return'],
            max_drawdown=metrics['max_drawdown'],
            sharpe_ratio=metrics['sharpe_ratio'],
            total_trades=metrics['total_trades'],
            win_rate=metrics['win_rate'],
            profit_factor=metrics['profit_factor'],
            trade_details=self._format_trade_details(max_trades=100),
            key_trades=self._analyze_key_trades()
        )
        
        return summary
    
    def _format_trade_details(self, max_trades: int = 100) -> str:
        """
        格式化交易明细
        
        Args:
            max_trades: 最大显示的交易数量，默认为100笔
        """
        if not self.trades:
            return "无交易记录"
        
        details = "日期,类型,价格,数量,利润\n"
        # 显示最近的交易记录，最多max_trades笔
        for trade in self.trades[-max_trades:]:
            timestamp = trade['timestamp'].strftime('%Y-%m-%d') if hasattr(trade['timestamp'], 'strftime') else str(trade['timestamp'])
            trade_type = trade['type']
            price = trade['price']
            quantity = trade['quantity']
            
            # 对于卖出交易，显示利润
            if trade_type == 'SELL' and 'profit' in trade:
                profit = trade['profit']
                profit_str = f"{profit:.2f}"
            else:
                profit_str = "-"
            
            details += f"{timestamp},{trade_type},{price:.2f},{quantity:.4f},{profit_str}\n"
        
        return details
    
    def _analyze_key_trades(self) -> str:
        """
        分析关键交易（最大盈利和最大亏损）
        """
        profits = [trade.get('profit', 0) for trade in self.trades if 'profit' in trade]
        
        if not profits:
            return "无交易可分析"
        
        # 找到最大盈利和最大亏损的索引
        max_profit_idx = profits.index(max(profits))
        max_loss_idx = profits.index(min(profits))
        
        # 对应的交易（买入和卖出是一对）
        max_profit_trades = self.trades[max_profit_idx*2 : max_profit_idx*2 + 2]
        max_loss_trades = self.trades[max_loss_idx*2 : max_loss_idx*2 + 2]
        
        analysis = ""
        
        # 分析最大盈利交易
        if max_profit_trades:
            buy_trade = max_profit_trades[0]
            sell_trade = max_profit_trades[1]
            analysis += f"\n最大盈利交易:\n"
            analysis += f"- 买入: {buy_trade['timestamp'].strftime('%Y-%m-%d')}, 价格: {buy_trade['price']:.2f}\n"
            analysis += f"- 卖出: {sell_trade['timestamp'].strftime('%Y-%m-%d')}, 价格: {sell_trade['price']:.2f}\n"
            analysis += f"- 利润: {sell_trade.get('profit', 0):.2f}\n"
        
        # 分析最大亏损交易
        if max_loss_trades and max_loss_trades != max_profit_trades:
            buy_trade = max_loss_trades[0]
            sell_trade = max_loss_trades[1]
            analysis += f"\n最大亏损交易:\n"
            analysis += f"- 买入: {buy_trade['timestamp'].strftime('%Y-%m-%d')}, 价格: {buy_trade['price']:.2f}\n"
            analysis += f"- 卖出: {sell_trade['timestamp'].strftime('%Y-%m-%d')}, 价格: {sell_trade['price']:.2f}\n"
            analysis += f"- 亏损: {sell_trade.get('profit', 0):.2f}\n"
        
        return analysis

# 示例策略 - 用于测试
class SimpleMovingAverageStrategy(Strategy):
    """
    简单移动平均线交叉策略
    """
    
    def __init__(self, params: Optional[Dict] = None):
        super().__init__(params)
        # 默认参数
        self.fast_period = self.params.get('fast_period', 20)
        self.slow_period = self.params.get('slow_period', 50)
        self.name = f"SMA_{self.fast_period}_{self.slow_period}"
    
    def initialize(self, data: pd.DataFrame):
        # 计算移动平均线
        self.fast_ma = data['close'].rolling(window=self.fast_period).mean()
        self.slow_ma = data['close'].rolling(window=self.slow_period).mean()
    
    def on_bar(self, index: int, row: pd.Series, data: pd.DataFrame) -> str:
        # 确保有足够的数据点
        if index < self.slow_period:
            return ''
        
        # 金叉信号：快线从下往上穿过慢线
        if self.fast_ma.iloc[index-1] <= self.slow_ma.iloc[index-1] and \
           self.fast_ma.iloc[index] > self.slow_ma.iloc[index]:
            return 'BUY'
        
        # 死叉信号：快线从上往下穿过慢线
        elif self.fast_ma.iloc[index-1] >= self.slow_ma.iloc[index-1] and \
             self.fast_ma.iloc[index] < self.slow_ma.iloc[index]:
            return 'SELL'
        
        return ''

# 示例用法
if __name__ == "__main__":
    # 这里只是为了展示框架用法，实际使用时需要导入数据
    print("回测框架已创建，使用方法:")
    print("""
    from data_reader import DataReader
    from backtester import Backtest, SimpleMovingAverageStrategy
    
    # 读取数据
    reader = DataReader()
    df = reader.read_csv_file("BINANCE_BTCUSDT, 1D_filtered.csv")
    
    # 创建策略
    strategy = SimpleMovingAverageStrategy({
        'fast_period': 20,
        'slow_period': 50
    })
    
    # 运行回测
    backtest = Backtest(df, strategy, initial_capital=10000.0)
    results = backtest.run()
    
    # 打印结果摘要
    print(backtest.get_results_summary())
    """)