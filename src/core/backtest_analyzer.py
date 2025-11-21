import pandas as pd
import numpy as np
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import json
from datetime import datetime
import os

class BacktestAnalyzer:
    """
    回测结果分析器，用于深度分析交易策略的回测结果
    提供更全面的性能指标、风险评估和可视化功能
    """
    
    def __init__(self, backtest_results: Dict):
        """
        初始化分析器
        
        Args:
            backtest_results: 回测结果字典
        """
        self.results = backtest_results
        self.equity_curve = backtest_results.get('equity_curve', pd.Series())
        self.returns = backtest_results.get('returns', pd.Series())
        self.trades = backtest_results.get('trades', [])
        self.positions = backtest_results.get('positions', pd.Series())
        self._validate_results()
    
    def _validate_results(self):
        """
        验证回测结果数据的有效性
        """
        if self.equity_curve.empty:
            raise ValueError("回测结果中没有资金曲线数据")
        if self.returns.empty:
            # 如果没有收益率数据，尝试从资金曲线计算
            if not self.equity_curve.empty:
                self.returns = self.equity_curve.pct_change().fillna(0)
        
    def calculate_comprehensive_metrics(self) -> Dict[str, Any]:
        """
        计算全面的性能指标
        
        Returns:
            Dict: 详细的性能指标
        """
        metrics = {}
        
        # 基本收益指标
        metrics['total_return'] = (self.equity_curve.iloc[-1] - self.equity_curve.iloc[0]) / self.equity_curve.iloc[0]
        metrics['initial_capital'] = self.equity_curve.iloc[0]
        metrics['final_capital'] = self.equity_curve.iloc[-1]
        metrics['total_profit'] = self.equity_curve.iloc[-1] - self.equity_curve.iloc[0]
        
        # 年化收益和风险调整指标
        trading_days = len(self.equity_curve)
        metrics['annual_return'] = (1 + metrics['total_return']) ** (365 / trading_days) - 1
        metrics['daily_volatility'] = self.returns.std()
        metrics['annual_volatility'] = metrics['daily_volatility'] * np.sqrt(365)
        
        # 风险调整回报指标
        if metrics['annual_volatility'] > 0:
            metrics['sharpe_ratio'] = metrics['annual_return'] / metrics['annual_volatility']
        else:
            metrics['sharpe_ratio'] = 0
        
        # Sortino比率（只考虑下行风险）
        downside_returns = self.returns[self.returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(365)
            metrics['sortino_ratio'] = metrics['annual_return'] / downside_deviation if downside_deviation > 0 else 0
        else:
            metrics['sortino_ratio'] = float('inf')
        
        # 最大回撤相关指标
        drawdown, max_drawdown, max_drawdown_start, max_drawdown_end = self._calculate_drawdowns()
        metrics['max_drawdown'] = max_drawdown
        metrics['max_drawdown_start'] = max_drawdown_start
        metrics['max_drawdown_end'] = max_drawdown_end
        metrics['avg_drawdown'] = np.mean(drawdown[drawdown < 0])
        metrics['drawdown_duration'] = self._calculate_drawdown_duration(max_drawdown_start, max_drawdown_end)
        
        # Calmar比率
        metrics['calmar_ratio'] = abs(metrics['annual_return'] / metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
        
        # Omega比率（需要设置目标收益率，这里设为0）
        threshold_return = 0
        excess_returns = self.returns - threshold_return
        if len(excess_returns[excess_returns < 0]) > 0:
            downside_sum = abs(excess_returns[excess_returns < 0].sum())
            metrics['omega_ratio'] = excess_returns[excess_returns > 0].sum() / downside_sum if downside_sum > 0 else 0
        else:
            metrics['omega_ratio'] = float('inf')
        
        # 偏度和峰度（评估收益分布特性）
        metrics['skewness'] = self.returns.skew()
        metrics['kurtosis'] = self.returns.kurtosis()
        
        # 胜率和盈亏比
        win_rate, profit_factor, avg_win, avg_loss = self._calculate_trade_metrics()
        metrics['win_rate'] = win_rate
        metrics['profit_factor'] = profit_factor
        metrics['avg_win'] = avg_win
        metrics['avg_loss'] = avg_loss
        metrics['expectancy'] = self._calculate_expectancy(avg_win, avg_loss, win_rate)
        
        # 交易频率指标
        metrics['total_trades'] = len(self.trades) // 2  # 每笔交易包含买入和卖出
        if metrics['total_trades'] > 0:
            metrics['avg_holding_period'] = self._calculate_avg_holding_period()
            metrics['trades_per_month'] = metrics['total_trades'] / (trading_days / 30)
        else:
            metrics['avg_holding_period'] = 0
            metrics['trades_per_month'] = 0
        
        # 贝塔系数（假设基准为0，即无风险收益率）
        metrics['beta'] = 0  # 简化计算
        metrics['alpha'] = metrics['annual_return']
        
        return metrics
    
    def _calculate_drawdowns(self) -> Tuple[np.ndarray, float, Optional[Any], Optional[Any]]:
        """
        计算回撤序列和最大回撤
        
        Returns:
            Tuple: (回撤序列, 最大回撤值, 最大回撤开始时间, 最大回撤结束时间)
        """
        # 计算累积最大值
        cum_max = self.equity_curve.cummax()
        # 计算回撤
        drawdown = (self.equity_curve - cum_max) / cum_max
        
        # 找到最大回撤
        max_drawdown = drawdown.min()
        
        # 找到最大回撤的开始和结束时间
        max_drawdown_idx = drawdown.idxmin()
        
        # 找到最大回撤前的累积最大值对应的索引
        if isinstance(self.equity_curve.index, pd.DatetimeIndex):
            before_max_idx = self.equity_curve.index[self.equity_curve.index <= max_drawdown_idx]
            if len(before_max_idx) > 0:
                cum_max_before = self.equity_curve.loc[before_max_idx].cummax()
                max_drawdown_start = cum_max_before.idxmax()
                max_drawdown_end = max_drawdown_idx
            else:
                max_drawdown_start = None
                max_drawdown_end = None
        else:
            # 如果不是时间索引，返回位置
            max_drawdown_start = np.argmax(self.equity_curve.values[:max_drawdown_idx+1])
            max_drawdown_end = max_drawdown_idx
        
        return drawdown.values, max_drawdown, max_drawdown_start, max_drawdown_end
    
    def _calculate_drawdown_duration(self, start: Any, end: Any) -> float:
        """
        计算最大回撤持续时间
        
        Args:
            start: 最大回撤开始位置/时间
            end: 最大回撤结束位置/时间
            
        Returns:
            float: 持续时间（天数）
        """
        if start is None or end is None:
            return 0
        
        if isinstance(start, pd.Timestamp) and isinstance(end, pd.Timestamp):
            return (end - start).days
        elif isinstance(start, int) and isinstance(end, int):
            # 假设是日线数据
            return end - start
        else:
            return 0
    
    def _calculate_trade_metrics(self) -> Tuple[float, float, float, float]:
        """
        计算交易相关指标
        
        Returns:
            Tuple: (胜率, 盈亏比, 平均盈利, 平均亏损)
        """
        profits = [trade.get('profit', 0) for trade in self.trades if 'profit' in trade]
        
        if not profits:
            return 0, 0, 0, 0
        
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        total_count = win_count + loss_count
        
        if total_count == 0:
            return 0, 0, 0, 0
        
        win_rate = win_count / total_count
        avg_win = sum(winning_trades) / win_count if win_count > 0 else 0
        avg_loss = abs(sum(losing_trades) / loss_count) if loss_count > 0 else 1
        
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        return win_rate, profit_factor, avg_win, avg_loss
    
    def _calculate_expectancy(self, avg_win: float, avg_loss: float, win_rate: float) -> float:
        """
        计算交易预期值
        
        Args:
            avg_win: 平均盈利
            avg_loss: 平均亏损
            win_rate: 胜率
            
        Returns:
            float: 预期值
        """
        return (avg_win * win_rate) - (avg_loss * (1 - win_rate))
    
    def _calculate_avg_holding_period(self) -> float:
        """
        计算平均持仓周期
        
        Returns:
            float: 平均持仓天数
        """
        if len(self.trades) < 2:
            return 0
        
        holding_periods = []
        for i in range(0, len(self.trades), 2):
            if i+1 < len(self.trades) and self.trades[i]['type'] == 'BUY' and self.trades[i+1]['type'] == 'SELL':
                buy_time = self.trades[i]['timestamp']
                sell_time = self.trades[i+1]['timestamp']
                
                if isinstance(buy_time, pd.Timestamp) and isinstance(sell_time, pd.Timestamp):
                    duration = (sell_time - buy_time).days
                    holding_periods.append(duration)
                elif hasattr(buy_time, 'strftime') and hasattr(sell_time, 'strftime'):
                    # 尝试转换为datetime
                    try:
                        buy_dt = pd.to_datetime(buy_time)
                        sell_dt = pd.to_datetime(sell_time)
                        duration = (sell_dt - buy_dt).days
                        holding_periods.append(duration)
                    except:
                        pass
        
        return sum(holding_periods) / len(holding_periods) if holding_periods else 0
    
    def analyze_trade_quality(self) -> Dict[str, Any]:
        """
        分析交易质量
        
        Returns:
            Dict: 交易质量分析
        """
        analysis = {}
        
        # 交易分布分析
        if self.trades:
            # 按时间段统计交易
            analysis['trade_distribution'] = self._analyze_trade_distribution()
            
            # 最大连续盈利和亏损
            analysis['streaks'] = self._analyze_trading_streaks()
            
            # 入场和出场分析
            analysis['entry_exit_analysis'] = self._analyze_entry_exit_points()
        
        # 波动率分析
        analysis['volatility_profile'] = self._analyze_volatility()
        
        # 相关性分析
        analysis['correlation_analysis'] = self._analyze_correlations()
        
        return analysis
    
    def _analyze_trade_distribution(self) -> Dict:
        """
        分析交易时间分布
        """
        distribution = {}
        # 简单实现：统计每个月的交易次数
        # 这里可以根据需要扩展更详细的分析
        return distribution
    
    def _analyze_trading_streaks(self) -> Dict[str, int]:
        """
        分析交易连续盈利和亏损的情况
        
        Returns:
            Dict: 连续盈利和亏损的统计
        """
        profits = [trade.get('profit', 0) for trade in self.trades if 'profit' in trade]
        
        if not profits:
            return {'max_winning_streak': 0, 'max_losing_streak': 0}
        
        max_winning_streak = 0
        max_losing_streak = 0
        current_winning_streak = 0
        current_losing_streak = 0
        
        for profit in profits:
            if profit > 0:
                current_winning_streak += 1
                current_losing_streak = 0
                max_winning_streak = max(max_winning_streak, current_winning_streak)
            else:
                current_losing_streak += 1
                current_winning_streak = 0
                max_losing_streak = max(max_losing_streak, current_losing_streak)
        
        return {
            'max_winning_streak': max_winning_streak,
            'max_losing_streak': max_losing_streak
        }
    
    def _analyze_entry_exit_points(self) -> Dict:
        """
        分析入场和出场点的质量
        """
        # 简单实现：可以分析入场点是否接近局部低点，出场点是否接近局部高点
        return {}
    
    def _analyze_volatility(self) -> Dict:
        """
        分析策略运行期间的波动率特征
        """
        if self.returns.empty:
            return {}
        
        # 计算滚动波动率
        rolling_vol = self.returns.rolling(window=20).std() * np.sqrt(252)  # 20日滚动年化波动率
        
        return {
            'avg_rolling_volatility': rolling_vol.mean(),
            'volatility_trend': 'increasing' if rolling_vol.iloc[-1] > rolling_vol.iloc[0] else 'decreasing'
        }
    
    def _analyze_correlations(self) -> Dict:
        """
        分析策略收益与市场因素的相关性
        """
        # 简化实现，实际应用中可以与基准指数等进行相关性分析
        return {}
    
    def generate_detailed_report(self) -> str:
        """
        生成详细的回测分析报告
        
        Returns:
            str: 格式化的分析报告
        """
        # 获取综合指标
        metrics = self.calculate_comprehensive_metrics()
        
        # 获取交易质量分析
        trade_quality = self.analyze_trade_quality()
        
        # 构建报告
        report = """
# 策略回测详细分析报告

## 1. 核心性能指标

### 收益指标
- 总收益率: {total_return:.2%}
- 年化收益率: {annual_return:.2%}
- 初始资金: {initial_capital:.2f}
- 最终资金: {final_capital:.2f}
- 总盈亏: {total_profit:.2f}

### 风险指标
- 最大回撤: {max_drawdown:.2%}
- 最大回撤持续: {drawdown_duration:.0f} 天
- 平均回撤: {avg_drawdown:.2%}
- 年化波动率: {annual_volatility:.2%}

### 风险调整回报指标
- 夏普比率: {sharpe_ratio:.2f}
- Sortino比率: {sortino_ratio:.2f}
- Calmar比率: {calmar_ratio:.2f}
- Omega比率: {omega_ratio:.2f}

### 交易统计
- 总交易次数: {total_trades}
- 胜率: {win_rate:.2%}
- 盈亏比: {profit_factor:.2f}
- 平均盈利: {avg_win:.2f}
- 平均亏损: {avg_loss:.2f}
- 交易预期值: {expectancy:.2f}
- 平均持仓周期: {avg_holding_period:.1f} 天
- 月均交易次数: {trades_per_month:.1f}

## 2. 交易质量分析

### 交易连续性分析
- 最长连续盈利: {max_winning_streak} 次
- 最长连续亏损: {max_losing_streak} 次

### 波动率特征
- 平均滚动波动率: {avg_rolling_volatility:.2%}
- 波动率趋势: {volatility_trend}

## 3. 策略评估与建议

### 优势
- {strengths}

### 劣势
- {weaknesses}

### 优化建议
- {suggestions}

## 4. 结论

{conclusion}
        """.format(
            total_return=metrics.get('total_return', 0),
            annual_return=metrics.get('annual_return', 0),
            initial_capital=metrics.get('initial_capital', 0),
            final_capital=metrics.get('final_capital', 0),
            total_profit=metrics.get('total_profit', 0),
            max_drawdown=metrics.get('max_drawdown', 0),
            drawdown_duration=metrics.get('drawdown_duration', 0),
            avg_drawdown=metrics.get('avg_drawdown', 0),
            annual_volatility=metrics.get('annual_volatility', 0),
            sharpe_ratio=metrics.get('sharpe_ratio', 0),
            sortino_ratio=metrics.get('sortino_ratio', 0),
            calmar_ratio=metrics.get('calmar_ratio', 0),
            omega_ratio=metrics.get('omega_ratio', 0),
            total_trades=metrics.get('total_trades', 0),
            win_rate=metrics.get('win_rate', 0),
            profit_factor=metrics.get('profit_factor', 0),
            avg_win=metrics.get('avg_win', 0),
            avg_loss=metrics.get('avg_loss', 0),
            expectancy=metrics.get('expectancy', 0),
            avg_holding_period=metrics.get('avg_holding_period', 0),
            trades_per_month=metrics.get('trades_per_month', 0),
            max_winning_streak=trade_quality.get('streaks', {}).get('max_winning_streak', 0),
            max_losing_streak=trade_quality.get('streaks', {}).get('max_losing_streak', 0),
            avg_rolling_volatility=trade_quality.get('volatility_profile', {}).get('avg_rolling_volatility', 0),
            volatility_trend=trade_quality.get('volatility_profile', {}).get('volatility_trend', 'stable'),
            strengths=self._generate_strengths(metrics),
            weaknesses=self._generate_weaknesses(metrics),
            suggestions=self._generate_suggestions(metrics),
            conclusion=self._generate_conclusion(metrics)
        )
        
        return report
    
    def _generate_strengths(self, metrics: Dict) -> str:
        """
        根据指标生成策略优势
        """
        strengths = []
        
        if metrics.get('sharpe_ratio', 0) > 1.0:
            strengths.append("夏普比率良好，表明风险调整后回报优秀")
        if metrics.get('win_rate', 0) > 0.5:
            strengths.append(f"胜率超过50%，达到 {metrics['win_rate']:.1%}")
        if metrics.get('profit_factor', 0) > 1.5:
            strengths.append(f"盈亏比较高，达到 {metrics['profit_factor']:.1f}")
        if abs(metrics.get('max_drawdown', 0)) < 0.2:
            strengths.append(f"最大回撤控制在合理范围内 ({metrics['max_drawdown']:.1%})")
        
        if not strengths:
            strengths.append("策略展现出一定的交易能力")
        
        return "\n- ".join(strengths)
    
    def _generate_weaknesses(self, metrics: Dict) -> str:
        """
        根据指标生成策略劣势
        """
        weaknesses = []
        
        if metrics.get('sharpe_ratio', 0) < 0.5:
            weaknesses.append("夏普比率偏低，风险调整后回报不足")
        if metrics.get('win_rate', 0) < 0.4:
            weaknesses.append(f"胜率较低，仅为 {metrics['win_rate']:.1%}")
        if metrics.get('profit_factor', 0) < 1.2:
            weaknesses.append(f"盈亏比较低，仅为 {metrics['profit_factor']:.1f}")
        if abs(metrics.get('max_drawdown', 0)) > 0.3:
            weaknesses.append(f"最大回撤较大，达到 {metrics['max_drawdown']:.1%}")
        if metrics.get('total_trades', 0) < 10:
            weaknesses.append("交易次数较少，策略稳定性有待验证")
        
        if not weaknesses:
            weaknesses.append("策略整体表现稳定，需在实盘中进一步检验")
        
        return "\n- ".join(weaknesses)
    
    def _generate_suggestions(self, metrics: Dict) -> str:
        """
        根据指标生成优化建议
        """
        suggestions = []
        
        # 基于夏普比率的建议
        if metrics.get('sharpe_ratio', 0) < 0.5:
            suggestions.append("考虑调整止损策略，控制单笔亏损幅度")
            suggestions.append("优化入场时机，提高胜率")
        
        # 基于最大回撤的建议
        if abs(metrics.get('max_drawdown', 0)) > 0.3:
            suggestions.append("增加仓位管理机制，避免满仓交易")
            suggestions.append("考虑引入对冲策略降低系统性风险")
        
        # 基于交易频率的建议
        if metrics.get('trades_per_month', 0) > 20:
            suggestions.append("交易频率过高，可能导致过度交易和滑点损失增加")
        elif metrics.get('trades_per_month', 0) < 2:
            suggestions.append("交易频率过低，可能错过较多交易机会")
        
        # 基于胜率和盈亏比的建议
        if metrics.get('win_rate', 0) < 0.4 and metrics.get('profit_factor', 0) < 1.2:
            suggestions.append("重新评估策略逻辑，考虑增加过滤条件提高信号质量")
        
        if not suggestions:
            suggestions.append("微调策略参数，优化入场和出场时机")
            suggestions.append("考虑在不同市场环境下调整策略参数")
        
        return "\n- ".join(suggestions)
    
    def _generate_conclusion(self, metrics: Dict) -> str:
        """
        生成策略结论
        """
        total_return = metrics.get('total_return', 0)
        max_drawdown = metrics.get('max_drawdown', 0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        
        if total_return > 0.5 and sharpe_ratio > 1.0 and abs(max_drawdown) < 0.2:
            return "该策略表现优秀，具有良好的收益风险比和稳定性，建议进一步在不同市场环境下进行验证，并考虑实盘测试。"
        elif total_return > 0 and sharpe_ratio > 0.5:
            return "该策略表现良好，能够产生正收益且风险控制合理，但仍有优化空间，建议继续改进以提高整体性能。"
        elif total_return > 0:
            return "该策略能够产生正收益，但风险调整后回报一般，建议重点关注风险控制方面的优化。"
        else:
            return "该策略未能产生正收益，需要重新评估策略逻辑，或者考虑完全重新设计策略框架。"
    
    def to_json(self, filename: Optional[str] = None) -> str:
        """
        将分析结果保存为JSON格式
        
        Args:
            filename: 保存的文件名
            
        Returns:
            str: JSON字符串
        """
        metrics = self.calculate_comprehensive_metrics()
        trade_quality = self.analyze_trade_quality()
        
        # 转换numpy类型为Python原生类型以便JSON序列化
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return obj.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        data = {
            'metrics': convert_numpy(metrics),
            'trade_quality': convert_numpy(trade_quality),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(json_str)
            print(f"分析结果已保存到 {filename}")
        
        return json_str

# 示例用法
if __name__ == "__main__":
    print("回测结果分析器已创建，使用方法:")
    print("""
    from data_reader import DataReader
    from backtester import Backtest, SimpleMovingAverageStrategy
    from backtest_analyzer import BacktestAnalyzer
    
    # 读取数据
    reader = DataReader()
    df = reader.read_csv_file("BINANCE_BTCUSDT, 1D_filtered.csv")
    
    # 创建策略并运行回测
    strategy = SimpleMovingAverageStrategy({'fast_period': 20, 'slow_period': 50})
    backtest = Backtest(df, strategy, initial_capital=10000.0)
    results = backtest.run()
    
    # 创建分析器
    analyzer = BacktestAnalyzer(results)
    
    # 生成详细报告
    report = analyzer.generate_detailed_report()
    print(report)
    
    # 保存分析结果
    analyzer.to_json("backtest_analysis.json")
    """)