# Daily Momentum Backtest (demo)

## 目标
实现一个最小可复现的日频动量回测示例 (TICKER = SPY, WINDOW = 20)

## 运行
1. 建议创建虚拟环境
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt <br>
    或 创建一个新的conda环境
conda create -n momentum python=3.12
conda activate momentum
pip install -r requirements.txt

2. 运行
若考虑交易成本时不影响动量策略结果，则运行 `momentum_backtest_1.py`；若考虑交易成本时影响动量策略结果，则运行 `momentum_backtest_2.py` (两者均位于 `src\`) <br>
    或 打开演示文本 `momentum-demo.ipynb` (位于`notebooks/`)

3. 结果位于 `results/`，分为 `daily_results.csv`（每日记录）、`summary_metrics.csv`（总览指标）、`performance.png`（累计净值对比图）

## 主要假设
- 无杠杆、无借贷；仓位为 0 或 100%（等权）；  
- 年化因子按 252 个交易日计算；  
- 初始版本没有交易成本（可选在脚本中打开交易成本模型）

## 指标 (将由脚本输出)
- 样本行数 (数据量)
- 估计交易次数、持仓天数
- 不考虑交易成本下的累计收益、年化收益、年化波动、Sharpe（假设无风险利率为 0）、最大回撤、胜率
- 考虑交易成本下的累计收益、年化收益、年化波动、Sharpe（假设无风险利率为 0）、最大回撤、胜率
- 考虑成本下且交易成本影响动量策略下的累计收益、年化收益、年化波动、Sharpe（假设无风险利率为 0）、最大回撤、胜率

## 许可证
MIT