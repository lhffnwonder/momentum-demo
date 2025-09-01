# momentum_backtest.py
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
# —— 可靠的 results 定位 + 保存代码（直接粘贴运行） ——
from pathlib import Path
import os

# 1) 自动寻找已有的 results 目录（从 cwd 向上查找）
def find_results_dir(start: Path = Path.cwd()) -> Path:
    if (start / "results").is_dir():
        return start / "results"
    for p in start.parents:
        if (p / "results").is_dir():
            return p / "results"
    # 找不到就返回 cwd/results（后面会创建）
    return start / "results"

if __name__ == "__main__":
    results_dir = find_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)  # 确保存在

    print("当前工作目录 (cwd):", Path.cwd())
    print("选定的 results 目录:", results_dir)
    print("results 目录存在且可写?:", os.access(results_dir, os.W_OK))

    # ------------- 参数 -------------
    TICKER = "SPY"            # 标的，示例用 SPY（美股标准 ETF）
    START = "2022-01-01"      # 起始日（改成近 3 年）
    END = datetime.today().strftime("%Y-%m-%d")
    WINDOW = 20               # 动量窗口（20 日）
    TRADING_DAYS = 252        # 年化换算系数
    # ------------- 拉数据 -------------
    df = yf.download(TICKER, start=START, end=END, progress=False)
    # 简单检查
    print("Rows of data:", len(df)) # type: ignore
    df = df[['Close']].dropna() # type: ignore

    # ------------- 计算信号（简单动量）-------------
    # 计算过去 WINDOW 天的收益率（收盘价比）
    df['mom'] = df['Close'].pct_change(WINDOW)
    # 信号：过去 WINDOW 天收益率 > 0 则持有（1），否则空仓（0）
    df['signal'] = (df['mom'] > 0).astype(int)
    # 我们避免未来函数（用今天的 signal 去决定明天的持仓）
    df['signal_shift'] = df['signal'].shift(1).fillna(0)

    # ------------- 计算每日回报 -------------
    df['ret_daily'] = df['Close'].pct_change().fillna(0)
    # 策略回报（简单：无杠杆、全仓/空仓）
    df['strat_ret'] = df['signal_shift'] * df['ret_daily']

    # ------------- 计算累计净值（wealth index）-------------
    df['wealth_market'] = (1 + df['ret_daily']).cumprod()
    df['wealth_strategy'] = (1 + df['strat_ret']).cumprod()

    # ------------- 绩效指标 -------------
    n_days = len(df)
    cumulative_return = df['wealth_strategy'].iloc[-1] - 1
    annualized_return = (1 + cumulative_return) ** (TRADING_DAYS / n_days) - 1
    # 更稳妥（几何平均）：annualized_return = np.expm1(df['strat_ret'].mean() * TRADING_DAYS)
    ann_vol = df['strat_ret'].std() * np.sqrt(TRADING_DAYS)
    sharpe = annualized_return / ann_vol if ann_vol != 0 else np.nan

    # 最大回撤
    rolling_max = df['wealth_strategy'].cummax()
    drawdown = (df['wealth_strategy'] / rolling_max) - 1
    max_drawdown = drawdown.min()

    # 胜率 & 交易次数
    trade_days = df['signal_shift'].diff().abs().sum() / 2  # type: ignore # 粗略估计换仓次数
    win_rate = (df.loc[df['strat_ret'] > 0, 'strat_ret'].count()) / (df.loc[df['signal_shift'] != 0, 'strat_ret'].count()+1e-9)

    print("样本行数:", n_days)
    print("累计收益 (strategy): {:.2%}".format(cumulative_return))
    print("年化收益: {:.2%}".format(annualized_return))
    print("年化波动: {:.2%}".format(ann_vol))
    print("Sharpe (rf=0): {:.2f}".format(sharpe))
    print("最大回撤: {:.2%}".format(max_drawdown))
    print("估计交易次数（粗略）:", int(trade_days))
    print("胜率（持仓期间的正日收益占比）: {:.2%}".format(win_rate))

    # ------------- 保存结果 -------------
    df.to_csv("results/daily_results.csv")
    pd.DataFrame({
        'metric': ['n_days', 'cumulative_return', 'annualized_return', 'annualized_vol', 'sharpe', 'max_drawdown', 'trade_days', 'win_rate'],
        'value': [n_days, cumulative_return, annualized_return, ann_vol, sharpe, max_drawdown, trade_days, win_rate]
    }).to_csv("results/summary_metrics.csv", index=False)

    # ------------- 绘图 -------------
    plt.figure(figsize=(10,6))
    plt.plot(df.index, df['wealth_market'], label='Market (Buy&Hold)')
    plt.plot(df.index, df['wealth_strategy'], label='Momentum Strategy')
    plt.legend()
    plt.title(f"{TICKER} Momentum Strategy vs Market")
    plt.ylabel("Wealth Index (start=1)")
    plt.xlabel("Date")
    plt.grid(True)
    plt.savefig("results/performance.png", dpi=150)
    plt.close()