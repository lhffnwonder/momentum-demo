# momentum_backtest.py
# 交易成本不影响动量策略的结果
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
# —— 可靠的 results 定位 + 保存代码（直接粘贴运行） ——
from pathlib import Path
import os

# ----------------- 简单工具函数 ----------------
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # --------------- 路径与结果目录 ---------------
    # ---- 更稳妥地确定 results 目录（替换原来的 Path.cwd() 版本） ----
    # 目标：把结果保存在项目的 momentum-demo/results/ 下，而不是当前工作目录。
    script_path = Path(__file__).resolve()     # 脚本文件的绝对路径
    script_dir  = script_path.parent            # 脚本所在目录（比如 .../momentum-demo/src 或 .../momentum-demo）

    # 如果脚本位于 src/，则把项目根设为 src 的父目录；否则把项目根设为脚本所在目录
    if script_dir.name.lower() == "src":
        project_root = script_dir.parent
    else:
        project_root = script_dir

    # 最终 results 目录在 project_root/results
    results_dir = project_root / "results"
    ensure_dir(results_dir)

    # 打印信息，方便调试（可删）
    print("脚本文件位置:", script_path)
    print("脚本所在目录:", script_dir)
    print("推断的项目根目录:", project_root)
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
    print("样本行数：", len(df)) # type: ignore
    df = df[['Close']].dropna() # type: ignore
    # 确保按时间升序
    df.sort_index(inplace=True)
    print("样本时间范围:", df.index.min(), "->", df.index.max()) # type: ignore

    # ------------- 计算信号（简单动量）-------------
    # 计算过去 WINDOW 天的收益率（收盘价比）
    df['mom'] = df['Close'].pct_change(WINDOW)
    # 信号：过去 WINDOW 天收益率 > 0 则持有（1），否则空仓（0）
    df['signal'] = (df['mom'] > 0).astype(int)
    # 我们避免未来函数（用今天的 signal 去决定明天的持仓）
    df['signal_shift'] = df['signal'].shift(1).fillna(0)

    # ------------- (1) 不考虑交易成本 -------------
    print("\n")
    print("不考虑交易成本：")
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
    transitions = int((df['signal_shift'] != df['signal_shift'].shift(1)).sum())  # type: ignore # 包含从 NaN->value 的变化
    trade_days = transitions // 2  # type: ignore # 粗略估计换仓次数
    position_days = int((df['signal_shift'] != 0).sum()) # type: ignore
    win_rate = (df.loc[df['strat_ret'] > 0, 'strat_ret'].count()) / (df.loc[df['signal_shift'] != 0, 'strat_ret'].count()+1e-9)

    print("累计收益 (strategy): {:.2%}".format(cumulative_return))
    print("年化收益: {:.2%}".format(annualized_return))
    print("年化波动: {:.2%}".format(ann_vol))
    print("Sharpe (rf=0): {:.2f}".format(sharpe))
    print("最大回撤: {:.2%}".format(max_drawdown))
    print("估计交易次数（粗略）:", int(trade_days))
    print("胜率（持仓期间的正日收益占比）: {:.2%}".format(win_rate))

    # ------------- 保存结果 -------------
    df.to_csv(results_dir / "daily_results_no_tc.csv")
    pd.DataFrame({
        'metric': ['n_days', 'cumulative_return', 'annualized_return', 'annualized_vol', 'sharpe', 'max_drawdown', 'trade_days', 'win_rate'],
        'value': [n_days, cumulative_return, annualized_return, ann_vol, sharpe, max_drawdown, trade_days, win_rate]
    }).to_csv(results_dir / "summary_metrics_no_tc.csv", index=False)

    # ------------- 绘制折线图：动量策略 VS 市场基准（不考虑交易成本）-------------
    plt.figure(figsize=(10,6))
    plt.plot(df.index, df['wealth_market'], label='Market (Buy&Hold)') # type: ignore
    plt.plot(df.index, df['wealth_strategy'], label='Momentum Strategy') # type: ignore
    plt.legend()
    plt.title(f"{TICKER} Momentum Strategy vs Market (Without Transaction Cost)")
    plt.ylabel("Wealth Index (start=1)")
    plt.xlabel("Date")
    plt.grid(True)
    plt.savefig(results_dir / "performance_no_tc.png", dpi=150)
    plt.close()

    # ------------- (2) 考虑交易成本 -------------
    print("")
    print("考虑交易成本：")
    # ------------- 计算每日回报 -------------
    # 每次换仓时扣 transaction cost（简单模型）
    tc = 0.0005
    df['trade'] = df['signal_shift'].diff().abs()  # type: ignore # 1 表示换仓
    df['strat_ret_net'] = df['strat_ret'] - df['trade'] * tc # type: ignore
    # 然后用 strat_ret_net 计算 wealth

    # ------------- 计算累计净值（wealth index）-------------
    df['wealth_strategy_net'] = (1 + df['strat_ret_net']).cumprod()

    # ------------- 绩效指标 -------------
    try:
        cumulative_return_net = df['wealth_strategy_net'].iloc[-1] - 1 # type: ignore
    except IndexError:
        cumulative_return_net = np.nan

    # 更稳妥的年化收益（基于日收益的几何平均）
    if n_days > 0 and not df['strat_ret_net'].isna().all(): # type: ignore
        annualized_return_net = np.expm1(df['strat_ret_net'].mean() * TRADING_DAYS) # type: ignore
    else:
        annualized_return_net = np.nan

    # 年化波动，明确 ddof（这里选 ddof=0）
    ann_vol_net = df['strat_ret_net'].std(ddof=0) * np.sqrt(TRADING_DAYS) if n_days > 0 else np.nan # type: ignore
    sharpe_net = (annualized_return_net / ann_vol_net) if (ann_vol_net and not np.isnan(ann_vol_net)) else np.nan

    # ------------- 最大回撤 -------------
    rolling_max_net = df['wealth_strategy_net'].cummax() # type: ignore
    max_drawdown_net = (df['wealth_strategy_net'] / rolling_max_net - 1).min() # type: ignore

    # ------------- 胜率 & 交易次数 -------------
    trade_days = transitions // 2  # type: ignore # 粗略估计换仓次数
    win_rate_net = (df.loc[df['strat_ret_net'] > 0, 'strat_ret_net'].count()) / (df.loc[df['signal_shift'] != 0, 'strat_ret_net'].count()+1e-9) # type: ignore

    print("净累计收益 (strategy): {:.2%}".format(cumulative_return_net))
    print("净年化收益: {:.2%}".format(annualized_return_net))
    print("净年化波动: {:.2%}".format(ann_vol_net))
    print("净夏普指数Sharpe (rf=0): {:.2f}".format(sharpe_net))
    print("净最大回撤:", f"{max_drawdown_net:.2%}" if not np.isnan(max_drawdown_net) else "nan")
    print("交易次数（transitions）:", trade_days)
    print("持仓天数:", position_days)
    print("净胜率（持仓期间的正日收益占比）: {:.2%}".format(win_rate_net))

    # ------------- 保存结果 -------------
    df.to_csv(results_dir / "daily_results_tc.csv", index=True) # type: ignore
    pd.DataFrame({
        'metric': ['n_days', 'cumulative_return', 'annualized_return', 'annualized_vol', 'sharpe', 'max_drawdown', 'trade_days', 'win_rate'],
        'value': [n_days, cumulative_return, annualized_return, ann_vol, sharpe, max_drawdown, trade_days, win_rate]
    }).to_csv(results_dir / "summary_metrics_tc.csv", index=False)

    # ------------- 绘制折线图：动量策略 VS 市场基准（考虑交易成本）-------------
    plt.figure(figsize=(10,6))
    plt.plot(df.index, df['wealth_market'], label='Market (Buy&Hold)') # type: ignore
    plt.plot(df.index, df['wealth_strategy'], label='Momentum Strategy') # type: ignore
    plt.legend()
    plt.title(f"{TICKER} Momentum Strategy vs Market (With Transaction Cost)")
    plt.ylabel("Wealth Index (start=1)")
    plt.xlabel("Date")
    plt.grid(True)
    plt.savefig(results_dir / "performance_tc.png", dpi=150)
    plt.close()