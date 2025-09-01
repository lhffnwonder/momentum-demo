# momentum_backtest.py
# 交易成本影响动量策略的结果
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
# —— 可靠的 results 定位 + 保存代码（直接粘贴运行） ——
from pathlib import Path
import os
from pathlib import Path

from compute_metrics import compute_metrics

# ----------------- 简单工具函数 ----------------
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def to_pct(x):
    try:
        return f"{x:.2%}"
    except:
        return str(x)

# ----------------- 参数区（可调） -----------------
TICKER = "SPY"            # 标的
START = "2022-01-01"      # 起始日（可改）
END = datetime.today().strftime("%Y-%m-%d")
WINDOW = 20               # 动量窗口（天）
TRADING_DAYS = 252        # 年化换算系数（交易日）

# 交易成本（单边），可调
tc = 0.0005  # 0.05% 单边成本
# 成本感知开仓/平仓阈值（可调）
open_threshold = 2 * tc   # 开仓阈值：只有当动量 > round-trip 成本时才考虑开仓
close_threshold = 0.0     # 平仓阈值：当动量 <= close_threshold 时离场

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

# ----------------- 拉数据（显式 auto_adjust） -----------------
# 为了可复现性，显式设置 auto_adjust=False（或设为 True 若要用调整后价格）
df = yf.download(TICKER, start=START, end=END, progress=False, auto_adjust=False)
# print("样本行数：", len(df)) # type: ignore
df = df[['Close']].dropna() # type: ignore
df.sort_index(inplace=True)
# print("样本时间范围:", df.index.min(), "->", df.index.max())

# ----------------- 计算动量信号 -----------------
# 过去 WINDOW 天收益率作为动量
df['mom'] = df['Close'].pct_change(WINDOW)

# 原始信号（简单动量 > 0）
df['signal_raw'] = (df['mom'] > 0).astype(int)

# 执行时延（避免未来函数）：使用前一天信号决定当日持仓
df['signal_shift'] = df['signal_raw'].shift(1).fillna(0)

# ----------------- 市场与策略回报（无成本） -----------------
df['ret_daily'] = df['Close'].pct_change().fillna(0)
df['strat_ret'] = df['signal_shift'] * df['ret_daily']
df['wealth_market'] = (1 + df['ret_daily']).cumprod()
df['wealth_strategy'] = (1 + df['strat_ret']).cumprod()

# ----------------- 把交易成本直接扣在换仓日，得到原始策略的净收益（信号不变） -------------
df['trade_side_raw'] = df['signal_shift'].diff().abs().fillna(0)
df['strat_ret_net'] = df['strat_ret'] - df['trade_side_raw'] * tc
df['wealth_strategy_net'] = (1 + df['strat_ret_net']).cumprod()

# ----------------- 成本感知信号（门槛 + 滞后/hysteresis） -----------------
mom_arr = df['mom'].to_numpy()   # numpy array of floats (with np.nan)
sig_ca = np.zeros(len(mom_arr), dtype=int)
prev = 0
for i, mom_v in enumerate(mom_arr):
    # mom_v 是 numpy.float 或 np.nan, np.isnan 能安全使用
    if np.isnan(mom_v):
        sig_ca[i] = prev
        continue
    if prev == 0:
        # 未持仓，判断是否开仓（动量需超过阈值）
        sig_ca[i] = 1 if mom_v > open_threshold else 0
    else:
        # 已持仓，判断是否平仓
        sig_ca[i] = 0 if mom_v <= close_threshold else 1
    prev = sig_ca[i]

df['signal_costaware'] = pd.Series(sig_ca, index=df.index)
# 执行时延：用前一天 cost-aware 信号决定当日持仓
df['signal_cost_shift'] = df['signal_costaware'].shift(1).fillna(0)

# ----------------- 成本感知策略回测（毛/净） -----------------
df['strat_ret_ca'] = df['signal_cost_shift'] * df['ret_daily']
df['trade_side_ca'] = df['signal_cost_shift'].diff().abs().fillna(0)
df['strat_ret_ca_net'] = df['strat_ret_ca'] - df['trade_side_ca'] * tc
df['wealth_strategy_ca'] = (1 + df['strat_ret_ca']).cumprod()
df['wealth_strategy_ca_net'] = (1 + df['strat_ret_ca_net']).cumprod()

# ----------------- 计算各类指标 -----------------
metrics = {}
metrics['gross'] = compute_metrics(df, 'strat_ret', 'wealth_strategy', 'signal_shift', TRADING_DAYS)
metrics['gross']['market_cumulative'] = df['wealth_market'].iloc[-1] - 1
metrics['net'] = compute_metrics(df, 'strat_ret_net', 'wealth_strategy_net', 'signal_shift', TRADING_DAYS)
metrics['cost_aware_gross'] = compute_metrics(df, 'strat_ret_ca', 'wealth_strategy_ca', 'signal_cost_shift', TRADING_DAYS)
metrics['cost_aware_net'] = compute_metrics(df, 'strat_ret_ca_net', 'wealth_strategy_ca_net', 'signal_cost_shift', TRADING_DAYS)

# ----------------- 打印清晰对比（尽量保持你原来的输出格式） -----------------
print("\n=== 对比总结 ===")
print(f"样本行数: {metrics['gross']['n_days']}  时间区间: {df.index.min()} -> {df.index.max()}")
def p(x): return f"{x:.2%}" if (isinstance(x, (float, np.floating)) and not np.isnan(x)) else str(x)

print("\n-- 原始策略（gross） --")
print("累计收益:", p(metrics['gross']['cumulative_return']), "年化:", p(metrics['gross']['annualized_return']), "Sharpe:", f"{metrics['gross']['sharpe']:.2f}")
print("最大回撤:", p(metrics['gross']['max_drawdown']), "换手(side_trades):", p(metrics['gross']['side_trades']), "round-trips:", metrics['gross']['round_trips'])
print("持仓天数:", metrics['gross']['holding_days'], f"({metrics['gross']['holding_pct']:.2%})", "胜率:", p(metrics['gross']['win_rate']))

print("\n-- 原始策略（net, after tc={:.4%}) --".format(tc))
print("累计收益:", p(metrics['net']['cumulative_return']), "年化:", p(metrics['net']['annualized_return']), "Sharpe:", f"{metrics['net']['sharpe']:.2f}")
print("最大回撤:", p(metrics['net']['max_drawdown']), "换手(side_trades):", p(metrics['net']['side_trades']), "round-trips:", metrics['net']['round_trips'])
print("持仓天数:", metrics['net']['holding_days'], f"({metrics['net']['holding_pct']:.2%})", "胜率:", p(metrics['net']['win_rate']))

print("\n-- 成本感知策略（gross） --")
print("累计收益:", p(metrics['cost_aware_gross']['cumulative_return']), "年化:", p(metrics['cost_aware_gross']['annualized_return']), "Sharpe:", f"{metrics['cost_aware_gross']['sharpe']:.2f}")
print("最大回撤:", p(metrics['cost_aware_gross']['max_drawdown']), "换手(side_trades):", p(metrics['cost_aware_gross']['side_trades']), "round-trips:", metrics['cost_aware_gross']['round_trips'])
print("持仓天数:", metrics['cost_aware_gross']['holding_days'], f"({metrics['cost_aware_gross']['holding_pct']:.2%})", "胜率:", p(metrics['cost_aware_gross']['win_rate']))

print("\n-- 成本感知策略（net, after tc={:.4%}) --".format(tc))
print("累计收益:", p(metrics['cost_aware_net']['cumulative_return']), "年化:", p(metrics['cost_aware_net']['annualized_return']), "Sharpe:", f"{metrics['cost_aware_net']['sharpe']:.2f}")
print("最大回撤:", p(metrics['cost_aware_net']['max_drawdown']), "换手(side_trades):", p(metrics['cost_aware_net']['side_trades']), "round-trips:", metrics['cost_aware_net']['round_trips'])
print("持仓天数:", metrics['cost_aware_net']['holding_days'], f"({metrics['cost_aware_net']['holding_pct']:.2%})", "胜率:", p(metrics['cost_aware_net']['win_rate']))

# ----------------- 保存 daily results（包含所有列） -----------------
df.to_csv(results_dir / "daily_results_comparison.csv", index=True)
# 保存 summary 对比 CSV
summary_rows = []
for k, v in metrics.items():
    prefix = k
    for metric_name, val in v.items():
        summary_rows.append({
            "strategy": prefix,
            "metric": metric_name,
            "value": val
        })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(results_dir / "summary_metrics_comparison.csv", index=False)
print("对比CSV文件保存在：", results_dir / "summary_metrics_comparison.csv")

# ----------------- 绘图：市场、原始策略、成本感知策略（net） -------------
plt.figure(figsize=(12,6))
plt.plot(df.index, df['wealth_market'], label='Market (Buy&Hold)', linestyle='--', linewidth=1.5)
plt.plot(df.index, df['wealth_strategy'], label='Momentum (gross)', linewidth=1)
plt.plot(df.index, df['wealth_strategy_net'], label='Momentum (net)', linewidth=1)
plt.plot(df.index, df['wealth_strategy_ca_net'], label='Momentum cost-aware (net)', linewidth=2)
plt.legend()
plt.title(f"{TICKER} Momentum Strategy Comparison (tc={tc:.4%})")
plt.ylabel("Wealth Index (start=1)")
plt.xlabel("Date")
plt.grid(True)
plt.tight_layout()
plt.savefig(results_dir / "performance_comparison.png", dpi=150)
plt.close()
print("对比折线图保存在：", results_dir / "performance_comparison.png")