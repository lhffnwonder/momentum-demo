import numpy as np

def compute_metrics(df, ret_col, wealth_col, signal_col, trading_days=252):
    """返回指标字典：用于任意收益列与信号列"""
    n_days = len(df)
    # cumulative
    cumulative = df[wealth_col].iloc[-1] - 1
    # annualized (geometric)
    if n_days > 0:
        annualized = (1 + cumulative) ** (trading_days / n_days) - 1
    else:
        annualized = np.nan
    ann_vol = df[ret_col].std(ddof=0) * np.sqrt(trading_days) if n_days > 0 else np.nan
    sharpe = annualized / ann_vol if (ann_vol and not np.isnan(ann_vol)) else np.nan
    rolling_max = df[wealth_col].cummax()
    max_dd = (df[wealth_col] / rolling_max - 1).min()
    # trades / holding
    side_trades = int(df[signal_col].diff().abs().sum())
    round_trips = side_trades // 2
    holding_days = int((df[signal_col] != 0).sum())
    holding_pct = holding_days / n_days if n_days > 0 else np.nan
    # win rate (during holding)
    if df.loc[df[signal_col] != 0, ret_col].count() > 0:
        win_rate = df.loc[df[signal_col] != 0, ret_col].gt(0).mean()
    else:
        win_rate = np.nan

    return {
        "n_days": n_days,
        "cumulative_return": cumulative,
        "annualized_return": annualized,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "side_trades": side_trades,
        "round_trips": round_trips,
        "holding_days": holding_days,
        "holding_pct": holding_pct,
        "win_rate": win_rate
    }