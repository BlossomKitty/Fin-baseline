# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# 基础工具
# -----------------------------
def load_symbols_from_txt(path: str) -> List[str]:
    p = Path(path)
    syms = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip().upper()
        if not s or s.startswith("#"):
            continue
        syms.append(s)
    # 去重保持顺序
    seen = set()
    out = []
    for s in syms:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def to_periodic_rebalance_dates(dates: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """
    把交易日序列转成重平衡日期集合。
    freq: daily/weekly/monthly
    - weekly: 每周最后一个交易日（W-FRI）
    - monthly: 每月最后一个交易日（ME）
    """
    freq = freq.lower()
    if freq == "daily":
        return dates
    s = pd.Series(1, index=dates)
    if freq == "weekly":
        return s.resample("W-FRI").last().dropna().index
    if freq == "monthly":
        # 旧版 "M" 被弃用，改成 "ME"（Month End）
        return s.resample("ME").last().dropna().index
    raise ValueError("freq 必须是 daily/weekly/monthly")


def zscore_cs(x: pd.Series) -> pd.Series:
    """横截面 z-score（同一天跨股票）"""
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True)
    if sd is None or sd == 0 or np.isnan(sd):
        return x * 0.0
    return (x - mu) / sd


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


@dataclass
class Perf:
    ann_return: float
    ann_vol: float
    sharpe: float
    max_dd: float
    avg_turnover: float
    total_cost: float


def compute_perf(daily_ret: pd.Series, daily_turnover: pd.Series, cost_bps: float) -> Perf:
    """
    daily_ret: 策略日收益（已扣成本或未扣成本均可）
    daily_turnover: 每日换手（0.5 * sum(|w_t - w_{t-1}|)）
    """
    n = len(daily_ret)
    if n == 0:
        return Perf(0, 0, 0, 0, 0, 0)

    equity = (1.0 + daily_ret.fillna(0)).cumprod()
    ann_return = float(equity.iloc[-1] ** (252.0 / n) - 1.0)
    ann_vol = float(daily_ret.std(ddof=0) * np.sqrt(252.0))
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else 0.0
    mdd = max_drawdown(equity)

    # 交易成本（bps）：cost = turnover * cost_bps/10000
    cost = (daily_turnover.fillna(0) * (cost_bps / 10000.0)).sum()

    return Perf(
        ann_return=ann_return,
        ann_vol=ann_vol,
        sharpe=sharpe,
        max_dd=mdd,
        avg_turnover=float(daily_turnover.fillna(0).mean()),
        total_cost=float(cost),
    )


def read_ohlcv(
    ohlcv_csv: str,
    start: str,
    end: str,
    symbols: Optional[List[str]] = None,
    ret_clip: Optional[float] = None,
    min_history: int = 0,
) -> pd.DataFrame:
    """
    读取 Alpaca 合并 OHLCV（date,tic,open,high,low,close,volume）
    返回 long 表：date,symbol,close,ret_1d

    ret_clip: 可选，裁剪日收益 [-ret_clip, +ret_clip]，用于抑制异常跳点
    min_history: 可选，过滤历史太短股票（交易日数量不足 min_history 的剔除）
    """
    df = pd.read_csv(ohlcv_csv)

    # 兼容列名
    if "tic" in df.columns and "symbol" not in df.columns:
        df = df.rename(columns={"tic": "symbol"})

    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str).str.upper()

    df = df[(df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))]

    if symbols is not None:
        sset = set([x.upper() for x in symbols])
        df = df[df["symbol"].isin(sset)]

    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    # close-to-close 日收益
    df["ret_1d"] = df.groupby("symbol")["close"].pct_change()

    # 可选：裁剪极端收益（避免少数跳点把动量/事件策略打穿）
    if ret_clip is not None and ret_clip > 0:
        df["ret_1d"] = df["ret_1d"].clip(-ret_clip, ret_clip)

    # 可选：过滤历史太短的股票（避免新上市/缺失导致信号不稳）
    if min_history and min_history > 0:
        cnt = df.groupby("symbol")["date"].count()
        keep = cnt[cnt >= int(min_history)].index
        df = df[df["symbol"].isin(set(keep))].reset_index(drop=True)

    return df[["date", "symbol", "close", "ret_1d"]]


def read_features_panel(panel_parquet: str, start: str, end: str, symbols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    读取 _panel.parquet（date,symbol,...features...）
    """
    df = pd.read_parquet(panel_parquet)
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df = df[(df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))]
    if symbols is not None:
        sset = set([x.upper() for x in symbols])
        df = df[df["symbol"].isin(sset)]
    return df


def make_returns_matrix(ohlcv_long: pd.DataFrame) -> Tuple[pd.DatetimeIndex, List[str], pd.DataFrame]:
    """
    返回：
    - dates: 交易日 index
    - symbols: 列顺序
    - R: DataFrame(index=dates, columns=symbols) 的 ret_1d
    """
    pivot = ohlcv_long.pivot(index="date", columns="symbol", values="ret_1d").sort_index()
    dates = pivot.index
    symbols = list(pivot.columns)
    return dates, symbols, pivot


def weights_to_turnover(w: pd.DataFrame) -> pd.Series:
    """
    换手（常用口径）：0.5 * sum(|w_t - w_{t-1}|)
    """
    dw = w.diff().abs()
    return 0.5 * dw.sum(axis=1)


def apply_weights_to_returns(w: pd.DataFrame, R: pd.DataFrame) -> pd.Series:
    """
    不提前知道未来：用 w_{t-1} 乘以 R_t
    修复点：当某些股票 R_t 缺失时，对有效权重重新归一，避免“投资比例漂移”导致波动/收益畸变。
    """
    w_lag = w.shift(1)

    # 有效掩码：当日该股票有收益
    mask = R.notna()

    # 只保留有效部分的权重
    w_eff = w_lag.where(mask, 0.0)

    # 每天有效权重之和（避免为0）
    denom = w_eff.sum(axis=1).replace(0.0, np.nan)

    # 归一到 1（只在有效股票上）
    w_eff = w_eff.div(denom, axis=0).fillna(0.0)

    port = (w_eff * R.fillna(0.0)).sum(axis=1)
    return port



def save_equity(out_dir: Path, name: str, daily_ret: pd.Series) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    equity = (1.0 + daily_ret.fillna(0)).cumprod()
    df = pd.DataFrame({"date": equity.index, "daily_ret": daily_ret.values, "equity": equity.values})
    p = out_dir / f"equity_{name}.csv"
    df.to_csv(p, index=False, encoding="utf-8")
    return p


# -----------------------------
# Strategy 1: 等权 / 买入持有
# -----------------------------
def strat_equal_weight(dates: pd.DatetimeIndex, symbols: List[str], rebalance: str = "daily") -> pd.DataFrame:
    """
    每次重平衡，把权重设为等权（1/N）
    """
    idx_reb = to_periodic_rebalance_dates(dates, rebalance)
    w = pd.DataFrame(0.0, index=dates, columns=symbols)
    if len(symbols) == 0:
        return w

    target = np.ones(len(symbols)) / len(symbols)
    for d in idx_reb:
        if d in w.index:
            w.loc[d, :] = target

    # 持有到下次重平衡：前向填充
    w = w.replace(0.0, np.nan).ffill().fillna(0.0)
    return w


def strat_buy_and_hold(dates: pd.DatetimeIndex, symbols: List[str]) -> pd.DataFrame:
    """
    买入并持有：第一天等权买入，之后不再调仓
    注意：必须用 NaN + ffill，不能用 0 + ffill（0 不会被视为缺失）
    """
    w = pd.DataFrame(np.nan, index=dates, columns=symbols)
    if len(symbols) == 0:
        return w.fillna(0.0)

    w.iloc[0, :] = 1.0 / len(symbols)
    w = w.ffill().fillna(0.0)
    return w


# -----------------------------
# Strategy 2: 传统信号 baseline
# -----------------------------
def calc_momentum_signal(price: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """
    动量：过去 lookback 日收益（close/close_{t-lookback}-1）
    """
    return price / price.shift(lookback) - 1.0


def calc_inv_vol_signal(R: pd.DataFrame, vol_window: int) -> pd.DataFrame:
    """
    inv-vol：1 / rolling std
    """
    vol = R.rolling(vol_window).std()
    return 1.0 / vol.replace(0.0, np.nan)


def cross_section_select_weights(
    signal: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    top_k: int,
    mode: str = "top",
    weight_mode: str = "equal",
) -> pd.DataFrame:
    """
    在每个 rebalance_date 做横截面选股
    - mode=top：选 signal 最大 top_k
    - mode=bottom：选 signal 最小 top_k
    weight_mode:
    - equal：等权
    - prop：按 signal 正比例（会自动非负化）
    """
    dates = signal.index
    symbols = list(signal.columns)
    w = pd.DataFrame(0.0, index=dates, columns=symbols)

    for d in rebalance_dates:
        if d not in signal.index:
            continue
        s = signal.loc[d].copy()
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) == 0:
            continue

        k = len(s) if top_k <= 0 or top_k > len(s) else top_k

        if mode == "top":
            picks = s.sort_values(ascending=False).head(k)
        elif mode == "bottom":
            picks = s.sort_values(ascending=True).head(k)
        else:
            raise ValueError("mode must be top/bottom")

        if weight_mode == "equal":
            w.loc[d, picks.index] = 1.0 / len(picks)
        elif weight_mode == "prop":
            v = picks.values
            v = np.maximum(v, 0.0)
            if v.sum() <= 0:
                w.loc[d, picks.index] = 1.0 / len(picks)
            else:
                w.loc[d, picks.index] = v / v.sum()
        else:
            raise ValueError("weight_mode must be equal/prop")

    w = w.replace(0.0, np.nan).ffill().fillna(0.0)
    return w


# -----------------------------
# Strategy 3: 结构化基本面 baseline
# -----------------------------
def make_fundamental_score(panel: pd.DataFrame) -> pd.DataFrame:
    """
    返回 score(date,symbol)（越大越好）
    """
    df = panel.copy()

    cols_need = [
        "r_q_priceToEarningsRatio",  # 越低越好
        "r_q_priceToBookRatio",      # 越低越好
        "r_q_netProfitMargin",       # 越高越好
        "km_q_netDebtToEBITDA",      # 越低越好
    ]
    for c in cols_need:
        if c not in df.columns:
            df[c] = np.nan

    for c in cols_need:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 季频延续
    df = df.sort_values(["symbol", "date"])
    df[cols_need] = df.groupby("symbol")[cols_need].ffill()

    out = []
    for d, g in df.groupby("date"):
        g = g.copy()
        pe_z = zscore_cs(g["r_q_priceToEarningsRatio"])
        pb_z = zscore_cs(g["r_q_priceToBookRatio"])
        mar_z = zscore_cs(g["r_q_netProfitMargin"])
        nd_z = zscore_cs(g["km_q_netDebtToEBITDA"])

        score = (-1.0) * pe_z + (-1.0) * pb_z + (1.0) * mar_z + (-1.0) * nd_z
        g["score"] = score.replace([np.inf, -np.inf], np.nan)
        out.append(g[["date", "symbol", "score"]])

    return pd.concat(out, ignore_index=True)


def weights_from_score(
    dates: pd.DatetimeIndex,
    symbols: List[str],
    score_long: pd.DataFrame,
    rebalance: str,
    top_k: int,
) -> pd.DataFrame:
    """
    用 score 做 top-k 选股，等权
    """
    reb = to_periodic_rebalance_dates(dates, rebalance)
    w = pd.DataFrame(0.0, index=dates, columns=symbols)
    score_pivot = score_long.pivot(index="date", columns="symbol", values="score").reindex(index=dates, columns=symbols)

    for d in reb:
        if d not in score_pivot.index:
            continue
        s = score_pivot.loc[d].dropna()
        if len(s) == 0:
            continue
        k = len(s) if top_k <= 0 or top_k > len(s) else top_k
        picks = s.sort_values(ascending=False).head(k)
        w.loc[d, picks.index] = 1.0 / len(picks)

    w = w.replace(0.0, np.nan).ffill().fillna(0.0)
    return w


# -----------------------------
# Strategy 4: 新闻/公告计数（不做LLM）
# -----------------------------
def make_event_intensity_score(panel: pd.DataFrame, field: str) -> pd.DataFrame:
    """
    field: news_count 或 filing_count
    返回 score(date,symbol)（越大越“事件多”）
    """
    df = panel[["date", "symbol", field]].copy()
    df[field] = pd.to_numeric(df[field], errors="coerce").fillna(0.0)
    df["score"] = np.log1p(df[field])
    return df[["date", "symbol", "score"]]


# -----------------------------
# 主程序
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ohlcv-csv", required=True, help="Alpaca 合并日线 CSV（combined_daily_adjusted.csv）")
    ap.add_argument("--panel-parquet", default=None, help="features_daily/_panel.parquet（做策略3/4用）")
    ap.add_argument("--out-dir", default="out/bt_results", help="输出目录")
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default=None)

    ap.add_argument("--symbols-file", default=None, help="可选：只回测指定股票（txt每行一个）")
    ap.add_argument("--strategy", required=True, choices=[
        "buyhold",
        "equal_weight",
        "mom_topk",
        "mr_bottomk",
        "invvol_all",
        "fund_topk",
        "news_topk",
        "filing_topk",
    ])

    ap.add_argument("--rebalance", default="weekly", choices=["daily", "weekly", "monthly"])
    ap.add_argument("--top-k", type=int, default=50, help="top-k 选股数量（策略2/3/4用）")
    ap.add_argument("--lookback", type=int, default=60, help="动量/均值回复 lookback（日）")
    ap.add_argument("--vol-window", type=int, default=20, help="inv-vol 波动窗口（日）")
    ap.add_argument("--cost-bps", type=float, default=2.0, help="单边交易成本（bps），按换手计")
    ap.add_argument("--save-weights", action="store_true", help="保存每日权重 parquet（可能较大）")

    # 新增：可选稳健开关
    ap.add_argument("--ret-clip", type=float, default=None, help="可选：裁剪日收益绝对值上限，如 0.2 表示±20%")
    ap.add_argument("--min-history", type=int, default=0, help="可选：过滤历史太短股票（交易日数量不足则剔除）")

    args = ap.parse_args()

    if args.end is None:
        args.end = pd.Timestamp.today().strftime("%Y-%m-%d")

    symbols = None
    if args.symbols_file:
        symbols = load_symbols_from_txt(args.symbols_file)

    # 读取 OHLCV
    ohlcv = read_ohlcv(
        args.ohlcv_csv,
        args.start,
        args.end,
        symbols=symbols,
        ret_clip=args.ret_clip,
        min_history=int(args.min_history or 0),
    )
    dates, syms, R = make_returns_matrix(ohlcv)

    # price matrix（动量需要）
    price = ohlcv.pivot(index="date", columns="symbol", values="close").reindex(index=dates, columns=syms).sort_index()

    # 构造权重
    if args.strategy == "buyhold":
        w = strat_buy_and_hold(dates, syms)

    elif args.strategy == "equal_weight":
        w = strat_equal_weight(dates, syms, rebalance=args.rebalance)

    elif args.strategy == "mom_topk":
        sig = calc_momentum_signal(price, lookback=int(args.lookback))
        reb = to_periodic_rebalance_dates(dates, args.rebalance)
        w = cross_section_select_weights(sig, reb, top_k=int(args.top_k), mode="top", weight_mode="equal")

    elif args.strategy == "mr_bottomk":
        sig = calc_momentum_signal(price, lookback=int(args.lookback))
        reb = to_periodic_rebalance_dates(dates, args.rebalance)
        w = cross_section_select_weights(sig, reb, top_k=int(args.top_k), mode="bottom", weight_mode="equal")

    elif args.strategy == "invvol_all":
        sig = calc_inv_vol_signal(R, vol_window=int(args.vol_window))
        reb = to_periodic_rebalance_dates(dates, args.rebalance)
        w = cross_section_select_weights(sig, reb, top_k=0, mode="top", weight_mode="prop")

    elif args.strategy in ("fund_topk", "news_topk", "filing_topk"):
        if not args.panel_parquet:
            raise ValueError("策略 fund_topk/news_topk/filing_topk 需要 --panel-parquet 指向 _panel.parquet")

        panel = read_features_panel(args.panel_parquet, args.start, args.end, symbols=syms)

        if args.strategy == "fund_topk":
            score_long = make_fundamental_score(panel)
            w = weights_from_score(dates, syms, score_long, rebalance=args.rebalance, top_k=int(args.top_k))

        elif args.strategy == "news_topk":
            score_long = make_event_intensity_score(panel, field="news_count")
            w = weights_from_score(dates, syms, score_long, rebalance=args.rebalance, top_k=int(args.top_k))

        else:
            score_long = make_event_intensity_score(panel, field="filing_count")
            w = weights_from_score(dates, syms, score_long, rebalance=args.rebalance, top_k=int(args.top_k))

    else:
        raise ValueError("未知策略")

    # 组合收益（用 w_{t-1} * R_t，避免未来函数）
    port_ret = apply_weights_to_returns(w, R).fillna(0.0)

    # 换手与成本
    turnover = weights_to_turnover(w).fillna(0.0)
    cost = turnover * (float(args.cost_bps) / 10000.0)
    port_ret_net = port_ret - cost

    perf_gross = compute_perf(port_ret, turnover, cost_bps=float(args.cost_bps))
    perf_net = compute_perf(port_ret_net, turnover, cost_bps=float(args.cost_bps))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    name = args.strategy
    if args.strategy in ("mom_topk", "mr_bottomk"):
        name += f"_lb{args.lookback}_k{args.top_k}_{args.rebalance}"
    if args.strategy in ("invvol_all",):
        name += f"_vw{args.vol_window}_{args.rebalance}"
    if args.strategy in ("fund_topk", "news_topk", "filing_topk"):
        name += f"_k{args.top_k}_{args.rebalance}"
    if args.strategy in ("equal_weight",):
        name += f"_{args.rebalance}"
    if args.ret_clip is not None:
        name += f"_clip{args.ret_clip}"
    if args.min_history and args.min_history > 0:
        name += f"_minhist{args.min_history}"

    eq_path = save_equity(out_dir, name, port_ret_net)

    if args.save_weights:
        w_path = out_dir / f"weights_{name}.parquet"
        w.to_parquet(w_path)
        print(f"[OK] weights saved -> {w_path}")

    print("\n================= BACKTEST SUMMARY =================")
    print(f"Strategy: {name}")
    print(f"Dates: {dates.min().date()} .. {dates.max().date()} | symbols={len(syms)}")
    if args.ret_clip is not None:
        print(f"ret_clip: ±{args.ret_clip}")
    if args.min_history and args.min_history > 0:
        print(f"min_history: {args.min_history} trading days")
    print(f"Cost (bps): {args.cost_bps}")
    print("---- Gross ----")
    print(f"AnnRet={perf_gross.ann_return:.4f}  AnnVol={perf_gross.ann_vol:.4f}  Sharpe={perf_gross.sharpe:.3f}  MaxDD={perf_gross.max_dd:.4f}")
    print("---- Net (after cost) ----")
    print(f"AnnRet={perf_net.ann_return:.4f}  AnnVol={perf_net.ann_vol:.4f}  Sharpe={perf_net.sharpe:.3f}  MaxDD={perf_net.max_dd:.4f}")
    print(f"AvgTurnover/day={perf_net.avg_turnover:.4f}  TotalCost(=sum turnover*bps)={perf_net.total_cost:.4f}")
    print(f"[OK] equity curve saved -> {eq_path}")
    print("====================================================\n")


if __name__ == "__main__":
    main()
