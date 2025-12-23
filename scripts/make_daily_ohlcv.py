# -*- coding: utf-8 -*-
"""
make_daily_ohlcv.py

用途
----
把已有的「复权后的日线合并 CSV」（例如你本地的：
  D:\FinML\out\combined_daily_adjusted_5y.csv
常见列：date, tic, open, high, low, close, volume）
整理成统一格式，并输出：

1) 标准日线 OHLCV（合并文件）：
   date,symbol,open,high,low,close,volume

2) 真实交易日历（唯一日期列表）：
   timestamp
   （timestamp 为 YYYY-MM-DD 字符串）

3)（可选）按股票拆分的 OHLCV：
   每个 symbol 一个 CSV

重要说明
--------
你的输入 date 已经是“日频日期”（如 2020/12/23 或 2020-12-23），
因此这里【不做任何时区转换】，避免出现日期被挪到前一天的情况。

PowerShell 运行示例
-------------------
python D:\FinML\scripts\make_daily_ohlcv.py `
  --input "D:\FinML\out\combined_daily_adjusted_5y.csv" `
  --out "D:\FinML\out\ohlcv_daily.csv" `
  --calendar-out "D:\FinML\out\alpaca_calendar.csv" `
  --per-symbol-dir "D:\FinML\out\ohlcv_by_symbol"

如果你的输入列名不一样，可以用显式映射：
  --date-col date --symbol-col tic --open-col open --high-col high --low-col low --close-col close --volume-col volume
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def pick_col(cols, candidates) -> Optional[str]:
    """从候选列名中（忽略大小写）选出第一个匹配的列名。"""
    cols_l = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_l:
            return cols_l[cand.lower()]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="输入：复权后的日线合并 CSV")
    ap.add_argument("--out", required=True, help="输出：标准日线 OHLCV 合并 CSV")
    ap.add_argument("--calendar-out", default=None, help="输出：真实交易日历 CSV（唯一日期）")
    ap.add_argument("--per-symbol-dir", default=None, help="可选：输出按股票拆分的 OHLCV CSV 文件夹")

    # 可选：显式指定列名映射（当你的输入 schema 发生变化时使用）
    ap.add_argument("--date-col", default=None, help="日期列名（可选）")
    ap.add_argument("--symbol-col", default=None, help="股票代码列名（可选）")
    ap.add_argument("--open-col", default=None, help="开盘价列名（可选）")
    ap.add_argument("--high-col", default=None, help="最高价列名（可选）")
    ap.add_argument("--low-col", default=None, help="最低价列名（可选）")
    ap.add_argument("--close-col", default=None, help="收盘价列名（可选）")
    ap.add_argument("--volume-col", default=None, help="成交量列名（可选）")

    args = ap.parse_args()

    inp = Path(args.input)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)

    # --- 自动识别列名（如有显式指定则优先使用） ---
    date_col = args.date_col or pick_col(df.columns, ["date", "t", "timestamp", "time", "datetime"])
    sym_col = args.symbol_col or pick_col(df.columns, ["symbol", "tic", "ticker", "sym"])
    o_col = args.open_col or pick_col(df.columns, ["open", "o"])
    h_col = args.high_col or pick_col(df.columns, ["high", "h"])
    l_col = args.low_col or pick_col(df.columns, ["low", "l"])
    c_col = args.close_col or pick_col(df.columns, ["close", "c"])
    v_col = args.volume_col or pick_col(df.columns, ["volume", "v", "vol"])

    # --- 检查必要列是否齐全 ---
    missing = []
    if not date_col:
        missing.append("date（日期列）")
    if not sym_col:
        missing.append("symbol/tic/ticker（股票代码列）")
    if not o_col:
        missing.append("open（开盘价列）")
    if not h_col:
        missing.append("high（最高价列）")
    if not l_col:
        missing.append("low（最低价列）")
    if not c_col:
        missing.append("close（收盘价列）")
    if not v_col:
        missing.append("volume（成交量列）")

    if missing:
        raise ValueError(
            "无法自动识别必要列，请检查输入文件列名。\n"
            f"缺失：{missing}\n"
            f"可用列：{df.columns.tolist()}\n"
            "提示：你可以用 --date-col/--symbol-col/... 显式指定列名映射。"
        )

    # --- 解析日期（不做时区转换：输入已是“日线日期”） ---
    dates = pd.to_datetime(df[date_col], errors="coerce")

    # --- 组装标准输出表 ---
    df_out = pd.DataFrame(
        {
            "date": dates.dt.strftime("%Y-%m-%d"),
            "symbol": df[sym_col].astype("string").str.upper(),
            "open": pd.to_numeric(df[o_col], errors="coerce"),
            "high": pd.to_numeric(df[h_col], errors="coerce"),
            "low": pd.to_numeric(df[l_col], errors="coerce"),
            "close": pd.to_numeric(df[c_col], errors="coerce"),
            "volume": pd.to_numeric(df[v_col], errors="coerce"),
        }
    )

    # --- 删除解析失败或缺失的行 ---
    df_out = df_out.dropna(subset=["date", "symbol", "open", "high", "low", "close", "volume"])

    # --- 排序 + 去重（同一 symbol 同一天保留最后一条） ---
    df_out = df_out.sort_values(["symbol", "date"])
    df_out = df_out.drop_duplicates(subset=["symbol", "date"], keep="last").reset_index(drop=True)

    # --- 保存合并 OHLCV ---
    df_out.to_csv(outp, index=False, encoding="utf-8")
    print(f"[OK] 已保存：合并日线 OHLCV -> {outp}")
    print(
        f"     行数={len(df_out)}, 股票数={df_out['symbol'].nunique()}, "
        f"日期范围=[{df_out['date'].min()} .. {df_out['date'].max()}]"
    )

    # --- 保存交易日历（唯一日期列表） ---
    if args.calendar_out:
        calp = Path(args.calendar_out)
        calp.parent.mkdir(parents=True, exist_ok=True)
        cal_dates = sorted(df_out["date"].unique())
        pd.DataFrame({"timestamp": cal_dates}).to_csv(calp, index=False, encoding="utf-8")
        print(f"[OK] 已保存：交易日历 -> {calp}（{len(cal_dates)} 行）")

    # --- 保存按股票拆分的 OHLCV（可选） ---
    if args.per_symbol_dir:
        d = Path(args.per_symbol_dir)
        d.mkdir(parents=True, exist_ok=True)
        for sym, g in df_out.groupby("symbol", sort=True):
            g.to_csv(d / f"{sym}.csv", index=False, encoding="utf-8")
        print(f"[OK] 已保存：按股票拆分 OHLCV -> {d}（{df_out['symbol'].nunique()} 个文件）")


if __name__ == "__main__":
    main()
