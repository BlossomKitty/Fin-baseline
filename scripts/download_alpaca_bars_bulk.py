# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests


BASE_URL = "https://data.alpaca.markets/v2/stocks/bars"


def load_symbols(path: str) -> List[str]:
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


def chunk_list(xs: List[str], n: int) -> List[List[str]]:
    return [xs[i : i + n] for i in range(0, len(xs), n)]


@dataclass
class AlpacaAuth:
    key_id: str
    secret_key: str
    feed: str  # iex or sip


def get_auth() -> AlpacaAuth:
    key_id = os.environ.get("ALPACA_API_KEY_ID") or os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY")
    feed = os.environ.get("ALPACA_DATA_FEED", "iex").strip().lower()

    if not key_id or not secret:
        raise RuntimeError(
            "未找到 Alpaca Key。请在环境变量设置：\n"
            "  $env:ALPACA_API_KEY_ID='xxx'\n"
            "  $env:ALPACA_API_SECRET_KEY='yyy'\n"
            "可选：$env:ALPACA_DATA_FEED='iex' 或 'sip'"
        )
    if feed not in ("iex", "sip"):
        raise RuntimeError("ALPACA_DATA_FEED 只能是 iex 或 sip")
    return AlpacaAuth(key_id=key_id, secret_key=secret, feed=feed)


def alpaca_get_bars(
    session: requests.Session,
    auth: AlpacaAuth,
    symbols: List[str],
    start: str,
    end: str,
    timeframe: str = "1Day",
    adjustment: str = "all",
    limit: int = 10000,
    max_retries: int = 6,
) -> Dict[str, List[dict]]:
    """
    使用 Alpaca /v2/stocks/bars 批量获取 bars。返回：{symbol: [bar, ...]}
    自动翻页（page_token）。
    """
    headers = {
        "APCA-API-KEY-ID": auth.key_id,
        "APCA-API-SECRET-KEY": auth.secret_key,
    }

    params = {
        "symbols": ",".join(symbols),
        "timeframe": timeframe,
        "start": start,
        "end": end,
        "adjustment": adjustment,
        "feed": auth.feed,
        "limit": limit,
    }

    out: Dict[str, List[dict]] = {s: [] for s in symbols}
    page_token = None

    while True:
        if page_token:
            params["page_token"] = page_token
        else:
            params.pop("page_token", None)

        # 带退避的重试
        last_err = None
        for attempt in range(max_retries):
            try:
                r = session.get(BASE_URL, headers=headers, params=params, timeout=60)
                if r.status_code == 429:
                    # rate limit
                    sleep_s = min(60, 2 ** attempt)
                    time.sleep(sleep_s)
                    continue
                if r.status_code >= 400:
                    text_head = r.text[:400]
                    raise RuntimeError(f"HTTP {r.status_code}: {text_head}")

                data = r.json()
                bars_map = data.get("bars") or {}

                for sym, bars in bars_map.items():
                    if sym in out and isinstance(bars, list):
                        out[sym].extend(bars)

                page_token = data.get("next_page_token")
                break
            except Exception as e:
                last_err = e
                sleep_s = min(60, 2 ** attempt)
                time.sleep(sleep_s)
        else:
            raise RuntimeError(f"请求失败（symbols chunk={len(symbols)}）。最后错误：{last_err}")

        if not page_token:
            break

    return out


def bars_to_finrl_rows(symbol: str, bars: List[dict]) -> List[dict]:
    """
    把 Alpaca bar dict 转成 FinRL 风格行：
    date,tic,open,high,low,close,volume
    """
    rows = []
    for b in bars:
        # Alpaca 返回 t 是 RFC3339 时间戳（UTC）
        t = b.get("t")
        if not t:
            continue
        # 只取日期部分（UTC 日期对日线没问题；我们后续也用这个日期做交易日历）
        date = str(t)[:10]
        rows.append(
            {
                "date": date,
                "tic": symbol,
                "open": b.get("o"),
                "high": b.get("h"),
                "low": b.get("l"),
                "close": b.get("c"),
                "volume": b.get("v"),
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols-file", required=True, help="txt，每行一个 symbol")
    ap.add_argument("--out-dir", required=True, help="输出目录")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD（含）")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD（含），默认今天")
    ap.add_argument("--timeframe", default="1Day")
    ap.add_argument("--adjustment", default="all", choices=["raw", "split", "dividend", "all"])
    ap.add_argument("--chunk-size", type=int, default=200, help="每次请求的 symbols 数（建议 100~200）")
    ap.add_argument("--per-symbol", action="store_true", help="同时输出每只股票单独 CSV")
    ap.add_argument("--make-calendar", action="store_true", help="输出交易日历 alpaca_calendar.csv")
    ap.add_argument("--combined-name", default="combined_daily_adjusted.csv", help="合并 CSV 文件名")
    args = ap.parse_args()

    if args.end is None:
        args.end = pd.Timestamp.today().strftime("%Y-%m-%d")

    auth = get_auth()
    symbols = load_symbols(args.symbols_file)
    if not symbols:
        raise RuntimeError("symbols-file 为空")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()

    chunks = chunk_list(symbols, args.chunk_size)
    all_rows: List[dict] = []
    per_symbol_dir = out_dir / "by_symbol"
    if args.per_symbol:
        per_symbol_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] symbols={len(symbols)} chunks={len(chunks)} feed={auth.feed} adjustment={args.adjustment}")
    print(f"[INFO] range: {args.start} .. {args.end}")

    for i, syms in enumerate(chunks, 1):
        print(f"[FETCH] chunk {i}/{len(chunks)}  n={len(syms)}")
        bars_map = alpaca_get_bars(
            session=session,
            auth=auth,
            symbols=syms,
            start=args.start,
            end=args.end,
            timeframe=args.timeframe,
            adjustment=args.adjustment,
        )

        # 写 per-symbol & 汇总
        for sym in syms:
            rows = bars_to_finrl_rows(sym, bars_map.get(sym, []))
            if not rows:
                continue
            all_rows.extend(rows)
            if args.per_symbol:
                df_sym = pd.DataFrame(rows).sort_values("date")
                df_sym.to_csv(per_symbol_dir / f"{sym}.csv", index=False, encoding="utf-8")

    if not all_rows:
        raise RuntimeError("没有拿到任何 bars 数据，请检查 feed 权限（iex/sip）、symbol 是否有效、日期范围。")

    df = pd.DataFrame(all_rows)
    # 转数值
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["tic"] = df["tic"].astype(str).str.upper()

    df = df.dropna(subset=["date", "tic", "open", "high", "low", "close", "volume"])
    df = df.sort_values(["tic", "date"]).drop_duplicates(subset=["tic", "date"], keep="last").reset_index(drop=True)

    combined_path = out_dir / args.combined_name
    df.to_csv(combined_path, index=False, encoding="utf-8")
    print(f"[OK] saved combined: {combined_path} rows={len(df)} symbols={df['tic'].nunique()} "
          f"range=[{df['date'].min()}..{df['date'].max()}]")

    if args.make_calendar:
        cal_path = out_dir / "alpaca_calendar.csv"
        cal = pd.DataFrame({"timestamp": sorted(df["date"].unique())})
        cal.to_csv(cal_path, index=False, encoding="utf-8")
        print(f"[OK] saved calendar: {cal_path} rows={len(cal)}")

    print("[DONE]")


if __name__ == "__main__":
    main()
