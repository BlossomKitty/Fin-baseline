# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

# 把项目根目录加入 Python 搜索路径，避免找不到 features 包
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from typing import Any, Dict, List

import pandas as pd

from features.jsonl_loader import load_symbol_jsonl
from features.align import build_calendar
from features.compute import FeatureSpec, EventSpec, build_event_rows, rows_to_daily_df


def load_config(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_specs(cfg: Dict[str, Any]) -> tuple[List[FeatureSpec], List[EventSpec]]:
    feature_specs: List[FeatureSpec] = []
    event_specs: List[EventSpec] = []
    for item in cfg.get("features", []):
        typ = item["type"]
        if "fields" in item:
            feature_specs.append(
                FeatureSpec(
                    typ=typ,
                    prefix=item.get("prefix", f"{typ}_"),
                    fields=item["fields"],
                )
            )
        if "events" in item:
            for ev in item["events"]:
                event_specs.append(
                    EventSpec(
                        typ=typ,
                        name=ev["name"],
                        mode=ev["mode"],
                        field=ev.get("field"),
                        value=ev.get("value"),
                    )
                )
    return feature_specs, event_specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl-dir", required=True, help="每只股票一个 *.jsonl 的文件夹")
    ap.add_argument("--out-dir", default="out/features_daily", help="输出文件夹")
    ap.add_argument("--config", default="configs/features_v1.json", help="特征配置文件")
    ap.add_argument("--start", default="2021-01-01", help="起始日期（含）")
    ap.add_argument("--end", default=None, help="结束日期（含），默认取今天")
    ap.add_argument(
        "--calendar-bars-csv",
        default=None,
        help="可选：用 bars/交易日历 CSV 构造真实交易日历（你现在用的是 alpaca_calendar.csv）",
    )
    ap.add_argument("--combine", action="store_true", help="是否额外输出合并后的 _panel.parquet")
    ap.add_argument("--symbols-file", default=None, help="可选：只处理指定 symbols（txt 每行一个）")
    ap.add_argument("--strict-calendar", action="store_true", help="启用后：若发现日期不在日历内则直接报错")
    args = ap.parse_args()

    if args.end is None:
        args.end = pd.Timestamp.today().strftime("%Y-%m-%d")

    cfg = load_config(args.config)
    feature_specs, event_specs = parse_specs(cfg)

    cal_cfg = cfg.get("calendar", {})
    avail_cfg = cfg.get("availability", {})
    date_field_pref = avail_cfg.get("date_field_preference", ["filingDate", "publishedDate", "date"])
    lag_by_type = avail_cfg.get("lag_days_by_type", {})
    default_lag = int(avail_cfg.get("default_lag_days", 1))

    # === 构造交易日历（关键：from_bars_csv 如果是纯日期 CSV，不会再出现“偏一天/周末”）===
    trading_idx = build_calendar(
        start=args.start,
        end=args.end,
        method=cal_cfg.get("method", "bday"),
        from_bars_csv=(args.calendar_bars_csv or cal_cfg.get("from_bars_csv")),
    )

    # 可选：只保留 start..end 范围（build_calendar 已经做了，但这里再保险）
    trading_idx = trading_idx[(trading_idx >= pd.Timestamp(args.start)) & (trading_idx <= pd.Timestamp(args.end))]

    jsonl_dir = Path(args.jsonl_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读取 symbols 白名单（可选）
    allow_symbols = None
    if args.symbols_file:
        allow_symbols = set(
            s.strip().upper()
            for s in Path(args.symbols_file).read_text(encoding="utf-8").splitlines()
            if s.strip()
        )

    files = sorted(jsonl_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"在该目录下未找到任何 .jsonl 文件: {jsonl_dir}")

    all_panels = []

    for fp in files:
        symbol = fp.stem.upper()
        if allow_symbols is not None and symbol not in allow_symbols:
            continue

        groups = load_symbol_jsonl(fp)

        numeric_rows = []
        event_rows = []

        for typ, records in groups.items():
            # typ 可能是 income_statement_annual / ratios_quarter / stock_news 等
            lag = int(lag_by_type.get(typ, default_lag))

            nrows, erows = build_event_rows(
                symbol=symbol,
                typ=typ,
                records=records,
                trading_idx=trading_idx,
                date_field_preference=date_field_pref,
                lag_days=lag,
                feature_specs=feature_specs,
                event_specs=event_specs,
            )
            numeric_rows.extend(nrows)
            event_rows.extend(erows)

        df = rows_to_daily_df(trading_idx, numeric_rows, event_rows)
        df.insert(0, "symbol", symbol)

        # 严格模式：检查 index 是否都是交易日历子集（用于防止“日期偏移”回归）
        if args.strict_calendar:
            cal_set = set(trading_idx.strftime("%Y-%m-%d"))
            idx_set = set(pd.to_datetime(df.index).strftime("%Y-%m-%d"))
            extra = idx_set - cal_set
            if extra:
                raise ValueError(f"[strict-calendar] {symbol} 有 {len(extra)} 个日期不在交易日历中，示例: {sorted(list(extra))[:5]}")

        out_path = out_dir / f"{symbol}.parquet"
        df.to_parquet(out_path, index=True)

        if args.combine:
            tmp = df.reset_index().rename(columns={"index": "date"})
            all_panels.append(tmp)

    if args.combine and all_panels:
        panel = pd.concat(all_panels, ignore_index=True)
        panel_path = out_dir / "_panel.parquet"
        panel.to_parquet(panel_path, index=False)

    print(f"Done. Features saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
