# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import argparse
import datetime as dt
from typing import Any, Dict, Iterable, List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from fmp_client import FMPClient
from symbol_utils import load_symbols_from_txt, to_fmp_symbol


def parse_iso_date(x: Any) -> Optional[dt.date]:
    if not x:
        return None
    try:
        s = str(x).strip()
        if len(s) >= 10 and s[4] == "-" and s[7] == "-":
            return dt.date.fromisoformat(s[:10])
        if len(s) == 8 and s.isdigit():
            return dt.date(int(s[0:4]), int(s[4:6]), int(s[6:8]))
    except Exception:
        return None
    return None


def in_range(d: Optional[dt.date], start: dt.date, end: dt.date) -> bool:
    if d is None:
        return False
    return (d >= start) and (d <= end)


def write_jsonl_line(fp, date_str: str, symbol: str, typ: str, result: Any) -> None:
    obj = {"date": date_str, "symbol": symbol, "type": typ, "result": result}
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def iter_list_payload(payload: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
    elif isinstance(payload, dict):
        for k in ("data", "results", "items"):
            v = payload.get(k)
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        yield item


def fetch_financial_statements_stable(
    c: FMPClient, fmp_symbol: str, start: dt.date, end: dt.date, out_fp, orig_symbol: str
) -> None:
    """
    Stable endpoints:
      - /stable/income-statement?symbol=...&period=annual|quarter&limit=...
      - /stable/balance-sheet-statement?symbol=...&period=annual|quarter&limit=...
      - /stable/cash-flow-statement?symbol=...&period=annual|quarter&limit=...
    """
    base = "https://financialmodelingprep.com/stable"

    endpoints = [
        ("income_statement_annual", f"{base}/income-statement", {"symbol": fmp_symbol, "period": "annual", "limit": 200}),
        ("income_statement_quarter", f"{base}/income-statement", {"symbol": fmp_symbol, "period": "quarter", "limit": 400}),
        ("balance_sheet_annual", f"{base}/balance-sheet-statement", {"symbol": fmp_symbol, "period": "annual", "limit": 200}),
        ("balance_sheet_quarter", f"{base}/balance-sheet-statement", {"symbol": fmp_symbol, "period": "quarter", "limit": 400}),
        ("cash_flow_annual", f"{base}/cash-flow-statement", {"symbol": fmp_symbol, "period": "annual", "limit": 200}),
        ("cash_flow_quarter", f"{base}/cash-flow-statement", {"symbol": fmp_symbol, "period": "quarter", "limit": 400}),
    ]

    for typ, url, params in endpoints:
        data, err = c.get_json(url, params=params)
        if err is not None:
            write_jsonl_line(out_fp, c.today_iso(), orig_symbol, typ, err)
            continue

        wrote_any = False
        for item in iter_list_payload(data):
            d = parse_iso_date(item.get("date"))
            if in_range(d, start, end):
                write_jsonl_line(out_fp, item.get("date", c.today_iso())[:10], orig_symbol, typ, item)
                wrote_any = True

        if not wrote_any:
            # Helpful note for debugging, but still "success" shape-wise.
            if isinstance(data, list):
                write_jsonl_line(out_fp, c.today_iso(), orig_symbol, typ, {"note": "no_records_in_range", "count": len(data)})
            else:
                write_jsonl_line(out_fp, c.today_iso(), orig_symbol, typ, {"note": "no_records_in_range_or_unexpected_shape"})


# ----------------------------
# NEW: ratios & key-metrics history (quarter/annual)
# ----------------------------
def fetch_ratios_and_key_metrics_history(
    c: FMPClient,
    fmp_symbol: str,
    start: dt.date,
    end: dt.date,
    out_fp,
    orig_symbol: str,
) -> None:
    """
    Stable endpoints:
      - /stable/ratios?symbol=AAPL         (supports period=annual|quarter, limit=...)
      - /stable/key-metrics?symbol=AAPL    (supports period=annual|quarter, limit=...)
    Docs:
      Ratios endpoint: https://financialmodelingprep.com/stable/ratios?symbol=AAPL :contentReference[oaicite:2]{index=2}
      Key-metrics endpoint: https://financialmodelingprep.com/stable/key-metrics?symbol=AAPL :contentReference[oaicite:3]{index=3}
    """
    base = "https://financialmodelingprep.com/stable"
    jobs = [
        ("ratios_quarter", f"{base}/ratios", {"symbol": fmp_symbol, "period": "quarter", "limit": 400}),
        ("ratios_annual", f"{base}/ratios", {"symbol": fmp_symbol, "period": "annual", "limit": 200}),
        ("key_metrics_quarter", f"{base}/key-metrics", {"symbol": fmp_symbol, "period": "quarter", "limit": 400}),
        ("key_metrics_annual", f"{base}/key-metrics", {"symbol": fmp_symbol, "period": "annual", "limit": 200}),
    ]

    for typ, url, params in jobs:
        data, err = c.get_json(url, params=params)
        if err is not None:
            write_jsonl_line(out_fp, c.today_iso(), orig_symbol, typ, err)
            continue

        wrote_any = False
        # Most of these are list-of-dict with a "date" field
        for item in iter_list_payload(data):
            d = parse_iso_date(item.get("date"))
            if in_range(d, start, end):
                write_jsonl_line(out_fp, item.get("date", c.today_iso())[:10], orig_symbol, typ, item)
                wrote_any = True

        if not wrote_any:
            if isinstance(data, list):
                write_jsonl_line(out_fp, c.today_iso(), orig_symbol, typ, {"note": "no_records_in_range", "count": len(data)})
            else:
                write_jsonl_line(out_fp, c.today_iso(), orig_symbol, typ, {"note": "no_records_in_range_or_unexpected_shape"})


def fetch_other_fundamentals(c: FMPClient, fmp_symbol: str, out_fp, orig_symbol: str) -> None:
    """
    Keep existing lightweight endpoints (TTM snapshots, EV series, etc.).
    """
    endpoints = [
        ("ratios_ttm", "https://financialmodelingprep.com/stable/ratios-ttm", {"symbol": fmp_symbol}),
        ("key_metrics_ttm", "https://financialmodelingprep.com/stable/key-metrics-ttm", {"symbol": fmp_symbol}),
        # enterprise-values is often a historical series; keep it for valuation features
        ("enterprise_values", "https://financialmodelingprep.com/stable/enterprise-values", {"symbol": fmp_symbol, "period": "quarter", "limit": 400}),
    ]

    for typ, url, params in endpoints:
        data, err = c.get_json(url, params=params)
        if err is not None:
            write_jsonl_line(out_fp, c.today_iso(), orig_symbol, typ, err)
            continue

        if isinstance(data, list) and data and isinstance(data[0], dict) and ("date" in data[0]):
            for item in data:
                date_str = item.get("date", c.today_iso())[:10]
                write_jsonl_line(out_fp, date_str, orig_symbol, typ, item)
        else:
            write_jsonl_line(out_fp, c.today_iso(), orig_symbol, typ, data)


def fetch_analyst_estimates(
    c: FMPClient, fmp_symbol: str, out_fp, orig_symbol: str, max_pages: int = 20, limit: int = 100
) -> None:
    url = "https://financialmodelingprep.com/stable/analyst-estimates"
    for page in range(max_pages):
        data, err = c.get_json(url, params={"symbol": fmp_symbol, "period": "annual", "page": page, "limit": limit})
        typ = "analyst_estimates_annual"
        if err is not None:
            write_jsonl_line(out_fp, c.today_iso(), orig_symbol, typ, err)
            return

        if not data:
            return

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    date_str = (item.get("date") or item.get("fillingDate") or item.get("period") or c.today_iso())
                    write_jsonl_line(out_fp, str(date_str)[:10], orig_symbol, typ, item)
        else:
            write_jsonl_line(out_fp, c.today_iso(), orig_symbol, typ, data)
            return


def fetch_sec_filings(
    c: FMPClient,
    fmp_symbol: str,
    start: dt.date,
    end: dt.date,
    out_fp,
    orig_symbol: str,
    forms: Optional[List[str]] = None,
    max_pages: int = 200,
    limit: int = 100,
) -> None:
    url = "https://financialmodelingprep.com/stable/sec-filings-search/symbol"
    forms_set = set([x.upper() for x in (forms or [])])

    for page in range(max_pages):
        data, err = c.get_json(
            url,
            params={
                "symbol": fmp_symbol,
                "from": start.isoformat(),
                "to": end.isoformat(),
                "page": page,
                "limit": limit,
            },
        )
        typ = "sec_filings"
        if err is not None:
            write_jsonl_line(out_fp, c.today_iso(), orig_symbol, typ, err)
            return

        if not data:
            return

        if isinstance(data, list):
            wrote = 0
            for item in data:
                if not isinstance(item, dict):
                    continue
                form = str(item.get("type", "") or item.get("formType", "")).upper()
                if forms_set and form not in forms_set:
                    continue

                d = parse_iso_date(item.get("filingDate") or item.get("acceptedDate") or item.get("date"))
                if in_range(d, start, end):
                    date_str = (item.get("filingDate") or item.get("acceptedDate") or item.get("date") or c.today_iso())
                    write_jsonl_line(out_fp, str(date_str)[:10], orig_symbol, typ, item)
                    wrote += 1

            if wrote == 0:
                continue


def fetch_press_releases_and_news(
    c: FMPClient,
    fmp_symbol: str,
    start: dt.date,
    end: dt.date,
    out_fp,
    orig_symbol: str,
    max_pages: int = 200,
    limit: int = 100,
) -> None:
    # press releases
    press_url = "https://financialmodelingprep.com/stable/news/press-releases"
    typ = "press_releases"

    for page in range(max_pages):
        params = {"symbols": fmp_symbol, "page": page, "limit": limit}
        data, err = c.get_json(press_url, params=params)
        if err is not None:
            if page == 0:
                data2, err2 = c.get_json(press_url, params={"symbols": fmp_symbol})
                if err2 is None:
                    data = data2
                else:
                    write_jsonl_line(out_fp, c.today_iso(), orig_symbol, typ, err2)
                    break
            else:
                break

        if not data:
            break

        items = list(iter_list_payload(data))
        if not items:
            break

        oldest: Optional[dt.date] = None
        wrote = 0
        for item in items:
            d = parse_iso_date(item.get("date"))
            if d:
                oldest = d if (oldest is None or d < oldest) else oldest
            if in_range(d, start, end):
                write_jsonl_line(out_fp, item.get("date", c.today_iso())[:10], orig_symbol, typ, item)
                wrote += 1

        if oldest is not None and oldest < start:
            break
        if wrote == 0 and oldest is None:
            break

    # stock news (stable)
    news_url = "https://financialmodelingprep.com/stable/news/stock"
    typ = "stock_news"

    for page in range(max_pages):
        params = {"symbols": fmp_symbol, "page": page, "limit": limit}
        data, err = c.get_json(news_url, params=params)
        if err is not None:
            if page == 0:
                data2, err2 = c.get_json(news_url, params={"symbols": fmp_symbol})
                if err2 is None:
                    data = data2
                else:
                    write_jsonl_line(out_fp, c.today_iso(), orig_symbol, typ, err2)
                    break
            else:
                break

        if not data:
            break

        items = list(iter_list_payload(data))
        if not items:
            break

        oldest: Optional[dt.date] = None
        wrote = 0
        for item in items:
            date_str = item.get("publishedDate") or item.get("date") or item.get("publishedAt")
            d = parse_iso_date(date_str)
            if d:
                oldest = d if (oldest is None or d < oldest) else oldest
            if in_range(d, start, end):
                write_jsonl_line(out_fp, str(date_str)[:10], orig_symbol, typ, item)
                wrote += 1

        if oldest is not None and oldest < start:
            break
        if wrote == 0 and oldest is None:
            break


def fetch_10k_10q_json(
    c: FMPClient,
    fmp_symbol: str,
    start: dt.date,
    end: dt.date,
    out_fp,
    orig_symbol: str,
) -> None:
    url = "https://financialmodelingprep.com/stable/financial-reports-json"
    years = list(range(start.year, end.year + 1))
    periods = ["FY", "Q1", "Q2", "Q3", "Q4"]

    for y in years:
        for p in periods:
            data, err = c.get_json(url, params={"symbol": fmp_symbol, "year": y, "period": p})
            typ = f"sec_report_json_{p}"
            if err is not None:
                if p == "FY":
                    write_jsonl_line(out_fp, c.today_iso(), orig_symbol, typ, err)
                continue
            if not data:
                continue

            if isinstance(data, dict):
                date_str = data.get("filingDate") or data.get("acceptedDate") or c.today_iso()
                d = parse_iso_date(date_str)
                if in_range(d, start, end) or p == "FY":
                    write_jsonl_line(out_fp, str(date_str)[:10], orig_symbol, typ, data)
            else:
                write_jsonl_line(out_fp, c.today_iso(), orig_symbol, typ, data)


def process_one_symbol(
    symbol: str,
    out_dir: Path,
    start: dt.date,
    end: dt.date,
    max_pages: int,
    fetch_news: bool,
    fetch_10k_json: bool,
    overwrite: bool,
) -> None:
    c = FMPClient()
    fmp_symbol = to_fmp_symbol(symbol)
    out_path = out_dir / f"{symbol}.jsonl"

    if out_path.exists() and not overwrite:
        return

    with out_path.open("w", encoding="utf-8") as fp:
        # 1) Core statements
        fetch_financial_statements_stable(c, fmp_symbol, start, end, fp, symbol)

        # 2) NEW: ratios + key-metrics history (quarter/annual)
        fetch_ratios_and_key_metrics_history(c, fmp_symbol, start, end, fp, symbol)

        # 3) Other fundamentals (TTM, EV series)
        fetch_other_fundamentals(c, fmp_symbol, fp, symbol)

        # 4) Analyst estimates
        fetch_analyst_estimates(c, fmp_symbol, fp, symbol, max_pages=min(20, max_pages), limit=100)

        # 5) SEC filings
        fetch_sec_filings(
            c, fmp_symbol, start, end, fp, symbol,
            forms=None,  # e.g. ["10-K","10-Q","8-K"]
            max_pages=max_pages, limit=100
        )

        # 6) News
        if fetch_news:
            fetch_press_releases_and_news(c, fmp_symbol, start, end, fp, symbol, max_pages=max_pages, limit=100)

        # 7) 10-K/10-Q JSON (big)
        if fetch_10k_json:
            fetch_10k_10q_json(c, fmp_symbol, start, end, fp, symbol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols-file", type=str, required=True, help="txt with one symbol per line")
    ap.add_argument("--out-dir", type=str, default="out/fmp_jsonl", help="output folder")
    ap.add_argument("--start", type=str, default="2021-01-01")
    ap.add_argument("--end", type=str, default=dt.date.today().isoformat())
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-pages", type=int, default=120, help="pagination limit for filings/news")
    ap.add_argument("--fetch-news", action="store_true", help="also fetch press releases + stock news")
    ap.add_argument("--fetch-10k-json", action="store_true", help="also fetch financial-reports-json (can be large)")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing jsonl files")
    args = ap.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)

    symbols = load_symbols_from_txt(args.symbols_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "date_generated": dt.date.today().isoformat(),
        "source": "Financial Modeling Prep (FMP)",
        "range": {"start": start.isoformat(), "end": end.isoformat()},
        "symbols_file": os.path.abspath(args.symbols_file),
        "includes": [
            "income_statement_(annual/quarter)",
            "balance_sheet_(annual/quarter)",
            "cash_flow_(annual/quarter)",
            "ratios_(annual/quarter)",          # NEW
            "key_metrics_(annual/quarter)",     # NEW
            "ratios_ttm",
            "key_metrics_ttm",
            "enterprise_values",
            "analyst_estimates_annual",
            "sec_filings",
            "press_releases (optional)",
            "stock_news (optional)",
            "financial-reports-json FY/Q1-Q4 (optional)",
        ],
        "security_note": "This downloader redacts apikey in any logged error/url/text written to jsonl.",
    }
    (out_dir / "_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(
                process_one_symbol,
                s,
                out_dir,
                start,
                end,
                args.max_pages,
                args.fetch_news,
                args.fetch_10k_json,
                args.overwrite,
            )
            for s in symbols
        ]

        for _ in tqdm(as_completed(futs), total=len(futs), desc="Downloading"):
            pass

    print(f"Done. Output folder: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
