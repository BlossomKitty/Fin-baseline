# -*- coding: utf-8 -*-
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def _to_date(x: Any) -> Optional[dt.date]:
    """把各种输入尽量转成 date（只取前 10 位 YYYY-MM-DD）。"""
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        s10 = s[:10]
        return dt.date.fromisoformat(s10)
    except Exception:
        return None


def _is_pure_yyyy_mm_dd(series: pd.Series) -> bool:
    """
    判断一列字符串是否都是纯日期 YYYY-MM-DD。
    注意：只要存在非纯日期（如带时间戳、带时区、epoch），就返回 False。
    """
    s = series.dropna().astype(str).str.strip()
    if len(s) == 0:
        return False
    pat = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    return bool(s.map(lambda x: pat.fullmatch(x) is not None).all())


def build_calendar(start: str, end: str, method: str = "bday", from_bars_csv: str | None = None) -> pd.DatetimeIndex:
    """
    method=bday: pandas business days (baseline 近似).
    如果提供 from_bars_csv，则使用 bars/日历 CSV 里的日期作为“真实交易日历”。

    ⚠️ 关键：如果 CSV 里的 timestamp 是纯日期（YYYY-MM-DD），不要做 utc/tz_convert，
    否则会整体偏一天（你之前出现周日、index not in calendar 就是这个原因）。
    """
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)

    if from_bars_csv:
        p = Path(from_bars_csv)
        df = pd.read_csv(p)

        # 自动猜测时间列名
        for cand in ["timestamp", "date", "t", "time", "datetime"]:
            if cand in df.columns:
                tcol = cand
                break
        else:
            raise ValueError(f"from_bars_csv 缺少时间列，当前列为：{df.columns.tolist()}")

        col = df[tcol]

        # 情况 1：纯日期 YYYY-MM-DD（例如你生成的 alpaca_calendar.csv）
        if _is_pure_yyyy_mm_dd(col):
            ts = pd.to_datetime(col, errors="coerce")  # 不要 utc=True
            days = ts.dt.normalize().dropna().unique()
            idx = pd.DatetimeIndex(sorted(days))  # 已经是“日期”，无需 tz_localize(None)
            return idx[(idx >= s) & (idx <= e)]

        # 情况 2：真正的时间戳（带时区/ISO/epoch 等）——才做时区转换
        ts = pd.to_datetime(col, utc=True, errors="coerce")
        days = ts.dt.tz_convert("America/New_York").dt.normalize().dropna().unique()
        idx = pd.DatetimeIndex(sorted(days)).tz_localize(None)
        return idx[(idx >= s) & (idx <= e)]

    # fallback: 近似交易日历
    if method == "bday":
        return pd.bdate_range(s, e, freq="B")
    return pd.date_range(s, e, freq="D")


def get_available_date(record: Dict[str, Any], date_field_preference: list[str]) -> Optional[dt.date]:
    """在多个候选字段里，找一个能解析出的日期。"""
    for k in date_field_preference:
        if k in record:
            d = _to_date(record.get(k))
            if d:
                return d
    return None


def apply_lag(d: dt.date, lag_days: int) -> dt.date:
    """把事件日期往后推 lag_days（自然日）。"""
    return d + dt.timedelta(days=int(lag_days))


def snap_to_calendar_next(trading_idx: pd.DatetimeIndex, d: dt.date) -> pd.Timestamp:
    """
    把任意日期 d 吸附到“第一个 >= d 的交易日”上。
    """
    t = pd.Timestamp(d)
    pos = trading_idx.searchsorted(t, side="left")
    if pos >= len(trading_idx):
        return trading_idx[-1]
    return trading_idx[pos]
