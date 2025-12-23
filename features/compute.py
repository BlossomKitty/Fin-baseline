# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .align import apply_lag, get_available_date, snap_to_calendar_next


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))


@dataclass
class FeatureSpec:
    typ: str
    prefix: str
    fields: List[str]


@dataclass
class EventSpec:
    typ: str
    name: str
    mode: str
    field: str | None = None
    value: str | None = None


def build_event_rows(
    symbol: str,
    typ: str,
    records: List[Dict[str, Any]],
    trading_idx: pd.DatetimeIndex,
    date_field_preference: List[str],
    lag_days: int,
    feature_specs: List[FeatureSpec],
    event_specs: List[EventSpec],
) -> Tuple[List[Tuple[pd.Timestamp, str, float]], List[Tuple[pd.Timestamp, str, float]]]:
    """
    returns:
      numeric_rows: [(ts, feature_name, value)]
      event_rows: [(ts, event_name, value)]  value is count (integer)
    """
    numeric_rows: List[Tuple[pd.Timestamp, str, float]] = []
    event_rows: List[Tuple[pd.Timestamp, str, float]] = []

    # numeric features
    for spec in feature_specs:
        if spec.typ != typ:
            continue
        for rec in records:
            if not isinstance(rec, dict):
                continue
            base_d = get_available_date(rec, date_field_preference)
            if not base_d:
                continue
            avail = apply_lag(base_d, lag_days)
            ts = snap_to_calendar_next(trading_idx, avail)

            for f in spec.fields:
                v = rec.get(f)
                if _is_number(v):
                    numeric_rows.append((ts, f"{spec.prefix}{f}", float(v)))

    # events
    for es in event_specs:
        if es.typ != typ:
            continue
        for rec in records:
            if not isinstance(rec, dict):
                continue
            base_d = get_available_date(rec, date_field_preference)
            if not base_d:
                continue
            avail = apply_lag(base_d, lag_days)
            ts = snap_to_calendar_next(trading_idx, avail)

            if es.mode == "count_all":
                event_rows.append((ts, es.name, 1.0))
            elif es.mode == "count_by_field_equals":
                val = rec.get(es.field) if es.field else None
                if val is not None and str(val).upper() == str(es.value).upper():
                    event_rows.append((ts, es.name, 1.0))

    return numeric_rows, event_rows


def rows_to_daily_df(
    trading_idx: pd.DatetimeIndex,
    numeric_rows: List[Tuple[pd.Timestamp, str, float]],
    event_rows: List[Tuple[pd.Timestamp, str, float]],
) -> pd.DataFrame:
    """
    numeric features: last observation then forward fill
    event features: daily sum (no ffill)
    """
    df = pd.DataFrame(index=trading_idx)

    if numeric_rows:
        ndf = pd.DataFrame(numeric_rows, columns=["ts", "feature", "value"])
        # last value per day-feature
        ndf = ndf.sort_values(["ts"])
        pivot = ndf.pivot_table(index="ts", columns="feature", values="value", aggfunc="last")
        df = df.join(pivot, how="left")
        df = df.ffill()

    if event_rows:
        edf = pd.DataFrame(event_rows, columns=["ts", "feature", "value"])
        pivot_e = edf.pivot_table(index="ts", columns="feature", values="value", aggfunc="sum")
        df = df.join(pivot_e, how="left")
        # events default 0
        for c in pivot_e.columns:
            df[c] = df[c].fillna(0.0)

    return df
