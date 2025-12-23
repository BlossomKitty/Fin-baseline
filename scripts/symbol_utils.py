# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import List

_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9\.\-]{0,14}$")  # max 15 chars


def load_symbols_from_txt(path: str) -> List[str]:
    """
    Read a txt: one symbol per line.
    Deduplicate, strip, ignore placeholders like "..." and empty lines.
    """
    symbols: List[str] = []
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip().upper()
            if not s:
                continue
            if s in {"...", "…"}:
                continue

            parts = [p.strip() for p in s.split(",") if p.strip()]
            for p in parts:
                if p in {"...", "…"}:
                    continue
                if not _TICKER_RE.match(p):
                    continue
                if p not in seen:
                    seen.add(p)
                    symbols.append(p)
    return symbols


def to_fmp_symbol(symbol: str) -> str:
    """
    FMP often uses '-' for tickers like BRK.B -> BRK-B
    """
    return symbol.replace(".", "-")
