# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def load_symbol_jsonl(jsonl_path: str | Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns: {type: [record, ...]}
    record is the *inner* result dict, plus a few top-level fields if needed.
    """
    jsonl_path = Path(jsonl_path)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            typ = obj.get("type")
            result = obj.get("result")

            # keep some outer fields if you want later debugging
            if isinstance(result, dict):
                result["_outer_date"] = obj.get("date")
                result["_outer_symbol"] = obj.get("symbol")
                result["_outer_type"] = typ

            groups[typ].append(result)

    return groups
