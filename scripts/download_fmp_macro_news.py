# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


BASE = "https://financialmodelingprep.com/stable"


def _today_utc_date() -> dt.date:
    return dt.datetime.utcnow().date()


def _parse_date_any(x: Any) -> Optional[dt.date]:
    """
    FMP news 常见字段：publishedDate / date
    可能带时间：2025-12-22 12:34:56 或 ISO
    """
    if not x:
        return None
    s = str(x).strip()
    if not s:
        return None
    s10 = s[:10]
    try:
        return dt.date.fromisoformat(s10)
    except Exception:
        return None


def _pick_published_date(item: Dict[str, Any]) -> Optional[dt.date]:
    for k in ("publishedDate", "date", "published_date", "published_at", "datetime"):
        if k in item:
            d = _parse_date_any(item.get(k))
            if d:
                return d
    return None


def _safe_err(resp: requests.Response) -> Dict[str, Any]:
    # 不回传包含 apikey 的 URL，也不输出完整 body（避免意外泄露）
    text = ""
    try:
        text = resp.text[:300]
    except Exception:
        text = ""
    return {
        "error": "http_error",
        "status_code": resp.status_code,
        "text_head": text,
    }


def _request_json(
    session: requests.Session,
    url: str,
    params: Dict[str, Any],
    timeout: int = 30,
    max_retries: int = 6,
    sleep_base: float = 0.6,
) -> Tuple[bool, Any]:
    """
    返回 (ok, payload_or_error)
    - 自动处理 429/5xx 重试
    """
    for i in range(max_retries):
        try:
            resp = session.get(url, params=params, timeout=timeout)
        except Exception as e:
            if i == max_retries - 1:
                return False, {"error": "request_exception", "message": str(e)[:300]}
            time.sleep(sleep_base * (2 ** i))
            continue

        if resp.status_code == 200:
            try:
                return True, resp.json()
            except Exception as e:
                return False, {"error": "json_decode_error", "message": str(e)[:300]}
        elif resp.status_code in (429, 500, 502, 503, 504):
            # 退避重试
            if i == max_retries - 1:
                return False, _safe_err(resp)
            time.sleep(sleep_base * (2 ** i))
            continue
        else:
            return False, _safe_err(resp)

    return False, {"error": "unknown"}


def fetch_latest_paged_until_cutoff(
    session: requests.Session,
    endpoint: str,
    apikey: str,
    cutoff: dt.date,
    limit: int,
    max_pages: int,
    polite_sleep: float,
) -> List[Dict[str, Any]]:
    """
    用 latest 分页接口，从 page=0 往后翻，直到最老一条 < cutoff 或返回空。
    注意：这些 “latest” 接口一般按时间倒序返回。
    """
    url = f"{BASE}{endpoint}"
    out: List[Dict[str, Any]] = []

    for page in range(max_pages):
        ok, payload = _request_json(
            session,
            url=url,
            params={"page": page, "limit": limit, "apikey": apikey},
        )
        if not ok:
            # 这里不中断全局，给个标记后退出该类型抓取
            # （你也可以改成 continue，跳过失败页）
            print(f"[WARN] {endpoint} page={page} 请求失败：{payload.get('status_code', '')} {payload.get('error', '')}")
            break

        if not payload:
            break
        if not isinstance(payload, list):
            # 有些端点可能返回 dict
            break

        # 过滤 cutoff
        page_items_kept = 0
        oldest_in_page: Optional[dt.date] = None

        for item in payload:
            if not isinstance(item, dict):
                continue
            d = _pick_published_date(item)
            if d:
                if (oldest_in_page is None) or (d < oldest_in_page):
                    oldest_in_page = d
                if d < cutoff:
                    continue
                out.append(item)
                page_items_kept += 1
            else:
                # 没日期：保守起见也收下（但不会用于停条件）
                out.append(item)
                page_items_kept += 1

        # 如果这一页最老的已经早于 cutoff，可以停
        if oldest_in_page is not None and oldest_in_page < cutoff:
            break

        if polite_sleep > 0:
            time.sleep(polite_sleep)

    return out


def write_jsonl(path: Path, typ: str, items: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            d = _pick_published_date(it)
            outer_date = d.isoformat() if d else _today_utc_date().isoformat()
            obj = {
                "date": outer_date,
                "symbol": "__MACRO__",
                "type": typ,
                "result": it,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="输出目录（会生成 macro_news_*.jsonl）")
    ap.add_argument("--days", type=int, default=90, help="最近 N 天（默认 90 天≈3个月）")
    ap.add_argument("--limit", type=int, default=200, help="每页条数（端点通常支持 limit）")
    ap.add_argument("--max-pages", type=int, default=200, help="最大翻页数（防止无限翻）")
    ap.add_argument(
        "--types",
        default="general,forex,crypto,fmp_articles",
        help="要抓取的类型：general,forex,crypto,fmp_articles（逗号分隔）",
    )
    ap.add_argument("--sleep", type=float, default=0.15, help="每次翻页之间的sleep，避免触发频控")
    ap.add_argument("--api-key-env", default="FMP_API_KEY", help="API key 的环境变量名")
    args = ap.parse_args()

    apikey = os.getenv(args.api_key_env, "").strip()
    if not apikey:
        raise RuntimeError(f"未检测到环境变量 {args.api_key_env}，请先在当前终端临时设置。")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cutoff = _today_utc_date() - dt.timedelta(days=int(args.days))

    # stable endpoints（来自 FMP 文档）
    # General News: /stable/news/general-latest :contentReference[oaicite:4]{index=4}
    # Forex News:   /stable/news/forex-latest  :contentReference[oaicite:5]{index=5}
    # Crypto News:  /stable/news/crypto-latest :contentReference[oaicite:6]{index=6}
    # FMP Articles: /stable/fmp-articles       :contentReference[oaicite:7]{index=7}
    mapping = {
        "general": ("/news/general-latest", "general_news"),
        "forex": ("/news/forex-latest", "forex_news"),
        "crypto": ("/news/crypto-latest", "crypto_news"),
        "fmp_articles": ("/fmp-articles", "fmp_articles"),
    }

    want = [x.strip() for x in args.types.split(",") if x.strip()]
    bad = [x for x in want if x not in mapping]
    if bad:
        raise ValueError(f"未知 types: {bad}，可选：{list(mapping.keys())}")

    session = requests.Session()
    session.headers.update({"User-Agent": "FinML-MacroNews/1.0"})

    total = 0
    for k in want:
        endpoint, typ_name = mapping[k]
        print(f"[FETCH] {typ_name} from {endpoint} (cutoff >= {cutoff.isoformat()})")

        items = fetch_latest_paged_until_cutoff(
            session=session,
            endpoint=endpoint,
            apikey=apikey,
            cutoff=cutoff,
            limit=int(args.limit),
            max_pages=int(args.max_pages),
            polite_sleep=float(args.sleep),
        )

        out_path = out_dir / f"macro_news_{typ_name}.jsonl"
        n = write_jsonl(out_path, typ_name, items)
        total += n
        print(f"[OK] {typ_name}: {n} lines -> {out_path}")

    print(f"[DONE] total_lines={total} out_dir={out_dir.resolve()}")


if __name__ == "__main__":
    main()
