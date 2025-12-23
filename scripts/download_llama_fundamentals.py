# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
from urllib.parse import quote

import pandas as pd
import requests
from tqdm import tqdm


# =========================================================
# 目标：
# 1) 仅调用 DefiLlama 开放端点：https://api.llama.fi
# 2) 免费限速友好：默认 sleep 更慢 + 429/5xx 指数退避
# 3) 输出 JSONL：每行 4 个 key：date / symbol / type / result
# 4) “最稳用法”：允许你直接复制 --list-chains 输出里的 name 列作为 --chains 输入
#    -> 程序会先拉 /v2/chains，用 name 列做“规范化匹配”，并自动用规范名请求历史 TVL
# =========================================================

LLAMA_OPEN_BASE = "https://api.llama.fi"


# -----------------------------
# 通用：安全 HTTP + 重试
# -----------------------------
@dataclass
class HttpCfg:
    timeout: int = 30
    max_retries: int = 8
    backoff: float = 1.8
    sleep: float = 1.0  # 免费端点建议慢一点（默认 1.0s/req）


class SafeHttp:
    def __init__(self, cfg: HttpCfg):
        self.cfg = cfg
        self.sess = requests.Session()

    def get_json(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        last_err = None
        for i in range(self.cfg.max_retries):
            try:
                if self.cfg.sleep > 0:
                    time.sleep(self.cfg.sleep)

                r = self.sess.get(url, params=params, headers=headers, timeout=self.cfg.timeout)

                # 常见限流/临时不可用：429/502/503/504
                if r.status_code in (429, 502, 503, 504):
                    wait = (self.cfg.backoff ** i)
                    time.sleep(wait)
                    continue

                r.raise_for_status()
                return r.json()

            except Exception as e:
                last_err = e
                wait = (self.cfg.backoff ** i)
                time.sleep(wait)

        raise RuntimeError(f"HTTP failed after retries: {type(last_err).__name__}: {last_err}")


# -----------------------------
# 文件/输出工具
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_jsonl_line(fp: Path, obj: Dict[str, Any]) -> None:
    with fp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_list_arg(x: Optional[str]) -> List[str]:
    """
    逗号分隔列表。会保留元素内部空格（因为 name 里可能有空格，比如 "OP Mainnet"）。
    """
    if not x:
        return []
    # 注意：这里不对内部空格做 split，只按逗号切
    return [i.strip() for i in x.split(",") if i.strip()]


def dump_daily_df_as_jsonl(df: pd.DataFrame, out_fp: Path, symbol: str, typ: str) -> None:
    """
    df 必须包含 date 列（python date 或 str）
    其余列都会写入 result
    """
    for _, r in df.iterrows():
        date = str(r["date"])
        result: Dict[str, Any] = {}
        for k in df.columns:
            if k == "date":
                continue
            v = r[k]
            if pd.isna(v):
                result[k] = None
            elif isinstance(v, (int, float)):
                result[k] = float(v)
            else:
                result[k] = v
        write_jsonl_line(out_fp, {"date": date, "symbol": symbol, "type": typ, "result": result})


def safe_token(s: str) -> str:
    """
    用于文件名/符号名：把空格和特殊字符替换成下划线，避免 Windows 路径问题
    """
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-\.]", "_", s)
    return s


def norm_name(s: str) -> str:
    """
    用于匹配：小写、去首尾空格、内部多空格折叠
    """
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


# -----------------------------
# DefiLlama：开放端点（TVL）
# -----------------------------
def llama_list_chains(http: SafeHttp) -> pd.DataFrame:
    """
    返回 chains 列表
    """
    url = f"{LLAMA_OPEN_BASE}/v2/chains"
    j = http.get_json(url)
    df = pd.DataFrame(j)
    return df


def llama_list_protocols(http: SafeHttp) -> pd.DataFrame:
    """
    返回协议列表（包含 slug/name 等）
    """
    url = f"{LLAMA_OPEN_BASE}/protocols"
    j = http.get_json(url)
    df = pd.DataFrame(j)
    return df


def resolve_chains_by_name(http: SafeHttp, user_chains: List[str]) -> Tuple[List[str], List[str]]:
    """
    最稳匹配逻辑：
    - 拉 /v2/chains
    - 只用返回的 name 列作为“规范名集合”
    - 用户传入 --chains 时可直接粘贴 name 列（大小写不敏感；多空格也可）
    返回：
    - resolved: 规范链名（用于请求 /v2/historicalChainTvl/{chain}）
    - missing: 没找到的输入
    """
    if not user_chains:
        return [], []

    df = llama_list_chains(http)
    if "name" not in df.columns:
        # 极端情况：API 结构变了，直接原样用
        return user_chains, []

    names = df["name"].dropna().astype(str).tolist()
    m = {norm_name(n): n for n in names}  # 规范化 -> 原始规范名

    resolved: List[str] = []
    missing: List[str] = []
    for x in user_chains:
        key = norm_name(x)
        if key in m:
            resolved.append(m[key])
        else:
            missing.append(x)

    # 去重保持顺序
    seen = set()
    out = []
    for n in resolved:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out, missing


def llama_chain_tvl_daily(http: SafeHttp, chain_name: str, start: str, end: str) -> pd.DataFrame:
    """
    链 TVL 日频序列：
    GET /v2/historicalChainTvl/{chain}
    """
    # chain_name 可能包含空格/特殊字符，必须 URL 编码
    chain_path = quote(chain_name, safe="")
    url = f"{LLAMA_OPEN_BASE}/v2/historicalChainTvl/{chain_path}"
    j = http.get_json(url)

    df = pd.DataFrame(j)
    if df.empty:
        return pd.DataFrame(columns=["date", "tvl_usd"])

    # 兼容 date/timestamp 字段
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], unit="s", utc=True).dt.date
    elif "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.date
    else:
        c0 = df.columns[0]
        df["date"] = pd.to_datetime(df[c0], unit="s", utc=True, errors="coerce").dt.date

    # 兼容 tvl 字段
    if "tvl" in df.columns:
        df["tvl_usd"] = pd.to_numeric(df["tvl"], errors="coerce")
    elif "totalLiquidityUSD" in df.columns:
        df["tvl_usd"] = pd.to_numeric(df["totalLiquidityUSD"], errors="coerce")
    else:
        cand = [c for c in df.columns if "tvl" in str(c).lower()]
        df["tvl_usd"] = pd.to_numeric(df[cand[0]], errors="coerce") if cand else pd.NA

    df = df.sort_values("date")[["date", "tvl_usd"]]

    s = pd.Timestamp(start).date()
    e = pd.Timestamp(end).date()
    return df[(df["date"] >= s) & (df["date"] <= e)]


def llama_protocol_tvl_daily(http: SafeHttp, protocol_slug: str, start: str, end: str) -> pd.DataFrame:
    """
    协议 TVL 日频序列：
    GET /protocol/{protocol_slug}
    """
    protocol_path = quote(protocol_slug, safe="")
    url = f"{LLAMA_OPEN_BASE}/protocol/{protocol_path}"
    j = http.get_json(url)

    tvl = j.get("tvl") or []
    df = pd.DataFrame(tvl)
    if df.empty:
        return pd.DataFrame(columns=["date", "tvl_usd"])

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], unit="s", utc=True).dt.date
    elif "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.date

    if "totalLiquidityUSD" in df.columns:
        df["tvl_usd"] = pd.to_numeric(df["totalLiquidityUSD"], errors="coerce")
    elif "tvl" in df.columns:
        df["tvl_usd"] = pd.to_numeric(df["tvl"], errors="coerce")
    else:
        cand = [c for c in df.columns if "liquidity" in str(c).lower() or "tvl" in str(c).lower()]
        df["tvl_usd"] = pd.to_numeric(df[cand[0]], errors="coerce") if cand else pd.NA

    df = df.sort_values("date")[["date", "tvl_usd"]]

    s = pd.Timestamp(start).date()
    e = pd.Timestamp(end).date()
    return df[(df["date"] >= s) & (df["date"] <= e)]


# -----------------------------
# 主流程
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="输出目录（写 jsonl）")
    ap.add_argument("--start", default="2020-12-23")
    ap.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))

    # 注意：这里强调“直接复制 name 列”
    ap.add_argument(
        "--chains",
        default="",
        help='链 TVL：直接粘贴 --list-chains 输出里的 name（逗号分隔），例如 "ethereum,arbitrum,OP Mainnet"',
    )
    ap.add_argument("--protocols", default="", help="协议 TVL：slug 逗号分隔（如 aave,uniswap,lido）")

    ap.add_argument("--sleep", type=float, default=1.0, help="请求间隔（秒），免费端点建议 >=0.8")
    ap.add_argument("--overwrite", action="store_true", help="存在则覆盖（否则 append）")

    ap.add_argument("--list-chains", action="store_true", help="只打印 chains 列表并退出")
    ap.add_argument("--list-protocols", action="store_true", help="只打印 protocols 列表并退出（较大）")
    ap.add_argument("--max-list-rows", type=int, default=50, help="list 模式最多打印多少行")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    http = SafeHttp(HttpCfg(sleep=float(args.sleep)))

    # 列表模式：帮你确认 chain 名称 / protocol slug 是否拼对
    if args.list_chains:
        df = llama_list_chains(http)

        # 优先把 name 列放到最前，方便复制
        cols_front = [c for c in ["name", "gecko_id", "tvl", "tokenSymbol", "cmcId", "chainId"] if c in df.columns]
        cols_rest = [c for c in df.columns if c not in cols_front]
        show_cols = cols_front + cols_rest

        print(df[show_cols].head(int(args.max_list_rows)).to_string(index=False))
        print(f"\n[INFO] total chains = {len(df)}")
        print("[TIP] 直接复制上面 name 列的值，逗号分隔后作为 --chains 输入（大小写/多空格不敏感）。")
        return

    if args.list_protocols:
        df = llama_list_protocols(http)
        cols = [c for c in ["name", "slug", "tvl", "chain", "category"] if c in df.columns]
        if cols:
            print(df[cols].head(int(args.max_list_rows)).to_string(index=False))
        else:
            print(df.head(int(args.max_list_rows)).to_string(index=False))
        print(f"\n[INFO] total protocols = {len(df)}")
        return

    start, end = args.start, args.end

    # 1) 链 TVL：先用 name 列“规范化匹配”
    user_chains = load_list_arg(args.chains)
    resolved_chains, missing = resolve_chains_by_name(http, user_chains)

    if user_chains and missing:
        # 这里不直接报错：只提示，并继续跑已匹配到的
        print("[WARN] 以下 chains 没在 /v2/chains 的 name 列里匹配到，将跳过：")
        for x in missing:
            print(f"  - {x}")
        print("[TIP] 请先运行 --list-chains，然后直接复制 name 列（最稳）。")

    for ch in tqdm(resolved_chains, desc="DefiLlama chain TVL"):
        # symbol/文件名用“安全版”，避免空格/特殊字符导致 Windows 路径问题
        token = safe_token(ch).upper()
        symbol = f"CHAIN_{token}"
        out_fp = out_dir / f"{symbol}.jsonl"

        if args.overwrite and out_fp.exists():
            out_fp.unlink()

        try:
            df = llama_chain_tvl_daily(http, ch, start, end)
            dump_daily_df_as_jsonl(df, out_fp, symbol, "llama_chain_tvl_daily")
        except Exception as e:
            write_jsonl_line(
                out_fp,
                {
                    "date": pd.Timestamp.today().strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "type": "llama_chain_tvl_daily",
                    "result": {"error": "http_error", "message": str(e), "chain": ch},
                },
            )

    # 2) 协议 TVL：slug 直接用（建议你先 list-protocols 查 slug）
    protocols = load_list_arg(args.protocols)
    for p in tqdm(protocols, desc="DefiLlama protocol TVL"):
        token = safe_token(p).upper()
        symbol = f"PROTOCOL_{token}"
        out_fp = out_dir / f"{symbol}.jsonl"

        if args.overwrite and out_fp.exists():
            out_fp.unlink()

        try:
            df = llama_protocol_tvl_daily(http, p, start, end)
            dump_daily_df_as_jsonl(df, out_fp, symbol, "llama_protocol_tvl_daily")
        except Exception as e:
            write_jsonl_line(
                out_fp,
                {
                    "date": pd.Timestamp.today().strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "type": "llama_protocol_tvl_daily",
                    "result": {"error": "http_error", "message": str(e), "protocol": p},
                },
            )

    print(f"[DONE] DefiLlama (open) fundamentals saved -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
