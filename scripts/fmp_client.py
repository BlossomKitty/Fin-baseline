# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import time
import random
import re
from typing import Any, Dict, Optional, Tuple
import requests


_APIKEY_RE = re.compile(r"(apikey=)([^&\"'\s]+)", re.IGNORECASE)


def redact_apikey_in_text(s: str) -> str:
    if not isinstance(s, str):
        return s
    return _APIKEY_RE.sub(r"\1REDACTED", s)


def redact_apikey_in_obj(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: redact_apikey_in_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_apikey_in_obj(v) for v in obj]
    if isinstance(obj, str):
        return redact_apikey_in_text(obj)
    return obj


class FMPClient:
    """
    FMP HTTP client:
    - API key from env
    - retries + backoff
    - automatic APIKEY redaction for any logged error/url/text
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 5,
        base_sleep: float = 0.25,
        user_agent: str = "fmp-fundamentals-downloader/1.1",
    ):
        self.api_key = api_key or os.getenv("FMP_API_KEY") or os.getenv("FINANCIALMODELINGPREP_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Missing API key. Please set env var FMP_API_KEY (or FINANCIALMODELINGPREP_API_KEY)."
            )
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_sleep = base_sleep
        self.sess = requests.Session()
        self.sess.headers.update({"User-Agent": user_agent})

    def get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        """
        Returns (data, err). If err is not None, it is safe to write to disk (apikey already redacted).
        """
        params = dict(params or {})
        params["apikey"] = self.api_key

        for attempt in range(1, self.max_retries + 1):
            try:
                r = self.sess.get(url, params=params, timeout=self.timeout)

                # Always redact the final URL before any logging
                safe_url = redact_apikey_in_text(r.url)

                if r.status_code == 200:
                    try:
                        return r.json(), None
                    except Exception:
                        return None, {
                            "error": "json_decode_error",
                            "status_code": r.status_code,
                            "text_head": redact_apikey_in_text(r.text[:500]),
                            "url": safe_url,
                        }

                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(self._backoff(attempt))
                    continue

                return None, {
                    "error": "http_error",
                    "status_code": r.status_code,
                    "text_head": redact_apikey_in_text(r.text[:800]),
                    "url": safe_url,
                }

            except requests.RequestException as e:
                time.sleep(self._backoff(attempt))
                if attempt == self.max_retries:
                    return None, {
                        "error": "request_exception",
                        "exception": redact_apikey_in_text(repr(e)),
                        "url": redact_apikey_in_text(url),
                    }

        return None, {"error": "max_retries_exceeded", "url": redact_apikey_in_text(url)}

    def _backoff(self, attempt: int) -> float:
        return self.base_sleep * (2 ** (attempt - 1)) + random.uniform(0, 0.2)

    @staticmethod
    def today_iso() -> str:
        import datetime as dt
        return dt.date.today().isoformat()

    @staticmethod
    def dump_json(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False)
