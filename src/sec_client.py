# src/sec_client.py
# -*- coding: utf-8 -*-

import json
import random
import time
from pathlib import Path
from typing import Dict, Optional

import requests

from .config import Config


class SECClient:
    """
    Minimal SEC client with:
    - rate limit sleep
    - retries + backoff
    - local JSON cache
    """
    def __init__(self, config: Config):
        self.cfg = config
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16)
        self.session.mount("https://", adapter)

        self.cache_dir = Path(self.cfg.SEC_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "companyfacts").mkdir(parents=True, exist_ok=True)

        self._last_t = 0.0

    def _sleep_rate_limit(self):
        dt = time.time() - self._last_t
        if dt < self.cfg.SEC_MIN_INTERVAL_SEC:
            time.sleep(self.cfg.SEC_MIN_INTERVAL_SEC - dt)
        self._last_t = time.time()

    @staticmethod
    def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8"):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding=encoding)
        tmp.replace(path)

    def _get(self, url: str) -> Optional[requests.Response]:
        if "@" not in self.cfg.SEC_USER_AGENT:
            raise RuntimeError("Config.SEC_USER_AGENT invalide: mets un email.")

        headers = {
            "User-Agent": self.cfg.SEC_USER_AGENT,
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }

        last_err = None
        for attempt in range(self.cfg.SEC_MAX_RETRIES):
            self._sleep_rate_limit()
            try:
                r = self.session.get(url, headers=headers, timeout=self.cfg.SEC_TIMEOUT_SEC)
                if r.status_code == 200:
                    return r

                retry_after = r.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        time.sleep(max(float(retry_after), 0.0))
                        continue
                    except Exception:
                        pass

                if r.status_code in (403, 429) or (500 <= r.status_code <= 599):
                    backoff = (0.5 * (2 ** attempt)) + random.uniform(0, 0.5)
                    time.sleep(backoff)
                    continue

                backoff = (0.3 * (2 ** attempt)) + random.uniform(0, 0.3)
                time.sleep(backoff)

            except Exception as e:
                last_err = e
                backoff = (0.5 * (2 ** attempt)) + random.uniform(0, 0.5)
                time.sleep(backoff)

        print(f"[SEC] FAIL url={url} err={last_err}")
        return None

    def get_json(self, url: str) -> Optional[dict]:
        r = self._get(url)
        if r is None:
            return None
        try:
            return r.json()
        except Exception:
            return None

    def fetch_ticker_cik_map(self, force_refresh: bool = False) -> Dict[str, int]:
        cache_path = self.cache_dir / "company_tickers.json"
        data = None

        if cache_path.exists() and not force_refresh:
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                data = None

        if data is None:
            url = "https://www.sec.gov/files/company_tickers.json"
            data = self.get_json(url)
            if data is None:
                raise RuntimeError("Impossible de télécharger company_tickers.json")
            try:
                self._atomic_write_text(cache_path, json.dumps(data), encoding="utf-8")
            except Exception:
                pass

        out = {}
        for _, rec in data.items():
            t = str(rec.get("ticker", "")).upper().strip()
            cik = rec.get("cik_str", None)
            if t and cik is not None:
                try:
                    out[t.replace(" ", "").replace("/", "")] = int(cik)
                except Exception:
                    continue
        return out

    @staticmethod
    def cik10(cik: int) -> str:
        return str(int(cik)).zfill(10)

    def fetch_companyfacts(self, cik: int, use_cache: bool = True, ttl_days: Optional[int] = None) -> Optional[dict]:
        path = self.cache_dir / "companyfacts" / f"companyfacts_{self.cik10(cik)}.json"

        if use_cache and path.exists():
            try:
                if ttl_days is not None:
                    age_sec = time.time() - path.stat().st_mtime
                    if age_sec > ttl_days * 86400:
                        raise ValueError("Cache expired")
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass

        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{self.cik10(cik)}.json"
        data = self.get_json(url)
        if data is None:
            return None

        try:
            self._atomic_write_text(path, json.dumps(data), encoding="utf-8")
        except Exception:
            pass

        return data
