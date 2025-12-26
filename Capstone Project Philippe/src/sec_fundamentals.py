# src/sec_fundamentals.py
# -*- coding: utf-8 -*-

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .sec_client import SECClient


def _pick_units(facts: dict, tag: str) -> dict:
    try:
        us = facts.get("facts", {}).get("us-gaap", {})
        if tag in us:
            return us[tag].get("units", {})
    except Exception:
        pass
    try:
        ifrs = facts.get("facts", {}).get("ifrs-full", {})
        if tag in ifrs:
            return ifrs[tag].get("units", {})
    except Exception:
        pass
    return {}


def _choose_unit(units: dict, preferred: Tuple[str, ...] = ("USD", "shares")) -> list:
    if not isinstance(units, dict) or len(units) == 0:
        return []
    for u in preferred:
        if u in units:
            return units[u]
    for k in sorted(units.keys()):
        return units[k]
    return []


def _to_annual_fy(obs: list, allow_forms: Tuple[str, ...] = ("10-K", "20-F", "40-F")) -> pd.DataFrame:
    rows = []
    if not isinstance(obs, list):
        return pd.DataFrame(columns=["fyear", "fye_end", "val"])

    for o in obs:
        fy = o.get("fy")
        fp = o.get("fp")
        form = (o.get("form") or "").strip()
        end = o.get("end")
        val = o.get("val")

        if fy is None or fp != "FY":
            continue
        if form and (form not in allow_forms):
            continue
        if val is None or end is None:
            continue

        try:
            year = int(fy)
        except Exception:
            continue

        rows.append({"fyear": year, "fye_end": pd.to_datetime(end, errors="coerce"), "val": val})

    if not rows:
        return pd.DataFrame(columns=["fyear", "fye_end", "val"])

    df = pd.DataFrame(rows).dropna(subset=["fye_end"])
    if df.empty:
        return pd.DataFrame(columns=["fyear", "fye_end", "val"])

    df = df.sort_values(["fyear", "fye_end"])
    return df.drop_duplicates(subset=["fyear"], keep="last")


class SECFundamentalsBuilder:
    def __init__(self, config: Config):
        self.cfg = config
        self.sec = SECClient(config)
        self.alias = {"BRK-B": ["BRK.B", "BRK-B", "BRKB"], "BF-B": ["BF.B", "BF-B", "BFB"]}

    def _ticker_variants(self, t: str) -> List[str]:
        t = str(t).upper().strip()
        out = [t]
        if "-" in t:
            out.append(t.replace("-", "."))
        if "." in t:
            out.append(t.replace(".", "-"))
        out += self.alias.get(t, [])

        out2 = []
        for s in out:
            s2 = s.replace(" ", "").replace("/", "").strip().upper()
            if s2:
                out2.append(s2)

        uniq, seen = [], set()
        for s in (out + out2):
            s = str(s).upper().strip()
            if s and s not in seen:
                uniq.append(s)
                seen.add(s)
        return uniq

    def build_annual_sec_table(self, tickers: List[str], start_year: int = 2010, end_year: int = 2024) -> pd.DataFrame:
        print("\n[SEC] Downloading SEC fundamentals (annual FY)...")
        t2c = self.sec.fetch_ticker_cik_map(force_refresh=False)

        rows_all = []
        n_ok, n_skip = 0, 0

        for t in tickers:
            cik, used = None, None
            for v in self._ticker_variants(t):
                vv = v.replace(" ", "").replace("/", "").upper()
                if vv in t2c:
                    cik, used = t2c[vv], vv
                    break
            if cik is None:
                n_skip += 1
                continue

            facts = self.sec.fetch_companyfacts(cik, use_cache=True, ttl_days=None)
            if facts is None:
                n_skip += 1
                continue

            rev_units = (
                _pick_units(facts, "Revenues")
                or _pick_units(facts, "SalesRevenueNet")
                or _pick_units(facts, "RevenueFromContractWithCustomerExcludingAssessedTax")
            )
            rev = _to_annual_fy(_choose_unit(rev_units, ("USD",))).rename(columns={"val": "revenue"})

            ni = _to_annual_fy(_choose_unit(_pick_units(facts, "NetIncomeLoss"), ("USD",))).rename(columns={"val": "net_income"})

            eq_units = (
                _pick_units(facts, "StockholdersEquity")
                or _pick_units(facts, "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")
            )
            eq = _to_annual_fy(_choose_unit(eq_units, ("USD",))).rename(columns={"val": "equity"})

            ltd = _to_annual_fy(_choose_unit(_pick_units(facts, "LongTermDebt"), ("USD",))).rename(columns={"val": "ltd"})
            std_units = _pick_units(facts, "DebtCurrent") or _pick_units(facts, "LongTermDebtCurrent")
            std = _to_annual_fy(_choose_unit(std_units, ("USD",))).rename(columns={"val": "std"})

            cash = _to_annual_fy(_choose_unit(_pick_units(facts, "CashAndCashEquivalentsAtCarryingValue"), ("USD",))).rename(columns={"val": "cash"})
            sh = _to_annual_fy(_choose_unit(_pick_units(facts, "CommonStockSharesOutstanding"), ("shares",))).rename(columns={"val": "shares"})

            ebitda = _to_annual_fy(_choose_unit(_pick_units(facts, "EarningsBeforeInterestTaxesDepreciationAmortization"), ("USD",))).rename(columns={"val": "ebitda"})
            if ebitda.empty:
                op = _to_annual_fy(_choose_unit(_pick_units(facts, "OperatingIncomeLoss"), ("USD",))).rename(columns={"val": "op_income"})
                da = _to_annual_fy(_choose_unit(_pick_units(facts, "DepreciationDepletionAndAmortization"), ("USD",))).rename(columns={"val": "da"})
                if (not op.empty) and (not da.empty):
                    tmp = op.merge(da[["fyear", "da"]], on="fyear", how="inner")
                    tmp["ebitda"] = pd.to_numeric(tmp["op_income"], errors="coerce") + pd.to_numeric(tmp["da"], errors="coerce")
                    ebitda = tmp[["fyear", "fye_end", "ebitda"]]

            base = rev[["fyear", "fye_end", "revenue"]].copy() if not rev.empty else pd.DataFrame(columns=["fyear", "fye_end", "revenue"])

            for add in [
                ni[["fyear", "net_income"]] if not ni.empty else pd.DataFrame(columns=["fyear", "net_income"]),
                eq[["fyear", "equity"]] if not eq.empty else pd.DataFrame(columns=["fyear", "equity"]),
                ltd[["fyear", "ltd"]] if not ltd.empty else pd.DataFrame(columns=["fyear", "ltd"]),
                std[["fyear", "std"]] if not std.empty else pd.DataFrame(columns=["fyear", "std"]),
                cash[["fyear", "cash"]] if not cash.empty else pd.DataFrame(columns=["fyear", "cash"]),
                sh[["fyear", "shares"]] if not sh.empty else pd.DataFrame(columns=["fyear", "shares"]),
            ]:
                base = base.merge(add, on="fyear", how="outer")

            if "ebitda" in ebitda.columns and not ebitda.empty:
                base = base.merge(ebitda[["fyear", "ebitda"]], on="fyear", how="outer")

            base["ticker"] = t
            base["CIK"] = self.sec.cik10(cik)
            base["SymbolUsedForCIK"] = used

            base = base[(base["fyear"] >= start_year) & (base["fyear"] <= end_year)].copy()
            if base.empty:
                n_skip += 1
                continue

            for c in ["revenue", "net_income", "equity", "ltd", "std", "cash", "shares", "ebitda"]:
                if c in base.columns:
                    base[c] = pd.to_numeric(base[c], errors="coerce")

            base["debt"] = base[["ltd", "std"]].fillna(0).sum(axis=1)

            rows_all.append(base)
            n_ok += 1
            if n_ok % 25 == 0:
                print(f"  SEC progress: ok={n_ok} | skipped={n_skip} | last={t}")

        if not rows_all:
            return pd.DataFrame()

        out = pd.concat(rows_all, ignore_index=True)
        out = out.sort_values(["ticker", "fyear", "fye_end"]).drop_duplicates(["ticker", "fyear"], keep="last")
        return out

    def annual_to_monthly_panel(self, annual: pd.DataFrame, month_index: pd.DatetimeIndex) -> pd.DataFrame:
        if annual is None or annual.empty:
            return pd.DataFrame()

        a = annual.copy()
        a["fye_end"] = pd.to_datetime(a["fye_end"], errors="coerce")
        a = a.dropna(subset=["fye_end"])
        if a.empty:
            return pd.DataFrame()

        a["available_from"] = a["fye_end"] + pd.Timedelta(days=int(self.cfg.SEC_LAG_DAYS))
        a = a.sort_values(["ticker", "available_from"])

        panel_rows = []
        for t, sub in a.groupby("ticker", sort=False):
            sub = sub.dropna(subset=["available_from"]).sort_values("available_from")
            if sub.empty:
                continue

            for d in month_index:
                eligible = sub[sub["available_from"] <= d]
                if eligible.empty:
                    continue
                last = eligible.iloc[-1].to_dict()
                last["date"] = d
                panel_rows.append(last)

        if not panel_rows:
            return pd.DataFrame()

        panel = pd.DataFrame(panel_rows)
        panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
        panel = panel.sort_values(["ticker", "date", "available_from"]).drop_duplicates(["ticker", "date"], keep="last")
        return panel
