#!/usr/bin/env python3
"""
03_data.py  --  raw daily price data from polygon.io for every 2016-2025 constituent.

Reads the ticker universe from results/02_sp500_constituents.csv (built by 02) and pulls
the LAST 20 YEARS of daily aggregates for each unique historical ticker. Polygon serves
DELISTED securities, so the names yfinance could never return (SIVB, FRC, AABA, ...) are
now recoverable -- this is the fix for the ~10% survivorship hole.

For each historical ticker the script tries, in order, and MERGES by date:
  1. the historical symbol itself          (Polygon keeps delisted history under it)
  2. its renamed successor from yf_ticker  (e.g. FB -> META: old span under FB, new
     span under META; the merge stitches the full series)
  3. dot/dash variants of both             (class shares: BRK.B / BRK-B)
Earlier candidates win on duplicate dates. source_ticker records where each row came from.

NOTES
  - adjusted=true on Polygon adjusts for SPLITS ONLY (no dividend reinvestment). This
    differs from the old yfinance auto_adjust=True total-return series. Consistent across
    all tickers here; can be revisited with Polygon's dividends endpoint later.
  - Ticker-symbol REUSE exists in history (one symbol, two companies, e.g. CEG). The
    20-year raw pull does not resolve this; the membership dates from 02 gate the
    backtest window to the correct company-era downstream.
  - RESUMABLE: tickers whose csv already exists in data/raw/ are skipped -- rerun to
    continue after an interruption. Failures/empties logged to data/raw/_missing.csv.

API key: read from the POLYGON_API_KEY environment variable
(export POLYGON_API_KEY=...). Never printed or logged.

    python3 03_data.py
Output -> data/raw/<TICKER>.csv   (Date,Open,High,Low,Close,Volume,VWAP,Transactions,source_ticker)
          data/raw/_missing.csv   (tickers with no data + reason)
"""
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone

UNIVERSE = os.path.join("results", "02_sp500_constituents.csv")
RAWDIR = os.path.join("data", "raw")
MISSING = os.path.join(RAWDIR, "_missing.csv")
YEARS_BACK = 20
SLEEP = 0.15                 # pacing between requests (paid plans tolerate far more)
RETRIES = [2, 5, 15, 60]     # backoff on 429/5xx
# PREDECESSOR symbols: the membership file retroactively records renamed companies under
# their NEW symbol for the old era, but Polygon stores that era under the OLD symbol.
# Without these, coverage audits show 0-85% member-day coverage on exactly these names.
# Predecessors are tried BEFORE the RENAME successor so the correct lineage wins duplicate
# dates (also displaces unrelated symbol-reuse rows, e.g. the pre-2013 "TNL" ~ WYND case).
PREDECESSOR = {
    "AABA": "YHOO", "CCEP": "CCE", "WYND": "WYN", "JEF": "LUK", "ANDV": "TSO",
    "KDP": "DPS", "BHGE": "BHI", "CBRE": "CBG", "WELL": "HCN", "BKNG": "PCLN",
    "APTV": "DLPH", "TPR": "COH", "UAA": "UA", "CPRI": "KORS",
}
os.makedirs(RAWDIR, exist_ok=True)


# ---------------------------------------------------------------- API key
def load_key():
    key = os.environ.get("POLYGON_API_KEY", "").strip()
    if key:
        return key
    sys.exit("POLYGON_API_KEY not set. Get a key at polygon.io, then:  "
             "export POLYGON_API_KEY=yourkey")


KEY = load_key()
END = date.today()
START = END - timedelta(days=round(YEARS_BACK * 365.25))


# ---------------------------------------------------------------- fetch one symbol
def fetch_symbol(sym):
    """All daily bars for sym over [START, END]. Returns list of row-dicts (may be empty).
    Follows next_url pagination. Retries on 429/5xx; 404/no-data -> empty."""
    url = (f"https://api.polygon.io/v2/aggs/ticker/{sym}/range/1/day/{START}/{END}"
           f"?adjusted=true&sort=asc&limit=50000&apiKey={KEY}")
    rows = []
    while url:
        payload = None
        for wait in RETRIES + [None]:
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:
                    payload = json.loads(resp.read().decode())
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return rows
                if e.code in (429, 500, 502, 503, 504) and wait is not None:
                    time.sleep(wait)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError):
                if wait is None:
                    raise
                time.sleep(wait)
        for b in payload.get("results") or []:
            d = datetime.fromtimestamp(b["t"] / 1000, tz=timezone.utc).date()
            rows.append({"Date": d.isoformat(),
                         "Open": b.get("o"), "High": b.get("h"), "Low": b.get("l"),
                         "Close": b.get("c"), "Volume": b.get("v"),
                         "VWAP": b.get("vw"), "Transactions": b.get("n"),
                         "source_ticker": sym})
        url = payload.get("next_url")
        if url:
            url += f"&apiKey={KEY}"
        time.sleep(SLEEP)
    return rows


def candidates(tk, yf_tk):
    """Symbols to try, in priority order, deduped: the historical symbol, its dot/dash
    variant, the PREDECESSOR (pre-rename) symbol, then the renamed successor."""
    pred = PREDECESSOR.get(tk, "")
    cands = [tk]
    for v in (tk.replace(".", "-") if "." in tk else tk.replace("-", "."),
              pred, pred.replace(".", "-"),
              yf_tk, yf_tk.replace("-", ".")):
        if v and v not in cands:
            cands.append(v)
    return cands


# ---------------------------------------------------------------- universe
tickers = {}                                           # historical ticker -> yf_ticker
with open(UNIVERSE) as f:
    for r in csv.DictReader(f):
        tickers[r["ticker"]] = r["yf_ticker"]
todo = sorted(tickers)
print(f"universe: {len(todo)} unique tickers   window {START} -> {END}   out {RAWDIR}/")

done = skipped = empty = 0
missing = []
t0 = time.time()
for i, tk in enumerate(todo, 1):
    path = os.path.join(RAWDIR, f"{tk}.csv")
    if os.path.exists(path):
        skipped += 1
        continue
    merged = {}                                        # date -> row (first candidate wins)
    used = []
    for sym in candidates(tk, tickers[tk]):
        try:
            rows = fetch_symbol(sym)
        except Exception as e:                         # hard failure on this candidate
            missing.append({"ticker": tk, "symbol": sym, "reason": f"error: {e}"})
            continue
        fresh = 0
        for r in rows:
            if r["Date"] not in merged:
                merged[r["Date"]] = r
                fresh += 1
        if fresh:
            used.append(f"{sym}:{fresh}")
    if not merged:
        empty += 1
        missing.append({"ticker": tk, "symbol": "|".join(candidates(tk, tickers[tk])),
                        "reason": "no data returned"})
        print(f"  [{i}/{len(todo)}] {tk:<8} EMPTY")
        continue
    rows = [merged[d] for d in sorted(merged)]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Date", "Open", "High", "Low", "Close",
                                          "Volume", "VWAP", "Transactions", "source_ticker"])
        w.writeheader()
        w.writerows(rows)
    done += 1
    if done % 25 == 0 or i == len(todo):
        rate = done / max(time.time() - t0, 1)
        print(f"  [{i}/{len(todo)}] {tk:<8} {len(rows):>5} rows "
              f"({', '.join(used)})   {rate:.1f} tickers/s")

with open(MISSING, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["ticker", "symbol", "reason"])
    w.writeheader()
    w.writerows(missing)

print(f"\nfetched {done}   skipped(existing) {skipped}   empty {empty}   "
      f"elapsed {(time.time()-t0)/60:.1f} min")
print(f"missing log -> {MISSING} ({len(missing)} entries)")
