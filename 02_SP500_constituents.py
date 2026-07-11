#!/usr/bin/env python3
"""
02_SP500_constituents.py  --  point-in-time S&P 500 membership, by year, 2016-2025.

Downloads the fja05680 historical-components dataset -- the "(Updated)" file; the plain
file FROZE at Jan-2019 and silently misses every membership change after that (the bug
that produced identical member-day counts across names). A freshness check hard-fails
if the snapshot history looks stale.

The dataset is a list of snapshots (date, comma-separated tickers); each snapshot is in
force until the next one. For every calendar year 2016-2025 this script emits one row per
member with its membership span WITHIN that year:

  year        2016..2025
  ticker      symbol as recorded historically in the dataset
  yf_ticker   symbol for price fetching today: RENAME map (ticker changes, e.g. FB->META)
              applied, then dots -> dashes (BRK.B -> BRK-B)
  first_date  first calendar day of membership within the year
  last_date   last calendar day of membership within the year
  full_year   True if the ticker was a member the entire calendar year
  episodes    number of disjoint membership spells touching the year (>1 = left & rejoined)

    python3 02_SP500_constituents.py
Output -> results/02_sp500_constituents.csv   (+ per-year add/drop audit on the console)
"""
import io
import os
import urllib.request

import pandas as pd

OUTDIR = "results"
OUT = os.path.join(OUTDIR, "02_sp500_constituents.csv")
SRC = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
       "S%26P%20500%20Historical%20Components%20%26%20Changes%20%28Updated%29.csv")
UA = "Mozilla/5.0"
YEARS = list(range(2016, 2026))          # 2016 .. 2025
FRESH_MIN = pd.Timestamp("2025-06-01")   # last snapshot must be at least this recent
RENAME = {                               # historical symbol -> current fetchable symbol
    "ABC": "COR", "ANTM": "ELV", "BHGE": "BKR", "ADS": "BFH", "FB": "META",
    "WLTW": "WTW", "UTX": "RTX", "HFC": "DINO", "FISV": "FI", "FBHS": "FBIN",
    "COG": "CTRA", "PKI": "RVTY", "RE": "EG",
    "FLT": "CPAY", "TMK": "GL", "CTL": "LUMN", "KORS": "CPRI", "WYND": "TNL", "HCP": "DOC",
}
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------- download + parse
req = urllib.request.Request(SRC, headers={"User-Agent": UA})
raw = urllib.request.urlopen(req, timeout=60).read().decode("utf-8")
mem = pd.read_csv(io.StringIO(raw))
mem["date"] = pd.to_datetime(mem["date"])
mem = mem.sort_values("date").reset_index(drop=True)
snap_dates = list(mem["date"])
snap_sets = [set(t.strip().upper() for t in row.split(",") if t.strip())
             for row in mem["tickers"]]
last_snap = snap_dates[-1]
print(f"snapshots: {len(mem)}   {snap_dates[0].date()} -> {last_snap.date()}")
if last_snap < FRESH_MIN:
    raise SystemExit(f"STALE MEMBERSHIP FILE: last snapshot {last_snap.date()} < {FRESH_MIN.date()} "
                     f"-- this is the frozen-file failure mode; do not trust it.")

# ---------------------------------------------------------------- per-ticker membership intervals
# snapshot i is in force over [date_i, date_{i+1}); the final snapshot extends onward.
END_CAP = pd.Timestamp(f"{YEARS[-1]}-12-31")
intervals = {}                                            # ticker -> list of [start, end] closed
for i, (d, s) in enumerate(zip(snap_dates, snap_sets)):
    end = (snap_dates[i + 1] - pd.Timedelta(days=1)) if i + 1 < len(snap_dates) else END_CAP
    if end < d:
        continue
    for tk in s:
        iv = intervals.setdefault(tk, [])
        if iv and iv[-1][1] + pd.Timedelta(days=1) >= d:  # contiguous with previous spell -> extend
            iv[-1][1] = max(iv[-1][1], end)
        else:
            iv.append([d, end])

# ---------------------------------------------------------------- per-year rows
def yf_symbol(tk):
    return RENAME.get(tk, tk).replace(".", "-")

rows = []
year_sets = {}
for y in YEARS:
    y0, y1 = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")
    members = set()
    for tk, ivs in intervals.items():
        hits = [(max(a, y0), min(b, y1)) for a, b in ivs if a <= y1 and b >= y0]
        if not hits:
            continue
        members.add(tk)
        first = min(h[0] for h in hits)
        last = max(h[1] for h in hits)
        rows.append({"year": y, "ticker": tk, "yf_ticker": yf_symbol(tk),
                     "first_date": first.strftime("%Y-%m-%d"),
                     "last_date": last.strftime("%Y-%m-%d"),
                     "full_year": bool(len(hits) == 1 and first == y0 and last == y1),
                     "episodes": len(hits)})
    year_sets[y] = members

out = pd.DataFrame(rows).sort_values(["year", "ticker"]).reset_index(drop=True)
out.to_csv(OUT, index=False)

# ---------------------------------------------------------------- console audit
print(f"\n{'year':<6}{'members':>8}{'full_yr':>8}{'added':>7}{'dropped':>8}")
prev = None
for y in YEARS:
    d = out[out["year"] == y]
    add = len(year_sets[y] - prev) if prev is not None else 0
    drop = len(prev - year_sets[y]) if prev is not None else 0
    print(f"{y:<6}{len(d):>8}{int(d['full_year'].sum()):>8}"
          f"{(add if prev is not None else '-'):>7}{(drop if prev is not None else '-'):>8}")
    prev = year_sets[y]
uniq = out["ticker"].nunique()
rejoin = out[out["episodes"] > 1]
print(f"\nunique tickers 2016-2025: {uniq}   rows: {len(out)}")
print(f"left-and-rejoined within a year: {len(rejoin)} rows"
      + (f"  ({', '.join(sorted(set(rejoin['ticker']))[:8])}...)" if len(rejoin) else ""))
print(f"renames applied in yf_ticker: {sum(out['ticker'] != out['yf_ticker'])} rows")
print(f"wrote {OUT}")
