#!/usr/bin/env python3
"""
04_data_prep.py  --  build the backtest-ready dataset: indicators + membership per ticker.

Reads every raw price file in data/raw/ (from the 03a data pull) and computes the full
MAD-Velocity feature stack -- identical math to 01_spy.py, applied independently per
ticker -- then stamps point-in-time S&P 500 membership from results/02_sp500_constituents.csv.

FEATURES (all strictly trailing/causal)
  sma_20   20-day simple moving average of Close
  mad      100 * (Close - sma_20) / Close
  sigma    255-day rolling std of mad
  z        mad / sigma
  regime   {-3,-2,-1,+1,+2,+3} at z thresholds 0, +/-1, +/-2 (no zero state)
  days_in  trading days spent in the current regime visit
  p_up     walk-forward calibrated P(next transition is up); logistic on z for interior
           regimes, trailing base rate otherwise; only visits RESOLVED before day t feed
           day t (no look-ahead). base_p_up = the base rate alone.
  price_vel_5, dmad_5, mad_price_contrib_5, mad_sma_contrib_5, price_share_5
           the 5-day price-vs-MA decomposition of the MAD change
  sig_cross_up   regime crossed -1 -> +1 today (the confirmed entry signal)
  sig_rollback   regime rolled back +2 -> +1 today (the shared exit signal)
  in_index       ticker was an S&P 500 constituent on this date (2016-2025 window)

Full 20-year history is kept per ticker (warm-up rows carry NaNs); the backtest applies
its own window. Output files overwrite -- the build is deterministic.

    python3 04_data_prep.py
Output -> data/<TICKER>.csv          one backtest-ready file per ticker
          data/_prep_summary.csv     per-ticker audit (rows, spans, coverage, signals)
"""
import glob
import os

import numpy as np
import pandas as pd

RAWDIR = os.path.join("data", "raw")
OUTDIR = "data"
UNIVERSE = os.path.join("results", "02_sp500_constituents.csv")
SUMMARY = os.path.join(OUTDIR, "_prep_summary.csv")
SMA_W, VOL_W = 20, 255
K = 5                                    # decomposition horizon (trading days)
ORDER = [-3, -2, -1, 1, 2, 3]
RANK = {r: i for i, r in enumerate(ORDER)}
INTERIOR = {-2, -1, 1, 2}
WINDOW_OBS, MIN_N = 500, 60              # walk-forward pool window / min sample
PCLIP = (0.02, 0.98)
SKIP_EXISTING = True                     # RESUME: skip tickers whose output is already
                                         # complete (row count + last date must match the
                                         # raw file, so a truncated file from a broken
                                         # pipe is detected and recomputed automatically)
os.makedirs(OUTDIR, exist_ok=True)


# ================================================================ walk-forward helpers
def logistic_fit(x, y, iters=30):
    a = b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a + b * x)))
        w = np.clip(p * (1 - p), 1e-6, None)
        ga, gb = np.sum(y - p), np.sum((y - p) * x)
        Haa, Hab, Hbb = np.sum(w), np.sum(w * x), np.sum(w * x * x)
        det = Haa * Hbb - Hab * Hab
        if abs(det) < 1e-9:
            break
        a += (Hbb * ga - Hab * gb) / det
        b += (Haa * gb - Hab * ga) / det
    return a, b


def clip(p):
    return float(min(max(p, PCLIP[0]), PCLIP[1]))


def build_visits(regime, z):
    n = len(regime); visits = []; i = 0
    while i < n and np.isnan(regime[i]):
        i += 1
    while i < n:
        if np.isnan(regime[i]):
            i += 1; continue
        j = i
        while j + 1 < n and regime[j + 1] == regime[i]:
            j += 1
        nxt = regime[j + 1] if (j + 1 < n and not np.isnan(regime[j + 1])) else None
        ex = None if nxt is None else ("up" if RANK[int(nxt)] > RANK[int(regime[i])] else "down")
        days = [{"idx": t, "z": float(z[t]), "days_in": t - i + 1} for t in range(i, j + 1)]
        visits.append({"regime": int(regime[i]), "exit": ex, "days": days})
        i = j + 1
    return visits


def walk_forward_pup(regime, z):
    """Per-ticker walk-forward P(up); returns (days_in, p_up, base_p_up) arrays."""
    n = len(regime)
    pool = {r: [] for r in ORDER}
    days_in = np.full(n, np.nan)
    pup = np.full(n, np.nan)
    base = np.full(n, np.nan)
    for v in build_visits(regime, z):
        r = v["regime"]
        for d in v["days"]:
            obs = pool[r][-WINDOW_OBS:]
            if len(obs) < MIN_N:
                continue
            ups = sum(u for _, u in obs)
            base_p = clip((ups + 1) / (len(obs) + 2))
            if r in INTERIOR and 0 < ups < len(obs):
                a, b = logistic_fit(np.array([zz for zz, _ in obs]),
                                    np.array([u for _, u in obs], float))
                model_p = clip(1 / (1 + np.exp(-(a + b * d["z"]))))
            else:
                model_p = base_p
            days_in[d["idx"]] = d["days_in"]
            pup[d["idx"]] = round(model_p, 4)
            base[d["idx"]] = round(base_p, 4)
        if v["exit"] is not None:
            up = 1 if v["exit"] == "up" else 0
            pool[r].extend((dd["z"], up) for dd in v["days"])
    return days_in, pup, base


# ================================================================ membership intervals
mem = pd.read_csv(UNIVERSE)
intervals = {}                                        # ticker -> [(start, end)], closed
for r in mem.itertuples():
    intervals.setdefault(r.ticker, []).append(
        (pd.Timestamp(r.first_date), pd.Timestamp(r.last_date)))


def in_index_flags(tk, dates):
    flags = np.zeros(len(dates), bool)
    for a, b in intervals.get(tk, []):
        flags |= (dates >= a) & (dates <= b)
    return flags


def expected_member_days(tk):
    """Weekday count of the membership intervals -- what member_days SHOULD be when the
    price file fully covers the membership era (runs ~2% high: exchange holidays).
    coverage_pct well below ~95 flags a data hole (e.g. an unfetched pre-rename symbol)."""
    return int(sum(np.busday_count(a.date(), (b + pd.Timedelta(days=1)).date())
                   for a, b in intervals.get(tk, [])))


# ================================================================ per-ticker build
files = sorted(f for f in glob.glob(os.path.join(RAWDIR, "*.csv"))
               if not os.path.basename(f).startswith("_"))
print(f"raw files: {len(files)}   universe tickers: {mem['ticker'].nunique()}")
no_member = mem["ticker"].nunique() - sum(
    os.path.exists(os.path.join(RAWDIR, f"{t}.csv")) for t in intervals)
print(f"universe tickers with no raw file (see data/raw/_missing.csv): {no_member}\n")

audit = []
n_skipped = 0
for i, path in enumerate(files, 1):
    tk = os.path.basename(path)[:-4]                  # strip ".csv"; keeps dots (BRK.B)
    raw = pd.read_csv(path)
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = (raw.dropna(subset=["Open", "Close"])
              .sort_values("Date")
              .drop_duplicates(subset="Date", keep="first")
              .reset_index(drop=True))
    dates = raw["Date"]
    C = raw["Close"].astype(float)
    n = len(raw)

    # ---- RESUME: skip if a COMPLETE output already exists (truncated files recompute)
    outpath = os.path.join(OUTDIR, f"{tk}.csv")
    if SKIP_EXISTING and os.path.exists(outpath):
        try:
            prev = pd.read_csv(outpath)
            complete = (len(prev) == n
                        and str(prev["Date"].iloc[-1]) == dates.iloc[-1].strftime("%Y-%m-%d"))
        except Exception:
            complete = False
        if complete:
            pm = prev["in_index"].astype(str).str.strip().isin(["True", "true", "1", "1.0"])
            px = prev["sig_cross_up"].astype(str).str.strip().isin(["True", "true", "1", "1.0"])
            exp = expected_member_days(tk)
            audit.append({"ticker": tk, "rows": len(prev),
                          "first_date": str(prev["Date"].iloc[0]),
                          "last_date": str(prev["Date"].iloc[-1]),
                          "regime_days": int(prev["regime"].notna().sum()),
                          "pup_days": int(prev["p_up"].notna().sum()),
                          "member_days": int(pm.sum()),
                          "expected_member_days": exp,
                          "coverage_pct": round(100 * pm.sum() / exp, 1) if exp else 100.0,
                          "member_days_with_regime": int((pm & prev["regime"].notna()).sum()),
                          "cross_up_events": int(px.sum()),
                          "in_universe": tk in intervals})
            n_skipped += 1
            if i % 50 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] {tk:<8} skipped (complete)")
            continue

    # indicator stack (identical to 01_spy.py)
    sma = C.rolling(SMA_W).mean()
    mad = 100 * (C - sma) / C
    sigma = mad.rolling(VOL_W).std()
    z = (mad / sigma).to_numpy()
    regime = np.select([z > 2, z > 1, z > 0, z > -1, z > -2],
                       [3, 2, 1, -1, -2], default=-3).astype(float)
    regime[np.isnan(z)] = np.nan

    days_in, pup, base_p = walk_forward_pup(regime, z)

    # 5-day price-vs-MA decomposition
    dP = C - C.shift(K)
    dSMA = sma - sma.shift(K)
    price_c = (100.0 * (sma / C**2) * dP).to_numpy()
    sma_c = (-100.0 * (1.0 / C) * dSMA).to_numpy()
    denom = np.abs(price_c) + np.abs(sma_c)
    pshare = np.where(denom > 0, price_c / denom, np.nan)

    # signals
    sig_cross_up = np.zeros(n, bool)
    sig_rollback = np.zeros(n, bool)
    for t in range(1, n):
        if np.isnan(regime[t]) or np.isnan(regime[t - 1]):
            continue
        if regime[t] == 1 and regime[t - 1] == -1:
            sig_cross_up[t] = True
        if regime[t] == 1 and regime[t - 1] == 2:
            sig_rollback[t] = True

    member = in_index_flags(tk, dates)

    out = pd.DataFrame({"Date": dates.dt.strftime("%Y-%m-%d"),
                        "Open": raw["Open"].astype(float).round(4),
                        "High": raw["High"].astype(float).round(4),
                        "Low": raw["Low"].astype(float).round(4),
                        "Close": C.round(4),
                        "Volume": raw["Volume"]})
    out["sma_20"] = sma.to_numpy().round(4)
    out["mad"] = mad.to_numpy().round(4)
    out["sigma"] = sigma.to_numpy().round(4)
    out["z"] = np.round(z, 4)
    out["regime"] = [int(x) if x == x else np.nan for x in regime]
    out["days_in"] = days_in
    out["p_up"] = pup
    out["base_p_up"] = base_p
    out[f"price_vel_{K}"] = (C / C.shift(K) - 1.0).to_numpy().round(6)
    out[f"dmad_{K}"] = (mad - mad.shift(K)).to_numpy().round(4)
    out[f"mad_price_contrib_{K}"] = np.round(price_c, 4)
    out[f"mad_sma_contrib_{K}"] = np.round(sma_c, 4)
    out[f"price_share_{K}"] = np.round(pshare, 4)
    out["sig_cross_up"] = sig_cross_up
    out["sig_rollback"] = sig_rollback
    out["in_index"] = member
    out.to_csv(os.path.join(OUTDIR, f"{tk}.csv"), index=False)

    exp = expected_member_days(tk)
    audit.append({"ticker": tk, "rows": n,
                  "first_date": dates.iloc[0].strftime("%Y-%m-%d") if n else "",
                  "last_date": dates.iloc[-1].strftime("%Y-%m-%d") if n else "",
                  "regime_days": int((~np.isnan(regime)).sum()),
                  "pup_days": int((~np.isnan(pup)).sum()),
                  "member_days": int(member.sum()),
                  "expected_member_days": exp,
                  "coverage_pct": round(100 * member.sum() / exp, 1) if exp else 100.0,
                  "member_days_with_regime": int((member & ~np.isnan(regime)).sum()),
                  "cross_up_events": int(sig_cross_up.sum()),
                  "in_universe": tk in intervals})
    if i % 50 == 0 or i == len(files):
        print(f"  [{i}/{len(files)}] {tk:<8} rows {n:>5}  member_days {int(member.sum()):>5}  "
              f"crossings {int(sig_cross_up.sum()):>3}")

aud = pd.DataFrame(audit)
aud.to_csv(SUMMARY, index=False)

# ---------------------------------------------------------------- console audit
ok = aud[aud["member_days"] > 0]
print(f"\nprepped {len(aud)} tickers -> {OUTDIR}/   "
      f"(computed {len(aud) - n_skipped}, resumed-skip {n_skipped})")
print(f"with membership overlap: {len(ok)}   without (check): {len(aud) - len(ok)}")
print(f"total member-days: {int(aud['member_days'].sum()):,}   "
      f"with regime formed: {int(aud['member_days_with_regime'].sum()):,} "
      f"({100 * aud['member_days_with_regime'].sum() / max(aud['member_days'].sum(), 1):.1f}%)")
print(f"total -1->+1 crossings: {int(aud['cross_up_events'].sum()):,}")
low = aud[(aud["expected_member_days"] > 40) & (aud["coverage_pct"] < 85)]
print(f"member-day coverage below 85% (data holes -- check PREDECESSOR map in 03): {len(low)}")
for r in low.sort_values("coverage_pct").itertuples():
    print(f"    {r.ticker:<8} expected {r.expected_member_days:>5}  got {r.member_days:>5}  "
          f"({r.coverage_pct:.1f}%)")
print(f"wrote {SUMMARY}")
