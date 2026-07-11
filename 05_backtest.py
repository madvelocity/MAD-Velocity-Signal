#!/usr/bin/env python3
"""
05_backtest.py  --  cross-sectional entry-timing backtest over the prepped universe.

Runs the consolidated variant set (01_spy.py's) INDEPENDENTLY on every ticker in data/,
gated to point-in-time S&P 500 membership (in_index from 02/04), and measured on an
EXPOSURE-ADJUSTED basis -- return per day actually deployed, never the full calendar.

VARIANTS (shared exit: regime rolls back +2 -> +1 -> out next open; long/flat; one
position; next-open execution; GROSS; no stop)
  buy_and_hold    hold every member day (each ticker's own benchmark)
  R1_confirmed    enter after a -1 -> +1 cross (late, confirmed)
  R1_share_ge50   confirmed cross AND price_share_5 >= 0.50 (price-dominant)
  ANT_pup50/60/70 ANTICIPATORY: enter AT the -1 state when P(up) >= threshold
  ANT_all         control: every -1 state day with P(up) available, forecast ignored

Membership gating: a ticker can only hold/enter while in_index is True (the membership
file spans 2016-2025, so the test window is 2016+ by construction). A name needs >= 252
member days with a formed regime to qualify.

METRICS per ticker x variant: days_in_trade (vs member days), exposure %, round trips,
total return, deployed_cagr = wealth^(252/days_in_trade)-1, in-trade Sharpe (ddof=1, rf=0),
bps/day = 10000*ln(wealth)/days_in_trade, days-per-trade (median), win rate, avg trade
return, maxdd, and beat-own-B&H flags. Summary reports MEAN and MEDIAN plus %-beat-B&H.
Paired McNemar tests (continuity-corrected) on the key variant comparisons ship in the
output so the significance calls are reproducible.

    python3 05_backtest.py
Outputs -> results/05_perticker_detail.csv    one row per ticker x variant
           results/05_perticker_summary.csv   mean+median per variant, %-beat-B&H
           results/05_top_performers.csv      highlights (>= 15 completed trades)
           results/05_paired_tests.csv        McNemar comparisons
"""
import csv
import glob
import math
import os
import statistics as st

import numpy as np
import pandas as pd

DATADIR = "data"
OUTDIR = "results"
DETAIL = os.path.join(OUTDIR, "05_perticker_detail.csv")
SUMMARY = os.path.join(OUTDIR, "05_perticker_summary.csv")
TOPPERF = os.path.join(OUTDIR, "05_top_performers.csv")
PAIRED = os.path.join(OUTDIR, "05_paired_tests.csv")
COST = 0.0                               # GROSS
PUP_MIN = 0.50                           # confirmed-cross / share-gate era threshold
ANT_THR = [0.50, 0.60, 0.70]             # anticipatory P(up) ladder
SHARE_MIN = 0.50
WIN_START = pd.Timestamp("2016-01-01")   # belt & suspenders; in_index already spans 2016+
MIN_DAYS = 252
MIN_TRADES_HL = 15
USECOLS = ["Date", "Open", "Close", "regime", "p_up", "price_share_5",
           "sig_cross_up", "sig_rollback", "in_index"]
VARIANTS = (["buy_and_hold", "R1_confirmed", "R1_share_ge50"]
            + [f"ANT_pup{int(round(100 * t))}" for t in ANT_THR] + ["ANT_all"])
STRATS = [v for v in VARIANTS if v != "buy_and_hold"]
KEY_PAIRS = [("ANT_pup50", "R1_confirmed"), ("ANT_pup50", "R1_share_ge50"),
             ("ANT_all", "ANT_pup50"), ("R1_share_ge50", "R1_confirmed"),
             ("ANT_pup50", "ANT_pup70")]
os.makedirs(OUTDIR, exist_ok=True)


def truthy(series):
    return series.astype(str).str.strip().isin(["True", "true", "1", "1.0"]).to_numpy()


def metrics(rets, intrade, trade_rets, trade_days, dit, total_days):
    r = np.asarray(rets, float)
    eq = np.cumprod(1 + r) if len(r) else np.array([1.0])
    wealth = float(eq[-1])
    deployed_cagr = wealth ** (252.0 / dit) - 1 if dit > 0 else float("nan")
    ir = np.asarray(intrade, float)
    sharpe = (np.sqrt(252) * ir.mean() / ir.std(ddof=1)
              if len(ir) > 1 and ir.std(ddof=1) > 0 else float("nan"))
    bps = 10000 * np.log(wealth) / dit if (dit > 0 and wealth > 0) else float("nan")
    dd = eq / np.maximum.accumulate(eq) - 1
    tr = np.asarray(trade_rets, float)
    td = np.asarray(trade_days, float)
    return {"days_in_trade": int(dit), "total_days": int(total_days),
            "exposure_pct": round(100 * dit / total_days, 1) if total_days else 0.0,
            "round_trips": len(tr),
            "total_return": round(wealth - 1, 4),
            "deployed_cagr": round(deployed_cagr, 4),
            "intrade_sharpe": (round(sharpe, 3) if sharpe == sharpe else float("nan")),
            "bps_per_day": round(bps, 2) if bps == bps else float("nan"),
            "med_days_per_trade": (float(np.median(td)) if len(td) else float("nan")),
            "win_rate": (round(float((tr > 0).mean()), 3) if len(tr) else float("nan")),
            "avg_trade_return": (round(float(tr.mean()), 4) if len(tr) else float("nan")),
            "maxdd": round(float(dd.min()), 4)}


def run_ticker(df):
    O, C = df["Open"].to_numpy(float), df["Close"].to_numpy(float)
    reg = df["regime"].to_numpy(float)
    pup = df["p_up"].to_numpy(float)
    pshare = df["price_share_5"].to_numpy(float)
    cross = truthy(df["sig_cross_up"])
    rollb = truthy(df["sig_rollback"])
    member = truthy(df["in_index"]) & (pd.to_datetime(df["Date"]).to_numpy()
                                       >= np.datetime64(WIN_START))
    n = len(C)
    valid = ~np.isnan(reg)
    member_days = int((valid & member).sum())
    if member_days < MIN_DAYS:
        return None
    start = int(np.argmax(valid))
    if (n - start) < 2:
        return None

    def bh():
        rets = [C[t] / C[t - 1] - 1 for t in range(start + 1, n) if member[t]]
        return metrics(rets, rets, [], [], len(rets), len(rets))

    def strat(entry_ok):
        rets, intrade, trade_rets, trade_days = [], [], [], []
        pos, eq, tstart, dit, days_this, tradeable = 0, 1.0, 1.0, 0, 0, 0
        for t in range(start + 1, n):
            if not member[t]:
                pos = 0
                continue
            tradeable += 1
            in_pos, entered, exited = False, False, False
            if pos == 1:
                in_pos = True
                dit += 1
                days_this += 1
                if rollb[t - 1]:
                    r = O[t] / C[t - 1] - 1
                    pos, exited = 0, True
                else:
                    r = C[t] / C[t - 1] - 1
            else:
                if entry_ok(t):
                    in_pos, entered = True, True
                    dit += 1
                    days_this = 1
                    r = C[t] / O[t] - 1
                    pos = 1
                else:
                    r = 0.0
            if entered or exited:
                r -= COST
            if entered:
                tstart = eq
            eq *= (1 + r)
            rets.append(r)
            if in_pos:
                intrade.append(r)
            if exited:
                trade_rets.append(eq / tstart - 1)
                trade_days.append(days_this)
        return metrics(rets, intrade, trade_rets, trade_days, dit, tradeable)

    def e_confirmed(t):
        return cross[t - 1]

    def e_share50(t):
        g = pshare[t - 1]
        return cross[t - 1] and g == g and g >= SHARE_MIN

    def e_ant(thr):
        def f(t):
            return reg[t - 1] == -1 and pup[t - 1] == pup[t - 1] and pup[t - 1] >= thr
        return f

    def e_ant_all(t):
        return reg[t - 1] == -1 and pup[t - 1] == pup[t - 1]

    out = {"buy_and_hold": bh(),
           "R1_confirmed": strat(e_confirmed),
           "R1_share_ge50": strat(e_share50)}
    for thr in ANT_THR:
        out[f"ANT_pup{int(round(100 * thr))}"] = strat(e_ant(thr))
    out["ANT_all"] = strat(e_ant_all)
    return out


# ------------------------------------------------ run all tickers
files = sorted(f for f in glob.glob(os.path.join(DATADIR, "*.csv"))
               if not os.path.basename(f).startswith("_"))
rows, skipped = [], 0
for i, f in enumerate(files, 1):
    tk = os.path.basename(f)[:-4]
    try:
        df = pd.read_csv(f, usecols=USECOLS)
    except (ValueError, KeyError):
        skipped += 1
        continue
    res = run_ticker(df)
    if res is None:
        skipped += 1
        continue
    b = res["buy_and_hold"]
    for name, m in res.items():
        rows.append({"ticker": tk, "variant": name,
                     "days_in_trade": m["days_in_trade"], "bh_days": b["days_in_trade"],
                     "exposure_pct": m["exposure_pct"], "round_trips": m["round_trips"],
                     "win_rate": m["win_rate"], "avg_trade_return": m["avg_trade_return"],
                     "med_days_per_trade": m["med_days_per_trade"],
                     "total_return": m["total_return"], "deployed_cagr": m["deployed_cagr"],
                     "edge_vs_bh": round(m["deployed_cagr"] - b["deployed_cagr"], 4),
                     "intrade_sharpe": m["intrade_sharpe"], "maxdd": m["maxdd"],
                     "bps_per_day": m["bps_per_day"],
                     "beat_bh_deployed_cagr": bool(m["deployed_cagr"] > b["deployed_cagr"]),
                     "beat_bh_bps": bool(m["bps_per_day"] > b["bps_per_day"]),
                     "beat_bh_intrade_sharpe": bool(m["intrade_sharpe"] > b["intrade_sharpe"])})
    if i % 100 == 0 or i == len(files):
        print(f"  [{i}/{len(files)}] processed")

detail = pd.DataFrame(rows)
detail.to_csv(DETAIL, index=False)
n_tickers = detail["ticker"].nunique() if len(detail) else 0


# ------------------------------------------------ summary: mean AND median per variant
def agg(col, d, fn):
    vals = [v for v in d[col].tolist() if isinstance(v, (int, float)) and v == v]
    return fn(vals) if vals else float("nan")


summ = []
for name in VARIANTS:
    d = detail[detail["variant"] == name]
    row = {"variant": name, "n_tickers": len(d),
           "total_days_in_trade": int(d["days_in_trade"].sum()) if len(d) else 0,
           "total_round_trips": int(d["round_trips"].sum()) if len(d) else 0}
    for col in ["days_in_trade", "exposure_pct", "deployed_cagr", "intrade_sharpe",
                "total_return", "bps_per_day", "maxdd", "round_trips", "win_rate",
                "med_days_per_trade", "avg_trade_return"]:
        row[f"median_{col}"] = round(agg(col, d, st.median), 4)
        row[f"mean_{col}"] = round(agg(col, d, st.mean), 4)
    for flag in ["beat_bh_deployed_cagr", "beat_bh_bps", "beat_bh_intrade_sharpe"]:
        row[f"pct_{flag}"] = round(100 * d[flag].mean(), 1) if len(d) else 0.0
    summ.append(row)
pd.DataFrame(summ).to_csv(SUMMARY, index=False)

# ------------------------------------------------ paired McNemar tests
piv = {}
for r in rows:
    piv.setdefault(r["ticker"], {})[r["variant"]] = r


def mcnemar(a, b, flag="beat_bh_deployed_cagr"):
    a_only = b_only = 0
    for d in piv.values():
        if a in d and b in d:
            A, B = bool(d[a][flag]), bool(d[b][flag])
            if A and not B:
                a_only += 1
            elif B and not A:
                b_only += 1
    m = a_only + b_only
    chi = (abs(a_only - b_only) - 1) ** 2 / m if m else 0.0
    p = math.erfc(math.sqrt(chi / 2)) if chi > 0 else 1.0
    return a_only, b_only, chi, p


paired = []
for a, b in KEY_PAIRS:
    a_only, b_only, chi, p = mcnemar(a, b)
    paired.append({"A": a, "B": b, "A_beats_only": a_only, "B_beats_only": b_only,
                   "chi2": round(chi, 3), "p_value": round(p, 5),
                   "winner": a if a_only > b_only else b,
                   "significant_05": bool(p < 0.05)})
pd.DataFrame(paired).to_csv(PAIRED, index=False)

# ------------------------------------------------ top performers (consistency-filtered)
bh_dc = {r["ticker"]: r["deployed_cagr"] for r in rows if r["variant"] == "buy_and_hold"}
top = []
for name in STRATS:
    for r in [r for r in rows if r["variant"] == name and r["round_trips"] >= MIN_TRADES_HL]:
        top.append(dict(r))
top.sort(key=lambda r: (r["edge_vs_bh"] if r["edge_vs_bh"] == r["edge_vs_bh"] else -9),
         reverse=True)
top_fields = ["ticker", "variant", "deployed_cagr", "edge_vs_bh", "bps_per_day",
              "intrade_sharpe", "win_rate", "round_trips", "med_days_per_trade",
              "days_in_trade", "bh_days", "total_return"]
with open(TOPPERF, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=top_fields)
    w.writeheader()
    for r in top[:40]:
        w.writerow({k: r.get(k, "") for k in top_fields})

# ------------------------------------------------ console
print(f"\ntested {n_tickers} tickers ({skipped} skipped)   GROSS (COST={COST}), no stop, "
      f"window {WIN_START.date()}+ (membership-gated)\n")
h = (f"{'variant':<15}{'medDays':>8}{'expo%':>7}{'medDeplCAGR':>12}{'meanDeplCAGR':>13}"
     f"{'medInShrp':>10}{'medBps/d':>9}{'medTotRet':>10}{'medWin%':>8}{'d/tr':>6}"
     f"{'medTrips':>9}{'totTrips':>9} | {'%beatCAGR':>10}{'%beatBps':>9}")
print(h)
for s in summ:
    mw = s["median_win_rate"]
    print(f"{s['variant']:<15}{s['median_days_in_trade']:>8.0f}{s['median_exposure_pct']:>7.1f}"
          f"{100*s['median_deployed_cagr']:>11.1f}%{100*s['mean_deployed_cagr']:>12.1f}%"
          f"{s['median_intrade_sharpe']:>10.3f}{s['median_bps_per_day']:>9.2f}"
          f"{100*s['median_total_return']:>9.1f}%{(100*mw if mw == mw else 0):>7.1f}%"
          f"{s['median_med_days_per_trade']:>6.0f}{s['median_round_trips']:>9.0f}"
          f"{s['total_round_trips']:>9} | {s['pct_beat_bh_deployed_cagr']:>9.1f}%"
          f"{s['pct_beat_bh_bps']:>8.1f}%")
print("\npaired McNemar (beat-own-B&H, deployed CAGR):")
for r in paired:
    sig = "**" if r["significant_05"] else "  "
    print(f"  {sig}{r['A']:>14} vs {r['B']:<15} {r['A_beats_only']:>3} / {r['B_beats_only']:<3} "
          f"chi2 {r['chi2']:>6.2f}  p {r['p_value']:.4f}  -> {r['winner']}")
print(f"\ntop 5 (>= {MIN_TRADES_HL} trades, by deployed-CAGR edge vs own B&H):")
for r in top[:5]:
    print(f"  {r['ticker']:<8}{r['variant']:<15} deplCAGR {100*r['deployed_cagr']:>6.1f}%  "
          f"edge {100*r['edge_vs_bh']:>+6.1f}%  trips {r['round_trips']:>3}  "
          f"win {(100*r['win_rate'] if isinstance(r['win_rate'], float) else 0):>4.0f}%")
print(f"\nwrote {DETAIL}\n      {SUMMARY}\n      {TOPPERF}\n      {PAIRED}")
