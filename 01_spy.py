#!/usr/bin/env python3
"""
01_spy.py  --  consolidated SPY pipeline: data + indicator + forecast + entry-timing backtest.

Self-contained consolidation of everything validated this week (old_2/01,02,07). Pulls the
FULL SPY history, builds the MAD indicator stack, the walk-forward P(up) forecast, the
price-vs-MA decomposition, then backtests the entry-timing variants head to head.

DATA
  yfinance, period="max", auto_adjust=True (split+dividend adjusted; total-return prices).
  MAD  = 100*(C - SMA20)/C          SMA20 = 20-day simple mean of Close
  z    = MAD / sigma                 sigma = 255-day rolling std of MAD
  regime in {-3,-2,-1,+1,+2,+3} at z thresholds 0, +/-1, +/-2 (no zero state)
  P(up) = walk-forward calibrated probability the next regime transition is up
          (logistic on z for interior regimes, trailing base rate otherwise; only visits
          that RESOLVED before day t feed day t -- strictly causal)
  price_share_5 = signed share of the 5-day MAD change attributable to price (not the MA),
          from dMAD ~ (100*SMA/C^2)*dC - (100/C)*dSMA;  in [-1, +1]

VARIANTS (shared exit; long/flat; one position; next-open execution; GROSS, no stop)
  buy_and_hold    hold every day from the first valid regime day
  R1_confirmed    enter after a -1 -> +1 cross (the late, confirmed entry)
  R1_share_ge50   confirmed cross AND price_share_5 >= 0.50 (price-dominant crossings)
  ANT_pup50/60/70 ANTICIPATORY: enter AT the -1 state when P(up) >= 0.50/0.60/0.70
  ANT_all         control: enter at EVERY -1 state day with a P(up) available (forecast
                  ignored) -- isolates whether the forecast adds anything beyond the state
  Exit (all): regime rolls back +2 -> +1  ->  exit next open.

MEASUREMENT (exposure-adjusted -- deployed days, not the full calendar)
  deployed_cagr   wealth ^ (252 / days_in_trade) - 1     annualized over DEPLOYED time only
  intrade_sharpe  sqrt(252) * mean / std (ddof=1) of IN-POSITION daily returns, rf = 0
  bps_per_day     10000 * ln(wealth) / days_in_trade     mean daily log-return while deployed
  days/trade      per completed trade, entry day through exit day inclusive (median & mean)
  plus: exposure %, round trips, win rate, mean & median trade return, best/worst trade,
  max drawdown of the strategy equity curve, total return.

    pip install yfinance pandas numpy
    python3 01_spy.py
Outputs -> results/01_spy_data.csv      daily dataset (indicator + forecast + decomposition)
           results/01_spy_trades.csv    every completed trade, all variants
           results/01_spy_summary.csv   one row of metrics per variant
"""
import os

import numpy as np
import pandas as pd
import yfinance as yf

OUTDIR = "results"
OUT_DATA = os.path.join(OUTDIR, "01_spy_data.csv")
OUT_TRADES = os.path.join(OUTDIR, "01_spy_trades.csv")
OUT_SUMMARY = os.path.join(OUTDIR, "01_spy_summary.csv")
TICKER = "SPY"
SMA_W, VOL_W = 20, 255
K = 5                                    # decomposition horizon (trading days)
SHARE_MIN = 0.50                         # price-dominance gate
ANT_THR = [0.50, 0.60, 0.70]             # anticipatory P(up) ladder
ORDER = [-3, -2, -1, 1, 2, 3]
RANK = {r: i for i, r in enumerate(ORDER)}
INTERIOR = {-2, -1, 1, 2}
WINDOW_OBS, MIN_N = 500, 60              # walk-forward pool window / min sample
PCLIP = (0.02, 0.98)
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


# ================================================================ data + indicator stack
data = yf.download(TICKER, period="max", auto_adjust=True, progress=False)
if isinstance(data.columns, pd.MultiIndex):
    data = data.droplevel(1, axis=1)
data = data[["Open", "Close"]].dropna()
dates = pd.to_datetime(data.index)
O, C = data["Open"].to_numpy(float), data["Close"].to_numpy(float)
n = len(C)

s = pd.Series(C, index=dates)
sma = s.rolling(SMA_W).mean()
mad = 100 * (s - sma) / s
sigma = mad.rolling(VOL_W).std()
z = (mad / sigma).to_numpy()
regime = np.select([z > 2, z > 1, z > 0, z > -1, z > -2],
                   [3, 2, 1, -1, -2], default=-3).astype(float)
regime[np.isnan(z)] = np.nan

# ---------------------------------------------------------------- walk-forward P(up)
pool = {r: [] for r in ORDER}
fc = {}
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
        fc[d["idx"]] = {"days_in": d["days_in"], "p_up": round(model_p, 4),
                        "base_p_up": round(base_p, 4)}
    if v["exit"] is not None:
        up = 1 if v["exit"] == "up" else 0
        pool[r].extend((dd["z"], up) for dd in v["days"])
pup = np.array([fc.get(i, {}).get("p_up", np.nan) for i in range(n)], float)

# ---------------------------------------------------------------- price-vs-MA decomposition
dP = s - s.shift(K)
dSMA = sma - sma.shift(K)
price_c = (100.0 * (sma / s**2) * dP).to_numpy()     # price part of the K-day MAD change
sma_c = (-100.0 * (1.0 / s) * dSMA).to_numpy()       # MA part (>0 when the MA is falling)
denom = np.abs(price_c) + np.abs(sma_c)
pshare = np.where(denom > 0, price_c / denom, np.nan)

# ---------------------------------------------------------------- signals (strictly trailing)
sig_cross_up = np.zeros(n, bool)                     # -1 -> +1 confirmed cross
sig_rollback = np.zeros(n, bool)                     # +2 -> +1 (shared exit)
for t in range(1, n):
    if np.isnan(regime[t]) or np.isnan(regime[t - 1]):
        continue
    if regime[t] == 1 and regime[t - 1] == -1:
        sig_cross_up[t] = True
    if regime[t] == 1 and regime[t - 1] == 2:
        sig_rollback[t] = True

valid = ~np.isnan(regime)
start = int(np.argmax(valid))                        # first valid regime day


# ================================================================ engine + metrics
def backtest(entry_ok):
    """Long/flat, one position, next-open execution. entry_ok(t): enter at O[t]?
    (signal evaluated at t-1 close). Exit: rollback at t-1 -> out at O[t].
    Every day a position is open counts as a deployed day, exit day included."""
    rets, intrade, trades = [], [], []
    pos, eq, tstart, dit = 0, 1.0, 1.0, 0
    entry_i, days_this = None, 0
    for t in range(start + 1, n):
        in_pos, entered, exited = False, False, False
        if pos == 1:
            in_pos = True
            dit += 1
            days_this += 1
            if sig_rollback[t - 1]:
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
                pos, entry_i = 1, t
            else:
                r = 0.0
        if entered:
            tstart = eq
        eq *= (1 + r)
        rets.append(r)
        if in_pos:
            intrade.append(r)
        if exited:
            trades.append({"entry_date": dates[entry_i].strftime("%Y-%m-%d"),
                           "exit_date": dates[t].strftime("%Y-%m-%d"),
                           "days_held": days_this,
                           "entry_open": round(O[entry_i], 4), "exit_open": round(O[t], 4),
                           "trade_return": round(eq / tstart - 1, 6)})
    return rets, intrade, trades, dit


def metrics(rets, intrade, trades, dit):
    r = np.asarray(rets, float)
    eq = np.cumprod(1 + r) if len(r) else np.array([1.0])
    wealth = float(eq[-1])
    total_days = len(r)
    deployed_cagr = wealth ** (252.0 / dit) - 1 if dit > 0 else float("nan")
    ir = np.asarray(intrade, float)
    sharpe = (np.sqrt(252) * ir.mean() / ir.std(ddof=1)
              if len(ir) > 1 and ir.std(ddof=1) > 0 else float("nan"))
    bps = 10000 * np.log(wealth) / dit if dit > 0 else float("nan")
    dd = eq / np.maximum.accumulate(eq) - 1
    tr = np.array([x["trade_return"] for x in trades], float)
    dh = np.array([x["days_held"] for x in trades], float)
    return {"days_in_trade": dit, "total_days": total_days,
            "exposure_pct": round(100 * dit / total_days, 1) if total_days else 0.0,
            "round_trips": len(trades),
            "total_return": round(wealth - 1, 4),
            "deployed_cagr": round(deployed_cagr, 4),
            "intrade_sharpe": round(sharpe, 3),
            "bps_per_day": round(bps, 2),
            "med_days_per_trade": (float(np.median(dh)) if len(dh) else float("nan")),
            "mean_days_per_trade": (round(float(dh.mean()), 1) if len(dh) else float("nan")),
            "max_days_per_trade": (int(dh.max()) if len(dh) else 0),
            "win_rate": (round(float((tr > 0).mean()), 3) if len(tr) else float("nan")),
            "mean_trade_return": (round(float(tr.mean()), 4) if len(tr) else float("nan")),
            "med_trade_return": (round(float(np.median(tr)), 4) if len(tr) else float("nan")),
            "best_trade": (round(float(tr.max()), 4) if len(tr) else float("nan")),
            "worst_trade": (round(float(tr.min()), 4) if len(tr) else float("nan")),
            "maxdd": round(float(dd.min()), 4)}


# ---------------------------------------------------------------- entry predicates
def e_confirmed(t):
    return sig_cross_up[t - 1]


def e_share50(t):
    g = pshare[t - 1]
    return sig_cross_up[t - 1] and g == g and g >= SHARE_MIN


def e_ant(thr):
    def f(t):
        return regime[t - 1] == -1 and pup[t - 1] == pup[t - 1] and pup[t - 1] >= thr
    return f


def e_ant_all(t):                       # control: state only, forecast available but ignored
    return regime[t - 1] == -1 and pup[t - 1] == pup[t - 1]


# ---------------------------------------------------------------- run all variants
runs = {}
bh_rets = [C[t] / C[t - 1] - 1 for t in range(start + 1, n)]
runs["buy_and_hold"] = metrics(bh_rets, bh_rets, [], len(bh_rets))
all_trades = []
specs = ([("R1_confirmed", e_confirmed), ("R1_share_ge50", e_share50)]
         + [(f"ANT_pup{int(round(100 * thr))}", e_ant(thr)) for thr in ANT_THR]
         + [("ANT_all", e_ant_all)])
for name, pred in specs:
    rets, intrade, trades, dit = backtest(pred)
    runs[name] = metrics(rets, intrade, trades, dit)
    for x in trades:
        all_trades.append({"variant": name, **x})

# ================================================================ outputs
out = pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"),
                    "Open": np.round(O, 4), "Close": np.round(C, 4),
                    "sma_20": sma.to_numpy().round(4), "mad": mad.to_numpy().round(4),
                    "sigma": sigma.to_numpy().round(4), "z": np.round(z, 4)})
out["regime"] = [int(x) if x == x else np.nan for x in regime]
out["days_in"] = [fc.get(i, {}).get("days_in") for i in range(n)]
out["p_up"] = [fc.get(i, {}).get("p_up") for i in range(n)]
out["base_p_up"] = [fc.get(i, {}).get("base_p_up") for i in range(n)]
out[f"price_vel_{K}"] = (s / s.shift(K) - 1.0).to_numpy().round(6)
out[f"dmad_{K}"] = (mad - mad.shift(K)).to_numpy().round(4)
out[f"mad_price_contrib_{K}"] = np.round(price_c, 4)
out[f"mad_sma_contrib_{K}"] = np.round(sma_c, 4)
out[f"price_share_{K}"] = np.round(pshare, 4)
out["sig_cross_up"] = sig_cross_up
out["sig_rollback"] = sig_rollback
out.to_csv(OUT_DATA, index=False)

pd.DataFrame(all_trades).to_csv(OUT_TRADES, index=False)

summary_rows = [{"variant": k, **v} for k, v in runs.items()]
pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False)

# ---------------------------------------------------------------- console
print(f"{TICKER} {dates[0].strftime('%Y-%m-%d')} -> {dates[-1].strftime('%Y-%m-%d')}   "
      f"{n} rows   backtest starts {dates[start].strftime('%Y-%m-%d')} (first valid regime)")
print(f"GROSS, no stop, shared +2->+1 rollback exit, next-open execution\n")
h = (f"{'variant':<15}{'days':>7}{'expo%':>7}{'trips':>6}{'totRet':>9}{'deplCAGR':>9}"
     f"{'Sharpe':>8}{'bps/d':>7}{'d/tr med':>9}{'d/tr avg':>9}{'win%':>7}{'avgTr':>8}"
     f"{'medTr':>8}{'maxDD':>8}")
print(h)
for k, m in runs.items():
    print(f"{k:<15}{m['days_in_trade']:>7}{m['exposure_pct']:>7.1f}{m['round_trips']:>6}"
          f"{100*m['total_return']:>8.1f}%{100*m['deployed_cagr']:>8.2f}%"
          f"{m['intrade_sharpe']:>8.3f}{m['bps_per_day']:>7.2f}"
          f"{m['med_days_per_trade']:>9.0f}{m['mean_days_per_trade']:>9.1f}"
          f"{(100*m['win_rate'] if m['win_rate'] == m['win_rate'] else 0):>6.1f}%"
          f"{(100*m['mean_trade_return'] if m['mean_trade_return'] == m['mean_trade_return'] else 0):>7.2f}%"
          f"{(100*m['med_trade_return'] if m['med_trade_return'] == m['med_trade_return'] else 0):>7.2f}%"
          f"{100*m['maxdd']:>7.1f}%")
print(f"""
formulas: deployed_cagr = wealth^(252/days_in_trade) - 1        (annualized over DEPLOYED days)
          intrade_sharpe = sqrt(252)*mean/std(ddof=1) of in-position daily returns, rf=0
          bps_per_day    = 10000*ln(wealth)/days_in_trade
          days/trade counts entry day through exit day inclusive
ANT_all = every -1 day with a P(up) available, forecast IGNORED -> if it matches ANT_pup50,
the forecast is redundant and the -1 state itself carries the entry-timing alpha.
wrote {OUT_DATA}, {OUT_TRADES}, {OUT_SUMMARY}""")
