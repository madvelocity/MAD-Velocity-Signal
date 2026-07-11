# MAD-Velocity-Signal

**Is price doing the moving, or is the average catching up?**

Every moving-average-distance indicator conflates two motions: genuine price movement, and the
average catching up to price. This repository separates them, gates trades on the difference,
and measures the result on the capital actually deployed — against the S&P 500 as it actually
was, delisted members included.

Companion code for the paper **"MAD-Velocity: Constructing and Evaluating a Moving-Average-Distance
Trading Signal"** (Lawson Arrington, July 2026). <!-- add paper/LinkedIn link -->
Extends the [MAD-Markov Model](#) <!-- add MAD-Markov repo link --> from forecasting to trading.

---

## Key results

| Finding | Evidence |
|---|---|
| The **velocity gate** (enter only price-dominated trend re-crossings) is the one refinement that survives a paired significance test | 59.8% of 672 point-in-time constituents beat their own buy-and-hold vs. 55.8% ungated (McNemar p = 0.021) |
| The **anticipatory entry** (buy the −1 state before the re-cross confirms) beats SPY buy-and-hold on every deployed measure over three decades | 13.4% vs. 10.9% deployed CAGR, in-trade Sharpe 0.74 vs. 0.65, at 69% exposure with a shallower max drawdown |
| The calibrated transition forecast is **redundant for trading** — its value is exhausted by naming the state | Unfiltered control ties the forecast-filtered variant 14–14 |
| A regime transition's composition **inverts with position in the band** | Deep-oversold "improvements": ~65% moving-average artifact. Trend re-crossings: ~65% genuine price movement |

Full tables, significance tests, and limitations are in the paper. Everything in `results/` is
exactly what the paper cites.

## The pipeline

| Script | Output | Role |
|---|---|---|
| `01_spy.py` | `results/01_spy_*.csv` | SPY history, indicator stack, walk-forward P(up), velocity decomposition, index backtest |
| `02_SP500_constituents.py` | `results/02_sp500_constituents.csv` | Point-in-time S&P 500 membership 2016–2025, freshness-guarded |
| `03_data.py` | `data/raw/<TICKER>.csv` | 20 years of daily bars per constituent (polygon.io), delisted names and predecessor symbols included |
| `04_data_prep.py` | `data/<TICKER>.csv` | Per-ticker indicators, membership flags, coverage audit (resumable) |
| `05_backtest.py` | `results/05_*.csv` | Cross-sectional backtest, paired McNemar tests, top performers |

## Reproducing

**The index study (free, ~2 minutes):**

```bash
pip install pandas numpy yfinance
python3 01_spy.py
```

Reproduces the SPY results (Tables IV–V of the paper) from free data. No key required.

**The cross-section (requires a polygon.io API key):**

```bash
export POLYGON_API_KEY=yourkey
python3 02_SP500_constituents.py   # point-in-time membership
python3 03_data.py                 # ~5 min on a paid plan; ~2.5 h on the free tier (5 req/min, auto-retries)
python3 04_data_prep.py            # indicator stack per ticker (resumable if interrupted)
python3 05_backtest.py             # the cross-sectional results (Tables VI–VII)
```

**Price data is not redistributed** (provider license). `data/` is gitignored; anyone with a
polygon.io key — the free tier works — can rebuild it exactly with the commands above.

## Method notes

- **Strictly causal.** Every indicator is trailing-only; signals evaluated at one close are
  executed at the next open. The walk-forward forecast uses only regime visits that resolved
  before the current day.
- **Exposure-adjusted.** Performance is measured per day of capital deployed
  (`deployed CAGR = wealth^(252/days_in_trade) − 1`, in-trade Sharpe, bps per deployed day),
  never against a fully invested calendar — a low-exposure rule is neither flattered nor
  penalized for its time in cash. Trade-consistency stats are reported so thin-exposure
  annualization artifacts can't masquerade as edge.
- **Survivorship-free.** Point-in-time membership; a name can only be held while it was
  actually in the index. Delisted members (Silicon Valley Bank, First Republic) are included.
  Fourteen renamed lineages (PCLN→BKNG, DPS→KDP, …) are stitched via predecessor symbols;
  coverage is self-audited at 99.2% of member-days.
- **Disclosed conventions.** Gross of transaction costs; cross-sectional prices are
  split-adjusted without dividend reinvestment (quantified in the paper, §V.D); the one
  residual coverage gap (Arconic, 77.5%) is documented rather than dropped.

## Variant set

| Paper name | Code name | Entry |
|---|---|---|
| Buy-and-hold | `buy_and_hold` | hold every member-day (benchmark) |
| Confirmed re-cross | `R1_confirmed` | regime crosses −1 → +1 |
| Velocity-gated re-cross | `R1_share_ge50` | confirmed cross **and** price share s ≥ 0.50 |
| Anticipatory (θ) | `ANT_pup50/60/70` | state −1 and P(up) ≥ θ |
| Anticipatory (unfiltered) | `ANT_all` | state −1, forecast ignored (control) |

All variants share one exit (regime rolls back +2 → +1 → out at next open), long/flat, one
position, no leverage, no stop.

## Disclaimer

Independent, self-directed research. Not investment advice; past performance does not indicate
future results. Results are gross of transaction costs, on a historical universe, under stated
conventions — read §VII of the paper before drawing conclusions.
