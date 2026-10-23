# BTC SMA120 Trend Filter (Long-Only)

Research on a simple trend filter for BTC: hold BTC when the daily close is above
the 120-day moving average (SMA120), otherwise hold cash.

**Conclusion:** the rule cuts maximum drawdown by roughly half and passed a single
out-of-sample test with criteria locked in advance. However, the return advantage
over buy&hold almost disappears out-of-sample. The benefit is risk reduction,
not extra return.

## Key results (BTCUSDT spot, daily, fee 0.10% per side)

| Metric | Buy&hold OOS (2024-26) | SMA120 OOS (2024-26) | SMA120 in-sample (2018-23) |
|---|---|---|---|
| CAGR | 28.1% | 31.9% | 75.5% |
| Max drawdown | -53.0% | -28.4% | -32.3% |
| Calmar | 0.53 | 1.12 | 2.34 |
| Time in market | 100% | 58% | 52% |

- All 5 pre-registered OOS criteria passed, including 5 of 5 neighboring SMA lengths.
- Plan with OOS numbers, not in-sample: the in-sample figures are optimistic
  (SMA120 was selected from 100+ configurations on the same data).
- At fee 0.30% per side the CAGR advantage disappears; the drawdown advantage remains.
- 13 of 17 OOS trades were small losses (whipsaw).

## Structure
```
reports/   full research reports (EN / ID)
scripts/   research scripts
```

## Scripts
| File | Purpose |
|---|---|
| `download_btc_spot.py` | Download daily BTCUSDT spot klines and convert to CSV |
| `explore_behavior.py` | Loader and cross-sectional behavior analysis (used as a library) |
| `btc_mdd_reduction.py` | Comparison of 65 MDD-reduction configurations |
| `btc_sma_robust.py` | SMA 80-160 robustness map with 12 filter variants |
| `btc_oos_sma120.py` | Single OOS test with locked criteria |

## How to reproduce
1. Run `download_btc_spot.py` to fetch the data (raw data is not included in this repo).
2. Run `btc_sma_robust.py` for the in-sample robustness map.
3. `btc_oos_sma120.py` is the one-shot OOS test; the OOS period has already been used.

## Methodology
Signal at the close of day t, position from the open of day t+1 (no lookahead).
Smoothed Calmar (average of three adjacent lengths) used to pick L instead of the raw peak.
Each extra filter had to beat the plain SMA by at least 0.15 in both sample halves.

## Disclaimer
Research only, not financial advice. Results come from backtests and do not
guarantee future performance.

Author: Raffi Arganta Viari
