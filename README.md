# Statistical Arbitrage and Pairs Trading

Independent quantitative research project for testing whether related U.S. equities exhibit stable, mean-reverting relationships after transaction costs.

## Research question

Can a systematic strategy identify pairs of liquid U.S. equities whose relative price relationship mean-reverts out of sample?

## Research design

- Daily adjusted prices from January 2018 through December 2025.
- Formation period: January 2018 through December 2022.
- Out-of-sample test period: January 2023 through December 2025.
- Candidate universe: 30 liquid U.S. equities grouped by related industries.
- Correlation screen: formation-period daily log-return correlation of at least 0.70.
- Pair test: OLS hedge-ratio estimation followed by an Engle–Granger residual stationarity test.
- Signal: 60-day rolling spread z-score.
- Entry: z-score below -2.0 or above +2.0.
- Exit: z-score returns inside +/-0.5.
- Execution: signals are held from the next trading observation.
- Transaction costs: 5 basis points per dollar of gross exposure traded per leg event.
- Pair portfolio: equal-weighted net returns across selected, non-overlapping pairs.

## Run

The code is intentionally readable and can run with NumPy and Pandas. Optional packages in `requirements.txt` provide formal Statsmodels p-values and plotting support.

```powershell
python run_research.py
```

The script writes the following to `results/`:

- `selected_pairs.csv` — formation-period screening and test statistics.
- `pair_metrics.csv` — out-of-sample metrics for each selected pair.
- `portfolio_metrics.csv` — equal-weighted portfolio metrics.
- `pair_trades.csv` — trade-level results.
- `portfolio_daily_returns.csv` — daily net portfolio returns and equity.

## Methodology

Correlation is used only to reduce the search space. It is not treated as proof of a tradable relationship. The hedge ratio is estimated using formation-period log prices, and the residual spread is tested for stationarity. Pair selection and all model parameters are fixed before the out-of-sample period.

The fallback ADF implementation reports the residual test statistic and uses a conservative approximate Engle–Granger 5% critical value when Statsmodels is unavailable. For a final research report, install the optional dependencies and report the formal p-values alongside the test statistic.

## Limitations

This is a research backtest, not investment advice. The project does not yet model borrow fees, short-sale constraints, corporate actions beyond adjusted prices, delisted securities, market impact, or live execution. The universe is also subject to survivorship bias because it is defined from currently known liquid names rather than a point-in-time historical constituent database.
