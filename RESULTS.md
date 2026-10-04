# /Research Results

This is the first reproducible run of the project using formation-period selection and a 2023–2025 out-of-sample test.

## Selected pairs

| Pair | Group | Formation correlation | Residual test statistic |
|---|---|---:|---:|
| V / MA | Payments | 0.918 | -4.740 |
| CAT / DE | Industrials | 0.754 | -3.724 |
| KO / PEP | Consumer | 0.755 | -3.356 |

The run selected non-overlapping pairs only. The fallback implementation used an approximate Engle–Granger 5% critical value of -3.34 because Statsmodels was not available in the bundled runtime. Formal p-values should be generated after installing the optional requirements and rerunning the project.

## Out-of-sample pair performance

| Pair | Annualized return | Sharpe ratio | Maximum drawdown | Trades | Win rate |
|---|---:|---:|---:|---:|---:|
| V / MA | 1.21% | 0.38 | -5.52% | 15 | 86.67% |
| CAT / DE | -12.23% | -0.97 | -40.72% | 12 | 66.67% |
| KO / PEP | -4.12% | -0.68 | -20.26% | 13 | 69.23% |

## Equal-weighted portfolio

| Metric | Result |
|---|---:|
| Total return | -14.17% |
| Annualized return | -4.99% |
| Annualized volatility | 4.81% |
| Sharpe ratio | -1.04 |
| Maximum drawdown | -18.47% |

## Research conclusion

The formation-period tests found several spreads with evidence of stationarity, but that statistical property did not translate into a profitable equal-weighted portfolio out of sample after transaction costs. The initial result therefore does not support the hypothesis that this simple specification produced a reliable trading edge during 2023–2025.

The next research questions are whether the result is sensitive to the approximate test, entry and exit thresholds, the fixed hedge ratio, transaction-cost assumptions, and the selected universe. The negative result should remain visible in the research log; the project should not be tuned until it produces a favorable Sharpe ratio.
