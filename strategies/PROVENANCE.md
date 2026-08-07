# Strategy provenance

Generation-time provenance for the 8 LLM-generated strategies in the paper. Each
record captures what the agent produced for that strategy: the mathematical
domain, its novelty claim, the frozen parameters, and the walk-forward **dev**
metric obtained during discovery.

**Scope and honesty note.** These are *provenance* records, not a full per-backtest
audit trail. The strategies were discovered by the agent on **development data
only** (test sets were withheld from the working directory), then frozen. The
dev metrics below are the in-development walk-forward figures the agent reported
during discovery — several are on BTC (the discovery instrument) — and are **not**
the paper's results. Every out-of-sample number in the paper was recomputed
independently by the authors' own vectorbt harness (`../validate.py` for Stage 1,
`../stage2_portfolio_ga/` for Stage 2), so the paper's results do not depend on
the agent's self-reported figures. Where a machine-written summary was retained
from the discovery run, the raw file is included under `provenance/`.

| # | Strategy | Domain | Discovery | Dev walk-forward metric | Raw record |
|---|---|---|---|---|---|
| 1 | Spectral Entropy | Information Theory | Pure LLM | Sharpe 1.34, 79% positive windows (BTC) | [`provenance/spectral_entropy.json`](provenance/spectral_entropy.json) |
| 2 | Rough Path Signatures | Stochastic Analysis | Pure LLM | Sharpe 1.08 (BTC daily, 2021+) | [`provenance/rough_path_signatures.json`](provenance/rough_path_signatures.json) |
| 3 | Information Geometry | Differential Geometry | Pure LLM | Sharpe 1.08, 58% positive windows (BTC) | [`provenance/information_geometry.json`](provenance/information_geometry.json) |
| 4 | Wasserstein Distance | Optimal Transport | Pure LLM | Sharpe 1.62, 64% positive windows (BTC) | [`provenance/wasserstein_distance.json`](provenance/wasserstein_distance.json) |
| 5 | Rate Distortion | Information Theory (Rate–Distortion) | LLM-generated | see `rate_distortion_complexity.py` docstring | docstring only |
| 6 | Fisher Information | Information Geometry / Statistical Estimation | LLM-generated | see `fisher_information_regime.py` docstring | docstring only |
| 7 | Volatility Phase Transition | Statistical Physics | LLM-generated | see `volatility_phase_transition.py` docstring | docstring only |
| 8 | Hurst Mean Reversion | Fractal Analysis | LLM-generated | Sharpe +0.84 crypto / +0.52 index (daily) | docstring only |

For strategies 5–8, the novelty claim, parameters, and dev validation are recorded
in the header docstring of the corresponding `.py` file in this directory rather
than in a separate JSON.

## Novelty claims (as stated by the agent at discovery)

- **Spectral Entropy** — TRUE spectral entropy from the FFT power spectrum with
  counter-intuitive logic: HIGH entropy = market inefficiency = mean reversion.
- **Rough Path Signatures** — first application of rough-path signatures (Lyons,
  stochastic analysis) to trading; uses the signed (Lévy) area of the
  (return, volatility) path.
- **Information Geometry** — Fisher–Rao metric to measure path curvature through
  the distribution manifold for regime detection.
- **Wasserstein Distance** — optimal-transport distance to detect regime changes
  via distribution shifts.
- **Rate Distortion** — rate–distortion (compressibility) of recent price action
  as a structure/noise signal.
- **Fisher Information** — trace of the Gaussian Fisher-information matrix (with a
  kurtosis adjustment) as a distributional-stability regime signal.
- **Volatility Phase Transition** — order/disorder phase framing with a
  critical-slowing (lag-1 autocorrelation) early-warning indicator.
- **Hurst Mean Reversion** — rescaled-range Hurst exponent to switch between
  trend-persistence and mean-reversion regimes.
