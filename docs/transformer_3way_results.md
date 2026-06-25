# Transformer vs XGBoost vs LSTM — 3-way bake-off

Feature-branch experiment (`feature/transformer`). A TensorFlow/Keras seq2seq
**transformer** (`src/loadforecast/models/transformer_quantile.py`) built to drop
into the existing scaffolding — same data windows, scaler, pinball loss, predict
wrappers, and dispatch harness. Sized for ~LSTM parameter parity
(`d_model=32, num_blocks=1, ff_dim=64` → ~47k params vs the LSTM's ~40k) so the
comparison is about architecture, not capacity. Same train/val/holdout splits,
same quantiles, raw model outputs (no conformal, no M10 clip).

Reproduce: `train_transformer_quantile.py`, `train_transformer_price_quantile.py`,
then `compare_transformer_3way_{load,price}.py`.

## Load — 70-day holdout (3-way tie)

| Model | P50 MAE (MW) | Skill vs TSO | Worst-10% MAE | 80% coverage |
|---|---|---|---|---|
| XGBoost (prod) | **388.5** | +21.1% | 815.1 | 70.0% |
| Transformer | 391.9 | +20.4% | **608.4** | 69.5% |
| LSTM | 393.2 | +20.1% | 608.2 | 78.3% |

All three within ~5 MW on average — a genuine tie. The transformer matches the
LSTM's tail robustness (worst-10% 608 ≈ 608, both well ahead of XGBoost's 815).

## Price — 61-day Mar–Apr 2026 holdout (XGBoost wins; Transformer beats LSTM)

| Model | P50 MAE (€/MWh) | Skill vs naive | 80% coverage | Battery P&L (% perfect-foresight) | Uplift vs naive |
|---|---|---|---|---|---|
| **XGBoost (prod)** | **17.74** | **+52.1%** | 71.4% | **96.9%** | **+€64.8k** |
| Transformer | 19.51 | +47.4% | 64.5% | 95.4% | +€58.8k |
| LSTM | 23.67 | +36.1% | 64.3% | 95.0% | +€57.0k |

The transformer closes ~70% of the LSTM→XGBoost MAE gap (23.7 → 19.5) and edges
the LSTM on dispatch P&L, but **XGBoost wins every metric**.

## Verdict & honest notes

- **Trees still win at this data scale** (~1.5–4k sequences) — consistent with the
  literature that transformers need far more data to beat gradient-boosted trees
  on tabular/calendar-driven forecasting. The transformer is a credible #2, not a
  flop: it ties on load and beats the LSTM on price.
- **Overfitting signal:** the price transformer's *validation* MAE was 11.8 €/MWh
  but *holdout* 19.5 — a large val→holdout gap. The neural models overfit the calm
  Jan–Feb validation regime more than XGBoost did (which held 17.7 on the volatile
  Mar–Apr holdout). Same regime-shift lesson as the conformal-calibration finding.
- **Decision:** keep on `feature/transformer` as a documented honest result. Do
  **not** promote — production stays XGBoost.
- **Where a transformer might actually win** (not tested here): the **72h
  multi-horizon** problem, where sequence models have a structural advantage over
  per-day tree models. That, not the 24h day-ahead bake-off, is where to point a
  transformer next.
