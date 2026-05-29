"""Smoke test: assert both the load and price models can produce a
non-NaN tomorrow forecast against the current parquet.

Used by the daily GitHub Action after `data.refresh` to catch silent
breakage between the data layer and the trained checkpoints (e.g. a
schema column rename, a missing source, a model that expects different
decoder dims).

Exits 0 if both models produce a non-NaN P50 for tomorrow, 1 otherwise.
The price-model check is allowed to fail in degraded mode (VRE not yet
published) since that's the production-expected behaviour pre-12:30.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from loadforecast.backtest import issue_time_for, load_smard_15min
from loadforecast.models.predict import (
    xgboost_load_predict_full,
    xgboost_price_predict_full,
)

PARQUET = "smard_merged_15min.parquet"


def main() -> int:
    df = load_smard_15min(PARQUET)
    tomorrow = datetime.now(ZoneInfo("Europe/Berlin")).date() + timedelta(days=1)
    issue = issue_time_for(tomorrow)
    print(f"Smoke: delivery {tomorrow}, issue {issue}, parquet max {df.index.max()}")

    failures: list[str] = []

    # --- Load model -----------------------------------------------------
    try:
        load_fc = xgboost_load_predict_full(df, issue)
        n_nan = int(load_fc["p50"].isna().sum())
        if n_nan > 0:
            failures.append(f"load: P50 has {n_nan}/96 NaN")
        else:
            print(f"  load:  P50 range {load_fc['p50'].min():.0f} -> "
                  f"{load_fc['p50'].max():.0f} MW   OK")
    except Exception as e:  # noqa: BLE001
        failures.append(f"load: {type(e).__name__}: {e}")

    # --- Price model ----------------------------------------------------
    try:
        price_fc = xgboost_price_predict_full(df, issue)
        n_nan = int(price_fc["p50"].isna().sum())
        if n_nan > 0:
            failures.append(f"price: P50 has {n_nan}/96 NaN")
        else:
            print(f"  price: P50 range {price_fc['p50'].min():.1f} -> "
                  f"{price_fc['p50'].max():.1f} EUR/MWh   OK")
    except Exception as e:  # noqa: BLE001
        failures.append(f"price: {type(e).__name__}: {e}")

    if failures:
        print("\nSmoke FAILED:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nSmoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
