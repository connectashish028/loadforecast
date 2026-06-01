"""Targeted backfill of the intraday-continuous price column into the
existing parquet, without re-fetching the ~30 other columns.

Phase B.4 step 1: SMARD filter 252 (DE-LU, quarter-hour) carries the
volume-weighted average of all intraday continuous trades per 15-min
delivery slot. Empirically verified vs day-ahead (filter 4169) on a
multi-day window: correlation 0.977, mean basis -1.15 EUR/MWh,
std 14.23 EUR/MWh -- the textbook intraday-vs-DA signature.

The full refresh.py would re-fetch every column from 2022-01-01, which
is several minutes of network + disk. This script only fetches the new
column and merges it in. Run once after schema.py registers the column;
afterwards, the daily refresh handles ongoing updates automatically.

Run from repo root:
    PYTHONPATH=src python scripts/backfill_intraday_continuous.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from loadforecast.data.schema import COLUMN_BY_NAME
from loadforecast.data.sources import smard_api

PARQUET = Path("smard_merged_15min.parquet")
COL = "price__intraday_continuous_de_lu"


def main() -> None:
    if not PARQUET.exists():
        raise SystemExit(f"Parquet not found: {PARQUET}")

    existing = pd.read_parquet(PARQUET)
    print(f"Existing parquet: {len(existing):,} rows  "
          f"x {existing.shape[1]} cols  "
          f"({existing.index.min()} -> {existing.index.max()})")
    if COL in existing.columns:
        nan_pct = existing[COL].isna().mean() * 100
        print(f"  {COL} already present ({nan_pct:.0f} % NaN). Refreshing.")
    else:
        print(f"  {COL} not yet in parquet. Adding.")

    column = COLUMN_BY_NAME[COL]
    start = existing.index.min()
    end = existing.index.max() + pd.Timedelta(minutes=15)
    print(f"Fetching SMARD filter {column.fetch_kwargs['filter_id']} "
          f"({column.fetch_kwargs['region']}) for {start} -> {end} ...")
    series = smard_api.fetch(column, start, end)
    print(f"  fetched {len(series):,} rows. "
          f"range [{series.min():.2f}, {series.max():.2f}] EUR/MWh, "
          f"mean {series.mean():.2f}")

    # Reindex to the parquet's exact 15-min UTC grid so the merge is clean.
    aligned = series.reindex(existing.index)
    nan_pct = aligned.isna().mean() * 100
    print(f"  aligned to parquet index: {nan_pct:.1f} % NaN")

    # Sanity vs day-ahead on the overlap.
    da_col = "price__germany_luxembourg"
    if da_col in existing.columns:
        joint = pd.DataFrame({"da": existing[da_col], "id": aligned}).dropna()
        if not joint.empty:
            corr = joint["da"].corr(joint["id"])
            basis_mean = (joint["id"] - joint["da"]).mean()
            basis_std = (joint["id"] - joint["da"]).std()
            print(f"  vs day-ahead: corr={corr:.4f}, "
                  f"basis_mean={basis_mean:+.2f}, basis_std={basis_std:.2f} "
                  f"(over {len(joint):,} aligned rows)")

    out = existing.copy()
    out[COL] = aligned

    tmp = PARQUET.with_suffix(".parquet.tmp")
    out.to_parquet(tmp)
    tmp.replace(PARQUET)
    print(f"Wrote {PARQUET}  rows={len(out):,}  cols={out.shape[1]}")


if __name__ == "__main__":
    main()
