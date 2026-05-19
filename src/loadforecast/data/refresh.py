"""Refresh the SMARD/EnergyCharts/ENTSO-E parquet idempotently.

Single source of truth: `schema.COLUMNS` — every column is declared once,
with its source. This module iterates the registry, fetches each column
for the requested window, and writes/updates `smard_merged_15min.parquet`.

Usage
-----
    python -m loadforecast.data.refresh                         # extend through "now"
    python -m loadforecast.data.refresh --through 2026-05-04    # extend through a date
    python -m loadforecast.data.refresh --rebuild --start 2022-01-01 --through 2026-05-04
                                                                # full rebuild

Behaviour
---------
- Reads the existing parquet (if any), takes its last timestamp.
- Fetches the missing window from each registered column's source.
- Joins on a canonical 15-min UTC index.
- Writes the parquet back, atomically.
- Printable progress per column. Exit code 0 if all required columns
  were updated, 1 if any required column failed.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .schema import (
    COLUMNS,
    SRC_ENERGY_CHARTS,
    SRC_ENTSOE,
    SRC_OPEN_METEO,
    SRC_SMARD_API,
    SRC_SMARD_DOWNLOADCENTER,
    Column,
)

DEFAULT_PARQUET = Path("smard_merged_15min.parquet")
DEFAULT_START = pd.Timestamp("2022-01-01", tz="UTC")
SOURCES = (
    SRC_ENERGY_CHARTS, SRC_SMARD_API, SRC_SMARD_DOWNLOADCENTER,
    SRC_OPEN_METEO, SRC_ENTSOE,
)

# Columns we *must* have (M4 dataset breaks without them).
REQUIRED = {
    "actual_cons__grid_load",
    "fc_cons__grid_load",
    "price__germany_luxembourg",
}


def _load_source(source_name: str):
    """Lazy import so missing creds for one source don't break the other."""
    if source_name == SRC_ENERGY_CHARTS:
        from .sources import energy_charts
        return energy_charts
    if source_name == SRC_SMARD_API:
        from .sources import smard_api
        return smard_api
    if source_name == SRC_SMARD_DOWNLOADCENTER:
        from .sources import smard_downloadcenter
        return smard_downloadcenter
    if source_name == SRC_ENTSOE:
        from .sources import entsoe
        return entsoe
    if source_name == SRC_OPEN_METEO:
        from .sources import open_meteo
        return open_meteo
    raise ValueError(f"unknown source {source_name}")


def fetch_column(column: Column, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    src = _load_source(column.source)
    return src.fetch(column, start, end)


def _existing_end(parquet_path: Path) -> pd.Timestamp | None:
    if not parquet_path.exists():
        return None
    # `columns=[]` returns a 0-column frame which `.empty` flags True even
    # when there are millions of index rows. Check the index directly.
    df = pd.read_parquet(parquet_path, columns=[])
    if len(df.index) == 0:
        return None
    return df.index.max()


def refresh(
    parquet_path: Path = DEFAULT_PARQUET,
    *,
    start: pd.Timestamp | None = None,
    through: pd.Timestamp | None = None,
    rebuild: bool = False,
) -> dict:
    """Fetch all registered columns for [start, through), merge with existing
    parquet, and write back.

    Returns a dict with keys: rows, columns, errors (list of column names
    that failed), parquet_path.
    """
    if through is None:
        # Push 2 days past "now" so day-ahead publications (TSO forecast,
        # weather NWP) for tomorrow's delivery day are captured. Sources
        # return NaN for columns that don't extend into the future
        # (e.g. actual load), which is exactly what we want.
        through = (
            pd.Timestamp(datetime.now(UTC)).floor("15min")
            + pd.Timedelta(days=2)
        )
    if start is None:
        if rebuild or not parquet_path.exists():
            start = DEFAULT_START
        else:
            existing_end = _existing_end(parquet_path)
            if existing_end is None:
                start = DEFAULT_START
            else:
                # Re-fetch the last 72h to absorb late corrections — but
                # never start in the future. The parquet's index can
                # legitimately extend past "now" (TSO publishes day-ahead
                # forecasts for tomorrow), so a naive `existing_end - 72h`
                # can leave every source with nothing to fetch.
                # 72h chosen because SMARD occasionally publishes
                # corrections 1-2 days after the original publication
                # (one TSO's feed dropping out gets retroactively patched);
                # the 24h window we used before missed those.
                now = pd.Timestamp(datetime.now(UTC)).floor("15min")
                start = (min(existing_end, now) - pd.Timedelta(hours=72)).floor("15min")

    existing_rows_before = (
        len(pd.read_parquet(parquet_path, columns=[]).index)
        if parquet_path.exists() else 0
    )
    print(f"Refresh window: {start}  ->  {through}")
    print(f"{len(COLUMNS)} columns across {len(SOURCES)} sources.\n")

    series_by_col: dict[str, pd.Series] = {}
    errors: list[str] = []
    for col in tqdm(COLUMNS, desc="fetching", unit="col"):
        try:
            s = fetch_column(col, start, through)
            series_by_col[col.name] = s
        except Exception as e:  # noqa: BLE001
            errors.append(col.name)
            tqdm.write(f"  ✗ {col.name}  ({col.source}): {e}")

    # Merge into a single 15-min frame.
    if not series_by_col:
        raise RuntimeError("No columns fetched successfully.")

    fresh_df = pd.concat(series_by_col.values(), axis=1)
    if fresh_df.empty:
        # Nothing new fetched (e.g. start was already past every source's
        # publication frontier). Leave the existing parquet untouched.
        existing_rows = pd.read_parquet(parquet_path) if parquet_path.exists() else None
        n = len(existing_rows) if existing_rows is not None else 0
        print(f"\nNo fresh rows in [{start}, {through}). Parquet unchanged ({n} rows).")
        return {
            "rows": n, "new_rows": 0,
            "columns": existing_rows.shape[1] if existing_rows is not None else 0,
            "errors": errors, "missing_required": [], "parquet_path": parquet_path,
        }
    fresh_df.index = fresh_df.index.tz_convert("UTC")
    fresh_df.index.name = "timestamp"

    # Combine with existing parquet (preserve cols we didn't fetch this run).
    if parquet_path.exists() and not rebuild:
        existing = pd.read_parquet(parquet_path)
        # Drop the overlap window from existing, then concatenate.
        keep = existing.loc[existing.index < start]
        # Add any columns existing has that fresh doesn't (e.g. missing source).
        for col in existing.columns:
            if col not in fresh_df.columns:
                fresh_df[col] = existing[col].reindex(fresh_df.index)
        merged = pd.concat([keep, fresh_df], axis=0).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
    else:
        merged = fresh_df.sort_index()

    # Atomic write
    tmp = parquet_path.with_suffix(".parquet.tmp")
    merged.to_parquet(tmp)
    tmp.replace(parquet_path)

    missing_required = [c for c in REQUIRED if c in errors or c not in merged.columns]
    new_rows = len(merged) - existing_rows_before
    print()
    print(f"Wrote {parquet_path}  rows={len(merged)}  (+{new_rows})  cols={merged.shape[1]}")
    print(f"Range: {merged.index.min()}  ->  {merged.index.max()}")
    if errors:
        print(f"Failed columns ({len(errors)}): {errors}")
    if missing_required:
        print(f"!! Missing REQUIRED columns: {missing_required}", file=sys.stderr)

    return {
        "rows": len(merged),
        "new_rows": new_rows,
        "columns": merged.shape[1],
        "errors": errors,
        "missing_required": missing_required,
        "parquet_path": parquet_path,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Refresh the SMARD parquet from API sources.")
    p.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    p.add_argument("--start", type=pd.Timestamp, default=None,
                   help="Start of fetch window (UTC). Defaults to 24h before existing parquet end.")
    p.add_argument("--through", type=pd.Timestamp, default=None,
                   help="End of fetch window (UTC). Defaults to now.")
    p.add_argument("--rebuild", action="store_true",
                   help="Rebuild from scratch (start defaults to 2022-01-01).")
    p.add_argument("--fail-if-no-new-rows", action="store_true",
                   help="Exit with code 2 if the refresh produced zero new rows. "
                        "Useful for CI: lets the daily action skip an empty commit.")
    args = p.parse_args()

    # Make tz-aware UTC if user passed naive timestamps.
    start = args.start.tz_localize("UTC") if (args.start is not None and args.start.tz is None) else args.start
    through = args.through.tz_localize("UTC") if (args.through is not None and args.through.tz is None) else args.through

    if start is not None and start > pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1) + pd.Timedelta(days=2):
        # Sanity: if start looks accidentally "in 2 days", bail.
        pass

    result = refresh(args.parquet, start=start, through=through, rebuild=args.rebuild)
    if result["missing_required"]:
        return 1
    if args.fail_if_no_new_rows and result.get("new_rows", 0) <= 0:
        print("\n--fail-if-no-new-rows: 0 new rows; exiting 2.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
