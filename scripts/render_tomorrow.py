"""Render tomorrow's forecast (model P50/P90 ribbon + TSO baseline) to PNG.

Output: docs/images/tomorrow.png

Used by:
- The README, as a non-interactive fallback when Streamlit Cloud sleeps.
- The daily GitHub Action (M11), which calls this after `data.refresh`
  so the README's tomorrow chart auto-updates without human intervention.

Matplotlib (not Plotly) so we don't need kaleido in CI. Theme matches
the dashboard's xAI-inspired dark palette.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loadforecast.backtest import issue_time_for, load_smard_15min
from loadforecast.models.predict import xgboost_load_predict_full

PARQUET = "smard_merged_15min.parquet"
TSO_COL = "fc_cons__grid_load"
OUT_PATH = Path("docs/images/tomorrow.png")

# Match the dashboard palette.
BG = "#1f2228"
TEXT = "#ffffff"
TEXT_70 = "#b3b3b3"
TEXT_30 = "#4d4d4d"
GRID = "#2a2d35"
PREDICTION = "#B8A1FF"
PREDICTION_FILL = "#B8A1FF"
TSO = "#7a7a7a"


def main() -> None:
    print("Loading parquet...")
    df = load_smard_15min(PARQUET)
    today = datetime.now(ZoneInfo("Europe/Berlin")).date()
    tomorrow = today + timedelta(days=1)

    issue = issue_time_for(tomorrow)
    if issue > df.index.max():
        raise SystemExit(
            f"Tomorrow ({tomorrow}) issue time {issue} is past parquet max "
            f"({df.index.max()}); refresh the parquet first."
        )

    print(f"Predicting delivery {tomorrow}, issue {issue}...")
    fc = xgboost_load_predict_full(df, issue)
    if fc["p50"].isna().any():
        raise SystemExit("Forecast contains NaN — encoder/decoder window incomplete.")

    tso = df[TSO_COL].reindex(fc.index)

    # Plot ----------------------------------------------------------------
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_70,
        "xtick.color": TEXT_70, "ytick.color": TEXT_70,
        "text.color": TEXT,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "font.family": "sans-serif",
    })

    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)

    x = fc.index
    ax.fill_between(x, fc["p10"], fc["p90"],
                    color=PREDICTION_FILL, alpha=0.18, linewidth=0)
    ax.plot(x, fc["p10"], color=PREDICTION, linewidth=0.5, linestyle=":")
    ax.plot(x, fc["p90"], color=PREDICTION, linewidth=0.5, linestyle=":")

    ax.plot(x, tso.values, color=TSO, linewidth=1.4, linestyle="--",
            label="TSO baseline")
    ax.plot(x, fc["p50"], color=PREDICTION, linewidth=2.2,
            label="Model forecast")

    ax.set_title(
        f"Tomorrow's forecast — delivery {tomorrow.isoformat()} · "
        f"issued {today.isoformat()} 12:00 Berlin",
        color=TEXT, fontsize=11, loc="left", pad=12,
    )
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Load (MWh / 15-min)")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT_70,
              loc="upper left", fontsize=9)

    # Spine cleanup: only bottom + left, dim.
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(GRID)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight", dpi=150)
    print(f"Wrote {OUT_PATH}  ({OUT_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
