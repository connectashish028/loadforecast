"""Render tomorrow's day-ahead price forecast (model P50/P90 ribbon) to PNG.

Output: docs/images/tomorrow_price.png

Used by:
- The README, as a non-interactive fallback when Streamlit Cloud sleeps.
- The daily GitHub Action, which calls this after `data.refresh` so the
  README's tomorrow-price chart auto-updates.

Mirrors `render_tomorrow.py` (the load preview) but for the price model.
Matplotlib (not Plotly) so we don't need kaleido in CI. Theme matches the
dashboard's xAI-inspired dark palette.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt

from loadforecast.backtest import issue_time_for, load_smard_15min
from loadforecast.models.predict import xgboost_price_predict_full

PARQUET = "smard_merged_15min.parquet"
PRICE_COL = "price__germany_luxembourg"
VRE_FC_COL = "fc_gen__photovoltaics_and_wind"
OUT_PATH = Path("docs/images/tomorrow_price.png")

# Match the dashboard palette.
BG = "#1f2228"
TEXT = "#ffffff"
TEXT_70 = "#b3b3b3"
TEXT_30 = "#4d4d4d"
GRID = "#2a2d35"
PREDICTION = "#B8A1FF"
PREDICTION_FILL = "#B8A1FF"


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

    print(f"Predicting price for delivery {tomorrow}, issue {issue}...")
    fc = xgboost_price_predict_full(df, issue)
    if fc["p50"].isna().any():
        raise SystemExit("Forecast contains NaN — encoder/decoder window incomplete.")

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
    ax.plot(x, fc["p50"], color=PREDICTION, linewidth=2.2,
            label="Model forecast (P50)")

    # Zero line — negative prices are routine on high-PV days.
    ax.axhline(0, color=TEXT_30, linewidth=0.8, linestyle=":")

    title = (
        f"Tomorrow's day-ahead price — delivery {tomorrow.isoformat()} · "
        f"issued {today.isoformat()} 12:00 Berlin"
    )
    ax.set_title(title, color=TEXT, fontsize=11, loc="left", pad=12)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Day-ahead price (€/MWh)")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT_70,
              loc="upper left", fontsize=9)

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
