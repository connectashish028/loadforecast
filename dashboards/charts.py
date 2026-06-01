"""Plotly chart helpers — all charts share the xAI-style dark theme.

Two accent colors only: lilac for prediction, blue for actual. TSO baseline
is rendered as a dimmed white-dashed line so it stays visually subordinate.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .styles import (
    ACTUAL,
    BG,
    BORDER,
    PREDICTION,
    PREDICTION_FILL,
    TEXT,
    TEXT_30,
    TEXT_50,
    TEXT_70,
    TSO,
)

_AXIS = dict(
    showgrid=True, gridcolor=BORDER, gridwidth=1,
    zeroline=False, color=TEXT_70,
    tickfont=dict(family="JetBrains Mono, monospace", size=11, color=TEXT_70),
    title_font=dict(family="JetBrains Mono, monospace", size=11, color=TEXT_50),
)


def _base_layout(title: str | None = None, height: int = 420) -> dict:
    layout = dict(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="Inter, sans-serif", color=TEXT),
        margin=dict(l=50, r=20, t=40 if title else 10, b=50),
        height=height,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(family="JetBrains Mono, monospace", size=10,
                      color=TEXT_70),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=_AXIS, yaxis=_AXIS,
        hoverlabel=dict(
            bgcolor="#2a2d35", bordercolor=BORDER,
            font=dict(family="JetBrains Mono, monospace", color=TEXT, size=11),
        ),
    )
    if title:
        layout["title"] = dict(
            text=title, font=dict(size=14, color=TEXT_70),
            x=0, xanchor="left",
        )
    return layout


def forecast_chart(
    forecast: pd.DataFrame,
    actuals: pd.Series | None = None,
    tso: pd.Series | None = None,
    title: str | None = None,
) -> go.Figure:
    """Quantile-aware forecast chart.

    `forecast` must have columns p10, p50, p90 indexed by tz-aware timestamp.
    `actuals` (optional) is the realised load on that day.
    `tso` (optional) is the TSO baseline forecast.
    """
    fig = go.Figure()

    # X-axis in Berlin local time (the operational tz for the German grid),
    # so a "tomorrow" chart spans 00:00 → 24:00 instead of weird UTC offsets.
    x_fc = forecast.index.tz_convert("Europe/Berlin")

    # P10 / P90 ribbon — pure visual; skipped from hover so the unified
    # tooltip only shows the lines users actually care about. Give them
    # explicit names to prevent Plotly's `tonexty` fill from injecting
    # an "undefined" legend entry.
    fig.add_trace(go.Scatter(
        x=x_fc, y=forecast["p90"], mode="lines",
        line=dict(color=PREDICTION, width=0.5, dash="dot"),
        name="p90_band", legendgroup="band",
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=x_fc, y=forecast["p10"], mode="lines",
        line=dict(color=PREDICTION, width=0.5, dash="dot"),
        fill="tonexty", fillcolor=PREDICTION_FILL,
        name="p10_band", legendgroup="band",
        hoverinfo="skip", showlegend=False,
    ))
    # Median (the headline line)
    fig.add_trace(go.Scatter(
        x=x_fc, y=forecast["p50"], mode="lines",
        line=dict(color=PREDICTION, width=2.2),
        name="Model forecast",
        hovertemplate="%{y:,.0f}<extra>Model</extra>",
    ))

    if tso is not None:
        fig.add_trace(go.Scatter(
            x=tso.index.tz_convert("Europe/Berlin"), y=tso.values, mode="lines",
            line=dict(color=TSO, width=1.4, dash="dash"),
            name="TSO baseline",
            hovertemplate="%{y:,.0f}<extra>TSO</extra>",
        ))

    if actuals is not None and actuals.notna().any():
        fig.add_trace(go.Scatter(
            x=actuals.index.tz_convert("Europe/Berlin"),
            y=actuals.values, mode="lines",
            line=dict(color=ACTUAL, width=2.2),
            name="Actual load",
            hovertemplate="%{y:,.0f}<extra>Actual</extra>",
        ))

    layout = _base_layout(title=title, height=440)
    layout["yaxis"] = {**_AXIS, "title": "Load (MWh / 15-min)"}
    layout["xaxis"] = {**_AXIS, "title": "Time (Berlin)"}
    layout["hovermode"] = "x unified"
    fig.update_layout(**layout)
    return fig


def skill_chart(rolling: pd.DataFrame, title: str | None = None) -> go.Figure:
    """Rolling MAE skill score over time.

    `rolling` must have columns model_mae, tso_mae, skill indexed by date.
    """
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=TEXT_30, width=1, dash="dot"))

    fig.add_trace(go.Scatter(
        x=rolling.index, y=rolling["skill"] * 100, mode="lines",
        line=dict(color=PREDICTION, width=2.2),
        fill="tozeroy", fillcolor=PREDICTION_FILL,
        name="30-day rolling improvement",
        hovertemplate="%{x|%Y-%m-%d}<br>error reduction: %{y:+.1f}%%<extra></extra>",
    ))
    layout = _base_layout(title=title, height=360)
    layout["yaxis"] = {**_AXIS, "tickformat": "+.0f",
                       "ticksuffix": " %",
                       "title": "Error reduction vs TSO (30-day rolling)"}
    layout["xaxis"] = {**_AXIS, "title": "Delivery date"}
    fig.update_layout(**layout)
    return fig


def error_chart(
    actuals: pd.Series,
    forecast: pd.DataFrame,
    tso: pd.Series,
    title: str | None = None,
) -> go.Figure:
    """Per-step forecast error — shows where the model out-performs TSO.

    Two traces: model error (lilac) and TSO error (dashed white). Errors
    are signed (positive = forecast too high), so under/over-forecasting
    is visible at a glance.
    """
    model_err = forecast["p50"].values - actuals.values
    tso_err = tso.values - actuals.values
    x_berlin = actuals.index.tz_convert("Europe/Berlin")

    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=TEXT_30, width=1, dash="dot"))

    fig.add_trace(go.Scatter(
        x=x_berlin, y=tso_err, mode="lines",
        line=dict(color=TSO, width=1.4, dash="dash"),
        name="TSO error",
        hovertemplate="%{y:+,.0f}<extra>TSO error</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x_berlin, y=model_err, mode="lines",
        line=dict(color=PREDICTION, width=2.0),
        name="Model error",
        hovertemplate="%{y:+,.0f}<extra>Model error</extra>",
    ))

    layout = _base_layout(title=title, height=240)
    layout["yaxis"] = {**_AXIS, "title": "Error (forecast − actual)",
                       "tickformat": "+.0f"}
    layout["xaxis"] = {**_AXIS, "title": "Time (Berlin)"}
    layout["hovermode"] = "x unified"
    fig.update_layout(**layout)
    return fig


def ablation_chart(ablation: pd.DataFrame, title: str | None = None) -> go.Figure:
    """Marginal-skill bar chart from the M4-pt3 ablation.

    `ablation` must have a `label` column (variant name) and `holdout_skill`
    column. We compute first-difference deltas and color positive bars in
    lilac, negative in dimmed white.
    """
    df = ablation.copy()
    df["delta"] = df["holdout_skill"].diff().fillna(df["holdout_skill"])
    colors = [
        PREDICTION if v > 0 else TSO
        for v in df["delta"]
    ]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["label"], y=df["delta"] * 100,
        marker=dict(color=colors,
                    line=dict(color=BORDER, width=1)),
        text=[f"{v*100:+.1f}%" for v in df["delta"]],
        textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace", color=TEXT, size=11),
        hovertemplate="<b>%{x}</b><br>marginal Δ: %{y:+.1f}%<extra></extra>",
    ))
    layout = _base_layout(title=title, height=360)
    layout["yaxis"] = {**_AXIS, "tickformat": "+.0f", "ticksuffix": " %",
                       "title": "Marginal error reduction"}
    layout["xaxis"] = {**_AXIS, "title": ""}
    layout["showlegend"] = False
    layout["margin"] = dict(l=50, r=20, t=40 if title else 30, b=60)
    fig.update_layout(**layout)
    return fig


def hour_profile_chart(
    backtest: pd.DataFrame,
    title: str | None = None,
) -> go.Figure:
    """Average absolute error by hour-of-day, model vs TSO.

    `backtest` must have columns y_true, y_model, y_tso, target_ts (UTC).
    We aggregate to local Berlin hour because that's the unit users feel.
    """
    df = backtest.copy()
    df["target_ts"] = pd.to_datetime(df["target_ts"], utc=True)
    df["hour"] = df["target_ts"].dt.tz_convert("Europe/Berlin").dt.hour
    df["err_model"] = (df["y_true"] - df["y_model"]).abs()
    df["err_tso"] = (df["y_true"] - df["y_tso"]).abs()
    profile = df.groupby("hour")[["err_model", "err_tso"]].mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=profile.index, y=profile["err_tso"], mode="lines",
        line=dict(color=TSO, width=1.6, dash="dash"),
        name="TSO baseline",
        hovertemplate="hour %{x}: TSO err %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=profile.index, y=profile["err_model"], mode="lines",
        line=dict(color=PREDICTION, width=2.2),
        fill="tonexty", fillcolor=PREDICTION_FILL,
        name="Model",
        hovertemplate="hour %{x}: model err %{y:,.0f}<extra></extra>",
    ))
    layout = _base_layout(title=title, height=320)
    layout["xaxis"] = {**_AXIS, "title": "Hour of day (Berlin local)",
                       "dtick": 3, "tickmode": "linear"}
    layout["yaxis"] = {**_AXIS, "title": "Mean absolute error (MWh / 15-min)"}
    layout["hovermode"] = "x unified"
    fig.update_layout(**layout)
    return fig


def volatility_quartile_chart(
    quartiles: pd.DataFrame,
    title: str | None = None,
) -> go.Figure:
    """Paired bars: mean daily MAE for model vs TSO, by price-spread quartile.

    `quartiles` must have columns:
      - `label`     ('Calm', 'Moderate', 'High', 'Extreme')
      - `range`     human-readable spread range, used as the secondary axis
      - `mae_model`, `mae_tso`  mean daily MAE in MWh / 15-min
      - `n_days`    sample size in that bin

    Visual treatment:
      - Model bars: solid lilac fill
      - TSO bars:   transparent fill with white-dashed border (mirrors
                    the line-chart convention)
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=quartiles["label"], y=quartiles["mae_tso"],
        name="TSO baseline",
        marker=dict(color="rgba(0,0,0,0)",
                    line=dict(color=TSO, width=1.5)),
        text=[f"{v:.0f}" for v in quartiles["mae_tso"]],
        textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace",
                      color=TEXT_70, size=11),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "TSO mean MAE %{y:,.0f}<extra></extra>"
        ),
    ))
    fig.add_trace(go.Bar(
        x=quartiles["label"], y=quartiles["mae_model"],
        name="Model",
        marker=dict(color=PREDICTION,
                    line=dict(color=PREDICTION, width=0)),
        text=[f"{v:.0f}" for v in quartiles["mae_model"]],
        textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace",
                      color=TEXT, size=11),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "model mean MAE %{y:,.0f}<extra></extra>"
        ),
    ))

    # Range subtitles printed below each bar group as a second tick row.
    range_text = [f"{r}<br>n={n}"
                  for r, n in zip(quartiles["range"], quartiles["n_days"],
                                  strict=True)]

    layout = _base_layout(title=title, height=380)
    layout["barmode"] = "group"
    layout["bargap"] = 0.45
    layout["bargroupgap"] = 0.08
    layout["xaxis"] = {
        **_AXIS,
        "title": "",
        "tickmode": "array",
        "tickvals": list(quartiles["label"]),
        "ticktext": [f"{lbl}<br><span style='color:rgba(255,255,255,0.4); "
                     f"font-size:10px;'>{txt}</span>"
                     for lbl, txt in zip(quartiles["label"], range_text,
                                         strict=True)],
    }
    layout["yaxis"] = {**_AXIS,
                       "title": "Mean daily MAE (MWh / 15-min)"}
    layout["margin"] = dict(l=50, r=20, t=40 if title else 30, b=80)
    fig.update_layout(**layout)
    return fig


def price_forecast_chart(
    forecast: pd.DataFrame,
    actuals: pd.Series | None = None,
    title: str | None = None,
) -> go.Figure:
    """Day-ahead price forecast with P10/P90 ribbon. €/MWh on the y-axis.

    Differs from `forecast_chart` in two ways: there's no TSO baseline (the
    day-ahead price is itself the auction outcome, no operator forecast to
    benchmark against), and we draw a zero line because negative prices
    are a routine feature of the German market on high-PV days.
    """
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=TEXT_30, width=1, dash="dot"))

    x_fc = forecast.index.tz_convert("Europe/Berlin")

    fig.add_trace(go.Scatter(
        x=x_fc, y=forecast["p90"], mode="lines",
        line=dict(color=PREDICTION, width=0.5, dash="dot"),
        name="p90_band", legendgroup="band",
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=x_fc, y=forecast["p10"], mode="lines",
        line=dict(color=PREDICTION, width=0.5, dash="dot"),
        fill="tonexty", fillcolor=PREDICTION_FILL,
        name="p10_band", legendgroup="band",
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=x_fc, y=forecast["p50"], mode="lines",
        line=dict(color=PREDICTION, width=2.2),
        name="Model forecast (P50)",
        hovertemplate="%{y:,.1f} €/MWh<extra>P50</extra>",
    ))

    if actuals is not None and actuals.notna().any():
        fig.add_trace(go.Scatter(
            x=actuals.index.tz_convert("Europe/Berlin"),
            y=actuals.values, mode="lines",
            line=dict(color=ACTUAL, width=2.2),
            name="Actual price",
            hovertemplate="%{y:,.1f} €/MWh<extra>Actual</extra>",
        ))

    layout = _base_layout(title=title, height=360)
    layout["yaxis"] = {**_AXIS, "title": "Day-ahead price (€/MWh)"}
    layout["xaxis"] = {**_AXIS, "title": "Time (Berlin)"}
    layout["hovermode"] = "x unified"
    fig.update_layout(**layout)
    return fig


def price_hour_profile_chart(
    backtest: pd.DataFrame,
    title: str | None = None,
) -> go.Figure:
    """Average absolute price error by hour-of-day, model vs naive yesterday.

    `backtest` must have columns y_true, p50, naive_1d, target_ts (UTC).
    Aggregates to local Berlin hour.
    """
    df = backtest.copy()
    df["target_ts"] = pd.to_datetime(df["target_ts"], utc=True)
    df["hour"] = df["target_ts"].dt.tz_convert("Europe/Berlin").dt.hour
    df["err_model"] = (df["y_true"] - df["p50"]).abs()
    df["err_naive"] = (df["y_true"] - df["naive_1d"]).abs()
    profile = df.groupby("hour")[["err_model", "err_naive"]].mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=profile.index, y=profile["err_naive"], mode="lines",
        line=dict(color=TSO, width=1.6, dash="dash"),
        name="Naive yesterday",
        hovertemplate="hour %{x}: naive err %{y:,.1f} €/MWh<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=profile.index, y=profile["err_model"], mode="lines",
        line=dict(color=PREDICTION, width=2.2),
        fill="tonexty", fillcolor=PREDICTION_FILL,
        name="Model (P50)",
        hovertemplate="hour %{x}: model err %{y:,.1f} €/MWh<extra></extra>",
    ))
    layout = _base_layout(title=title, height=320)
    layout["xaxis"] = {**_AXIS, "title": "Hour of day (Berlin local)",
                       "dtick": 3, "tickmode": "linear"}
    layout["yaxis"] = {**_AXIS, "title": "Mean absolute error (€/MWh)"}
    layout["hovermode"] = "x unified"
    fig.update_layout(**layout)
    return fig


def price_skill_chart(rolling: pd.DataFrame, title: str | None = None) -> go.Figure:
    """30-day rolling MAE skill score for the price model vs naive yesterday.

    `rolling` must have columns model_mae_30d, naive_mae_30d, skill indexed by date.
    Identical visual to the load skill chart, just different baseline.
    """
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=TEXT_30, width=1, dash="dot"))
    fig.add_trace(go.Scatter(
        x=rolling.index, y=rolling["skill"] * 100, mode="lines",
        line=dict(color=PREDICTION, width=2.2),
        fill="tozeroy", fillcolor=PREDICTION_FILL,
        name="30-day rolling improvement",
        hovertemplate="%{x|%Y-%m-%d}<br>error reduction: %{y:+.1f}%%<extra></extra>",
    ))
    layout = _base_layout(title=title, height=360)
    layout["yaxis"] = {**_AXIS, "tickformat": "+.0f", "ticksuffix": " %",
                       "title": "Error reduction vs naive (30-day rolling)"}
    layout["xaxis"] = {**_AXIS, "title": "Delivery date"}
    fig.update_layout(**layout)
    return fig


def price_spread_quartile_chart(
    quartiles: pd.DataFrame,
    title: str | None = None,
) -> go.Figure:
    """Paired bars: actual vs model-predicted daily spread by spread quartile.

    `quartiles` must have columns:
      - `label`         ('Calm', 'Moderate', 'High', 'Extreme')
      - `range`         human-readable spread range
      - `actual_spread` mean actual daily spread in €/MWh
      - `model_spread`  mean model P50 spread in €/MWh
      - `n_days`        sample size in that bin
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=quartiles["label"], y=quartiles["actual_spread"],
        name="Actual spread",
        marker=dict(color="rgba(0,0,0,0)",
                    line=dict(color=ACTUAL, width=1.5)),
        text=[f"{v:.0f}" for v in quartiles["actual_spread"]],
        textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace",
                      color=TEXT_70, size=11),
        hovertemplate="<b>%{x}</b><br>actual %{y:,.0f} €/MWh<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=quartiles["label"], y=quartiles["model_spread"],
        name="Model P50 spread",
        marker=dict(color=PREDICTION,
                    line=dict(color=PREDICTION, width=0)),
        text=[f"{v:.0f}" for v in quartiles["model_spread"]],
        textposition="outside",
        textfont=dict(family="JetBrains Mono, monospace",
                      color=TEXT, size=11),
        hovertemplate="<b>%{x}</b><br>model %{y:,.0f} €/MWh<extra></extra>",
    ))

    range_text = [f"{r}<br>n={n}"
                  for r, n in zip(quartiles["range"], quartiles["n_days"],
                                  strict=True)]
    layout = _base_layout(title=title, height=380)
    layout["barmode"] = "group"
    layout["bargap"] = 0.45
    layout["bargroupgap"] = 0.08
    layout["xaxis"] = {
        **_AXIS, "title": "",
        "tickmode": "array",
        "tickvals": list(quartiles["label"]),
        "ticktext": [f"{lbl}<br><span style='color:rgba(255,255,255,0.4); "
                     f"font-size:10px;'>{txt}</span>"
                     for lbl, txt in zip(quartiles["label"], range_text,
                                         strict=True)],
    }
    layout["yaxis"] = {**_AXIS, "title": "Daily spread (€/MWh)"}
    layout["margin"] = dict(l=50, r=20, t=40 if title else 30, b=80)
    fig.update_layout(**layout)
    return fig


def price_pnl_chart(pnl: pd.DataFrame, title: str | None = None) -> go.Figure:
    """Cumulative battery P&L over the holdout — perfect-foresight / naive / model P50 / band.

    `pnl` must have columns issue_date, oracle_pnl, naive_pnl, model_p50_pnl,
    model_band_pnl. The dispatch simulation lives in `loadforecast.dispatch`.
    """
    pnl = pnl.sort_values("issue_date").reset_index(drop=True)
    x = pd.to_datetime(pnl["issue_date"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=pnl["oracle_pnl"].cumsum(), mode="lines",
        name="Perfect-foresight",
        line=dict(color=TEXT_70, width=1.4),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f} €<extra>Perfect-foresight</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=pnl["model_p50_pnl"].cumsum(), mode="lines",
        name="Model P50",
        line=dict(color=PREDICTION, width=2.4),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f} €<extra>Model P50</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=pnl["model_band_pnl"].cumsum(), mode="lines",
        name="Model P10/P90",
        line=dict(color=PREDICTION, width=1.4, dash="dot"),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f} €<extra>Model P10/P90</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=pnl["naive_pnl"].cumsum(), mode="lines",
        name="Naive yesterday",
        line=dict(color=TSO, width=1.4, dash="dash"),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f} €<extra>Naive</extra>",
    ))
    layout = _base_layout(title=title, height=400)
    layout["yaxis"] = {**_AXIS, "title": "Cumulative P&L (€)"}
    layout["xaxis"] = {**_AXIS, "title": "Delivery date"}
    layout["hovermode"] = "x unified"
    fig.update_layout(**layout)
    return fig


def architecture_drift_chart(drift_log, model: str, title: str | None = None) -> go.Figure:
    """Daily MAE trace: LSTM (lilac) vs XGBoost (blue) on the same target.

    `drift_log` must have columns:
      - delivery_date
      - {load,price}_mae_lstm_{mw,eur}
      - {load,price}_mae_xgb_{mw,eur}
    `model`: 'load' or 'price'.
    """
    df = drift_log.copy()
    df["delivery_date"] = pd.to_datetime(df["delivery_date"])
    df = df.sort_values("delivery_date")

    if model == "load":
        lstm_col, xgb_col = "load_mae_lstm_mw", "load_mae_xgb_mw"
        unit = "MW per QH"
    else:
        lstm_col, xgb_col = "price_mae_lstm_eur", "price_mae_xgb_eur"
        unit = "€/MWh"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["delivery_date"], y=df[lstm_col], mode="lines+markers",
        line=dict(color=PREDICTION, width=2),
        marker=dict(size=4),
        name="LSTM",
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra>LSTM</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["delivery_date"], y=df[xgb_col], mode="lines+markers",
        line=dict(color=ACTUAL, width=2),
        marker=dict(size=4),
        name="XGBoost",
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra>XGBoost</extra>",
    ))
    layout = _base_layout(title=title, height=280)
    layout["yaxis"] = {**_AXIS, "title": f"Daily P50 MAE ({unit})"}
    layout["xaxis"] = {**_AXIS, "title": "Delivery date"}
    layout["hovermode"] = "x unified"
    fig.update_layout(**layout)
    return fig


def intraday_spike_chart(
    forecasts: pd.DataFrame,
    delivery_date,
    title: str | None = None,
) -> go.Figure:
    """Plot the price model's P50 at multiple issue times for one delivery day,
    plus the realised price. `forecasts` is the wide xgboost_intraday_spike
    output, filtered or unfiltered.

    Cadence is conveyed by colour intensity — earliest issue is faded, latest
    is bold — so the eye reads "as time progresses, the forecast firms up."
    """
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=TEXT_30, width=1, dash="dot"))

    df_day = forecasts[forecasts["delivery_date"] == str(delivery_date)].copy()
    df_day["target_ts"] = pd.to_datetime(df_day["target_ts"], utc=True)
    df_day = df_day.sort_values(["issue_label", "target_ts"])

    # Cadence shades — light → dark as issue moves later. All in the lilac
    # PREDICTION hue so the theme stays consistent.
    SHADES = {
        "D-1 06:00":         ("#dac6ff", 1.4),
        "D-1 12:00 (gate)":  ("#b8a1ff", 2.2),
        "D-1 18:00":         ("#8c6dff", 1.6),
        "D-1 21:00":         ("#6537d6", 1.8),
    }
    for label, group in df_day.groupby("issue_label"):
        color, width = SHADES.get(label, (PREDICTION, 1.6))
        x = group["target_ts"].dt.tz_convert("Europe/Berlin")
        fig.add_trace(go.Scatter(
            x=x, y=group["p50"], mode="lines",
            line=dict(color=color, width=width),
            name=f"Forecast P50 · {label}",
            hovertemplate=f"%{{y:,.1f}} €/MWh<extra>{label}</extra>",
        ))

    # Realised price — same series for every issue, just take from the first
    # group with non-null values.
    first = df_day.drop_duplicates("target_ts")[["target_ts", "y_true"]].dropna()
    if not first.empty:
        x_act = first["target_ts"].dt.tz_convert("Europe/Berlin")
        fig.add_trace(go.Scatter(
            x=x_act, y=first["y_true"], mode="lines",
            line=dict(color=ACTUAL, width=2.4),
            name="Actual price",
            hovertemplate="%{y:,.1f} €/MWh<extra>Actual</extra>",
        ))

    layout = _base_layout(title=title, height=380)
    layout["yaxis"] = {**_AXIS, "title": "Day-ahead price (€/MWh)"}
    layout["xaxis"] = {**_AXIS, "title": "Time (Berlin)"}
    layout["hovermode"] = "x unified"
    fig.update_layout(**layout)
    return fig


__all__ = ["architecture_drift_chart",
           "forecast_chart", "skill_chart", "error_chart",
           "ablation_chart", "hour_profile_chart",
           "intraday_spike_chart",
           "price_forecast_chart",
           "price_hour_profile_chart",
           "price_pnl_chart",
           "price_skill_chart",
           "price_spread_quartile_chart",
           "volatility_quartile_chart"]
