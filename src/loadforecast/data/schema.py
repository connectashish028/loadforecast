"""Column registry — every output column declared once.

Every column we keep in the parquet has exactly one entry here. The
column name is a stable contract used across the project (M2 features
reference these names verbatim, the LSTM windowing uses two of them).

Adding a new column: append a `Column(...)` to `COLUMNS` below.
- `name`            parquet column name. Must be unique. Snake-case prefix
                    encodes the kind: `actual_cons__`, `actual_gen__`,
                    `fc_cons__`, `fc_gen__`, `price__`. Used by
                    `availability.classify_column` for leakage rules.
- `source`          which sources/<file>.py knows how to fetch it.
- `fetch_kwargs`    source-specific params.

Source priority (most automated → least):

  1. SRC_ENERGY_CHARTS       — REST, no auth. Day-ahead prices (15 zones),
                                actual generation by source.
  2. SRC_SMARD_API           — REST, no auth. Filter IDs we know
                                (total grid load = 410).
  3. SRC_SMARD_DOWNLOADCENTER — REST, no auth. The downloadcenter JSON
                                endpoint, used for TSO consumption
                                forecasts (fc_cons__*).
  4. SRC_OPEN_METEO          — REST, no auth. NWP weather forecasts.
  5. SRC_ENTSOE              — token-gated; not currently used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Source identifiers
SRC_ENERGY_CHARTS = "energy_charts"
SRC_SMARD_API = "smard_api"
SRC_SMARD_DOWNLOADCENTER = "smard_downloadcenter"
SRC_ENTSOE = "entsoe"
SRC_OPEN_METEO = "open_meteo"


# Bidding zones we pull day-ahead prices for, with the zone code suffix.
PRICE_ZONES: dict[str, dict[str, str]] = {
    "germany_luxembourg":   {"ec": "DE-LU"},
    "france":               {"ec": "FR"},
    "netherlands":          {"ec": "NL"},
    "austria":              {"ec": "AT"},
    "belgium":              {"ec": "BE"},
    "switzerland":          {"ec": "CH"},
    "czech_republic":       {"ec": "CZ"},
    "poland":               {"ec": "PL"},
    "denmark_1":            {"ec": "DK1"},
    "denmark_2":            {"ec": "DK2"},
    "norway_2":             {"ec": "NO2"},
    "sweden_4":             {"ec": "SE4"},
    "slovenia":             {"ec": "SI"},
    "hungary":              {"ec": "HU"},
}


@dataclass(frozen=True)
class Column:
    name: str
    source: str
    description: str
    fetch_kwargs: dict = field(default_factory=dict)


def _build_columns() -> list[Column]:
    cols: list[Column] = []

    # 1. Day-ahead prices — Energy-Charts (no auth).
    for suffix, codes in PRICE_ZONES.items():
        cols.append(Column(
            name=f"price__{suffix}",
            source=SRC_ENERGY_CHARTS,
            description=f"Day-ahead spot price, bidding zone {codes['ec']}, EUR/MWh",
            fetch_kwargs={"endpoint": "price", "bzn": codes["ec"]},
        ))

    # 2. Actual generation by source — Energy-Charts.
    EC_GEN = [
        ("photovoltaics",        "Solar"),
        ("wind_onshore",         "Wind onshore"),
        ("wind_offshore",        "Wind offshore"),
        ("biomass",              "Biomass"),
        ("hydropower",           "Hydro Run-of-River"),
        ("hydro_pumped_storage", "Hydro pumped storage"),
        ("lignite",              "Fossil brown coal / lignite"),
        ("hard_coal",            "Fossil hard coal"),
        ("fossil_gas",           "Fossil gas"),
        ("other_renewable",      "Renewable share of load"),
        ("other_conventional",   "Others"),
    ]
    for suffix, ec_name in EC_GEN:
        cols.append(Column(
            name=f"actual_gen__{suffix}",
            source=SRC_ENERGY_CHARTS,
            description=f"Actual {suffix} generation, MW",
            fetch_kwargs={"endpoint": "public_power", "production_type": ec_name},
        ))

    # 3. Actual grid load — SMARD API filter 410 (no auth).
    cols.append(Column(
        name="actual_cons__grid_load",
        source=SRC_SMARD_API,
        description="Actual total grid load (national consumption), MW",
        fetch_kwargs={"filter_id": 410, "region": "DE-LU"},
    ))
    cols.append(Column(
        name="actual_cons__residual_load",
        source=SRC_SMARD_API,
        description="Actual residual load (load - VRE generation), MW",
        fetch_kwargs={"filter_id": 4359, "region": "DE-LU"},
    ))

    # 4. TSO consumption forecasts via SMARD downloadcenter JSON API.
    for suffix in ("grid_load", "residual_load"):
        cols.append(Column(
            name=f"fc_cons__{suffix}",
            source=SRC_SMARD_DOWNLOADCENTER,
            description=f"Day-ahead {suffix} forecast (TSO baseline), MW",
        ))
    # Day-ahead renewable forecast — the dominant price driver. Re-added
    # for the price model after we discovered the load model didn't need
    # it but the price model very much does.
    cols.append(Column(
        name="fc_gen__photovoltaics_and_wind",
        source=SRC_SMARD_DOWNLOADCENTER,
        description="Day-ahead PV + wind generation forecast, MWh per quarter-hour",
    ))

    # 5. Weather (Open-Meteo NWP, population-weighted across 6 load centres).
    OM_VARS = [
        ("temperature_2m",      "Temperature at 2m, degC, pop-weighted national"),
        ("shortwave_radiation", "Shortwave (solar) radiation, W/m^2, pop-weighted"),
        ("wind_speed_100m",     "Wind speed at 100m, km/h, pop-weighted"),
        ("cloud_cover",         "Cloud cover %, pop-weighted"),
    ]
    for var, desc in OM_VARS:
        cols.append(Column(
            name=f"weather__{var}",
            source=SRC_OPEN_METEO,
            description=desc,
            fetch_kwargs={"variable": var},
        ))

    return cols


COLUMNS: list[Column] = _build_columns()
COLUMN_BY_NAME: dict[str, Column] = {c.name: c for c in COLUMNS}


def columns_by_source(source: str) -> list[Column]:
    return [c for c in COLUMNS if c.source == source]


__all__ = [
    "COLUMNS",
    "COLUMN_BY_NAME",
    "Column",
    "PRICE_ZONES",
    "SRC_ENERGY_CHARTS",
    "SRC_ENTSOE",
    "SRC_OPEN_METEO",
    "SRC_SMARD_API",
    "SRC_SMARD_DOWNLOADCENTER",
    "columns_by_source",
]
