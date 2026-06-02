"""Battery-dispatch P&L simulation — what's a price forecast worth in €?

Operational framing — what a trading desk would actually do:

- Battery: 10 MW / 20 MWh, 90 % round-trip efficiency, max 3 cycles/day
  → 3 × 20 MWh = 60 MWh charge throughput per day
  → at 10 MW power: 60 MWh / 10 MW = 6 hours = 24 quarter-hour slots charging
  → same for discharging (24 slots)

- Strategy: greedy dispatch given a forecast for tomorrow.
    1. Rank slots by forecast price ascending → take the cheapest 24 to charge.
    2. Rank slots by forecast price descending → take the most expensive 24 to discharge.
    3. Drop any slot that's in both lists (no cycling against ourselves).
    4. Realise P&L at *actual* prices the next day.

- Quantile twist: when the forecast is a probabilistic model, **don't use the
  median for both decisions**. Use:
      - P10 prices to find cheap charging slots (the model's "low-end" estimates)
      - P90 prices to find expensive discharging slots (the "high-end" estimates)
  This sidesteps the median-collapse problem we caught in section 6 of the
  notebook — where P50 - P50 underestimates daily price spread.

P&L per cycle (per MWh of charge):
    profit = energy × (RTE × p_discharge − p_charge)
where energy = power × slot_duration and RTE is round-trip efficiency.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BatterySpec:
    power_mw: float = 10.0
    capacity_mwh: float = 20.0
    rte: float = 0.9         # round-trip efficiency (0.9 = 90 %)
    cycles_per_day: float = 3.0  # max throughput as multiples of capacity
    slot_hours: float = 0.25     # 15-minute resolution

    @property
    def energy_per_slot(self) -> float:
        """MWh delivered or absorbed in a single quarter-hour slot at full power."""
        return self.power_mw * self.slot_hours

    @property
    def max_slots_per_direction(self) -> int:
        """How many quarter-hour slots can we charge (or discharge) in a day,
        given the cycle limit? `floor(cycles × capacity / energy_per_slot)`."""
        return int(self.cycles_per_day * self.capacity_mwh / self.energy_per_slot)


# Module-level default — BatterySpec is frozen, so sharing one instance
# across calls is safe (B008 fix).
_DEFAULT_SPEC = BatterySpec()


def dispatch_pnl(
    charge_signal: np.ndarray,
    discharge_signal: np.ndarray,
    actual_prices: np.ndarray,
    spec: BatterySpec = _DEFAULT_SPEC,
) -> dict:
    """Greedy dispatch + realised P&L.

    Args:
        charge_signal: forecast prices used to pick charge slots
            (low values → take). Length 96 (one delivery day).
        discharge_signal: forecast prices used to pick discharge slots
            (high values → take). Length 96.
        actual_prices: realised €/MWh used to compute the P&L. Length 96.
        spec: battery configuration.

    Returns dict with keys:
        - charge_slots, discharge_slots: index arrays (sorted)
        - charge_cost: € paid to charge
        - discharge_revenue: € earned from discharging (post-RTE)
        - net_pnl: € profit
        - n_cycles_realised: actual cycles after de-overlap
    """
    n = spec.max_slots_per_direction
    # Greedy pick — both signals can be the same array (point forecast) or
    # different (e.g. P10 for charge, P90 for discharge).
    charge_idx = set(np.argsort(charge_signal)[:n].tolist())
    discharge_idx = set(np.argsort(discharge_signal)[-n:].tolist())

    # Resolve overlaps: a slot in both lists is dropped from both — we'd
    # be cycling against ourselves at no profit, and the cycle counts.
    overlap = charge_idx & discharge_idx
    charge_idx -= overlap
    discharge_idx -= overlap
    charge_idx = sorted(charge_idx)
    discharge_idx = sorted(discharge_idx)

    e = spec.energy_per_slot
    rte = spec.rte
    cost = sum(actual_prices[i] * e for i in charge_idx)
    revenue = sum(actual_prices[i] * e * rte for i in discharge_idx)
    net = revenue - cost
    n_cycles = (len(discharge_idx) * e) / spec.capacity_mwh

    return {
        "charge_slots": charge_idx,
        "discharge_slots": discharge_idx,
        "charge_cost": float(cost),
        "discharge_revenue": float(revenue),
        "net_pnl": float(net),
        "n_cycles_realised": float(n_cycles),
    }


__all__ = ["BatterySpec", "dispatch_pnl"]
