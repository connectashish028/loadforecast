# Forecast incident post-mortem

A standing format for explaining a bad forecast day to a non-technical
stakeholder (trader / BESS operator / account manager). Trust in a forecast
is won or lost on the day it's *wrong* — so a miss gets a written, plain-
language explanation, not just a red bar on a chart.

Pair this with the dashboard's **"Why this forecast?"** panel
(`src/loadforecast/models/explain.py`, native TreeSHAP attribution) and the
daily drift log (`backtest_results/drift_log.csv`).

---

## Incident: <delivery date> — <one-line summary>

**Severity:** <e.g. P50 MAE 4× the holdout median / dispatch lost vs naive>

### 1. What happened (plain language, no jargon)
- Forecast said: <e.g. "prices stay positive all afternoon">
- Reality was: <e.g. "prices went to −80 €/MWh from 12:00–15:00">
- Impact on the user: <e.g. "a battery dispatched on this would have
  charged during the cheapest window correctly, but mis-timed the discharge
  peak by one hour">

### 2. What the model was looking at (from the SHAP panel)
- Top drivers of the forecast that day: <paste the 3 plain-language drivers,
  e.g. "TSO residual-load forecast pulled price down 27 €/MWh; cross-border
  France price pulled down 7 €/MWh">
- The driver that misfired: <e.g. "the renewable forecast it trusted
  (`tso_vre_fc`) under-stated the midday solar peak by X GW">

### 3. Root cause
- [ ] Input data was wrong/stale (which feed, how stale)
- [ ] Input was right but the regime was rare (e.g. holiday × record solar —
      few training examples)
- [ ] Model structural limit (e.g. continuous P50 can't reach the deep
      negative tail)
- [ ] Leakage / pipeline bug (should be caught by the corrupt-future test)
- Narrative: <one paragraph>

### 4. Was it foreseeable from the bands?
- Did the realised value fall inside the P10–P90 interval? <yes/no>
- If no: the point miss was also an uncertainty miss — note for calibration.

### 5. Action
- [ ] None — known regime, within expected error, no change
- [ ] Feature/data fix: <what>
- [ ] Add to the stress-test slice for the next retrain
- [ ] Stakeholder comms: <what we told the customer>

### 6. One-line lesson
<the thing we'd tell the next person>
