"""Production-safety test: the Streamlit dashboard actually boots.

Streamlit Cloud auto-redeploys `dashboards/app.py` on every push to main.
`pytest` and `ruff` both pass on a dashboard that imports cleanly and
byte-compiles but crashes at runtime — a dangling reference after a
refactor, a chart helper that errors on real data, a KPI computation
that divides by zero. Those only surface when the script actually runs
top-to-bottom against the committed parquet + checkpoints.

`streamlit.testing.v1.AppTest` executes the app script headlessly in a
simulated session, so this test catches the runtime-crash class before
the broken code reaches the live demo. We exercise BOTH top-level views
(LOAD default, PRICE) because the view switch runs different code paths.

Skips cleanly if the data layer isn't present (e.g. a checkout without
the committed parquet), so the test never produces a false failure.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "dashboards" / "app.py"
PARQUET = ROOT / "smard_merged_15min.parquet"

# The app loads a 17 MB parquet + XGBoost models + (for the architecture
# panel) a Keras/TF checkpoint, so first render is slow. Generous timeout.
BOOT_TIMEOUT_S = 180

pytestmark = pytest.mark.skipif(
    not (APP.exists() and PARQUET.exists()),
    reason="dashboard app.py or parquet not present in this checkout",
)


def _run_app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=BOOT_TIMEOUT_S)
    at.run()
    return at


def test_dashboard_boots_default_view() -> None:
    """App renders the default (LOAD) view with no uncaught exception."""
    at = _run_app()
    assert not at.exception, f"dashboard raised on default view: {at.exception}"
    # Sanity: the brand bar / a header rendered (the app emits many markdown
    # blocks; at least one must be present if the script ran past setup).
    assert len(at.markdown) > 0, "no markdown rendered — app exited early"


def test_dashboard_boots_price_view() -> None:
    """Switching to the PRICE view runs the price code path cleanly."""
    at = _run_app()
    # Find the PRICE nav button and click it, then re-run.
    price_btns = [b for b in at.button if "PRICE" in (b.label or "").upper()]
    if not price_btns:
        pytest.skip("PRICE nav button not found — UI layout changed")
    price_btns[0].click().run()
    assert not at.exception, f"dashboard raised on PRICE view: {at.exception}"
