"""Production-safety test: requirements.txt covers the dashboard's runtime.

Streamlit Cloud installs from `requirements.txt`; CI installs from
`pyproject.toml` `.[dev]`. They are maintained by hand and can drift —
and when they do, CI stays green while the live deploy fails at import.
This exact regression happened once already (xgboost was in pyproject
but missing from requirements.txt, so the deployed dashboard crashed
with ModuleNotFoundError).

This test asserts every third-party package the dashboard imports at
runtime is pinned in requirements.txt. It's the cheapest guard against
"CI passes, production breaks".
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"

# Third-party PyPI packages the deployed dashboard transitively needs at
# runtime (dashboards/app.py -> loadforecast.* -> these). Keep in sync with
# what dashboards/app.py + src/loadforecast import outside the stdlib.
# If you add a runtime import, add it here AND to requirements.txt.
RUNTIME_PACKAGES = {
    "numpy",
    "pandas",
    "pyarrow",        # parquet engine — silent import, easy to forget
    "scikit-learn",
    "scipy",
    "xgboost",        # the production models
    "tensorflow",     # LSTM comparison-baseline panel
    "holidays",       # calendar features
    "requests",       # data refresh (imported transitively)
    "streamlit",
    "plotly",
}


def _requirement_names(text: str) -> set[str]:
    """Lowercased package names from a requirements.txt body."""
    names = set()
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        # split on the first version specifier or extras bracket
        name = re.split(r"[<>=!~\[ ]", line, maxsplit=1)[0].strip().lower()
        if name:
            names.add(name)
    return names


def test_requirements_file_exists() -> None:
    assert REQUIREMENTS.exists(), "requirements.txt missing — Streamlit Cloud needs it"


def test_dashboard_runtime_deps_are_pinned() -> None:
    """Every runtime package the dashboard needs is in requirements.txt."""
    pinned = _requirement_names(REQUIREMENTS.read_text())
    missing = {pkg for pkg in RUNTIME_PACKAGES if pkg.lower() not in pinned}
    assert not missing, (
        f"requirements.txt is missing dashboard runtime deps: {sorted(missing)}. "
        f"CI installs from pyproject so it would pass, but the Streamlit Cloud "
        f"deploy installs from requirements.txt and would crash. Add them."
    )
