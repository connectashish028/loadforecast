"""Regression tests for smard_downloadcenter — the source with the
most plumbing complexity (90-day chunking, English-locale numerics,
'-' placeholders for unpublished forecast slots).

We mock the HTTP layer so these run offline in CI.
"""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from loadforecast.data.schema import COLUMN_BY_NAME
from loadforecast.data.sources import smard_downloadcenter as src

# A minimal CSV body matching what the live SMARD downloadcenter returns
# for the consumption-forecast feed: BOM, semicolon-delimited, English
# locale (`,` thousands, `.` decimal), `-` for unpublished slots, blank
# lines between rows, Berlin-local "MMM d, YYYY h:mm AM/PM" timestamps.
SAMPLE_CSV = (
    "﻿Start date;End date;grid load [MWh] Original resolutions\r\n"
    "Apr 1, 2026 2:00 AM;Apr 1, 2026 2:15 AM;11,667.99\r\n"
    "Apr 1, 2026 2:15 AM;Apr 1, 2026 2:30 AM;11,581.65\r\n"
    "Apr 1, 2026 2:30 AM;Apr 1, 2026 2:45 AM;11,523.25\r\n"
    "Apr 1, 2026 2:45 AM;Apr 1, 2026 3:00 AM;-\r\n"  # unpublished slot
    "Apr 1, 2026 3:00 AM;Apr 1, 2026 3:15 AM;11,810.91\r\n"
)


def test_parse_chunk_handles_dash_and_thousands():
    """`pd.to_numeric` would silently NaN every cell if we left the column
    as object dtype with '11,667.99' strings mixed with '-'. The parser
    has to strip both before coercing — this is what _parse_chunk does."""
    df = src._parse_chunk(SAMPLE_CSV)
    assert len(df) == 5, "all 5 data rows kept"
    val_col = "grid load [MWh] Original resolutions"
    values = df[val_col].tolist()
    # 4 numeric rows survived parsing, 1 '-' row coerced to NaN.
    assert values[0] == pytest.approx(11667.99)
    assert values[1] == pytest.approx(11581.65)
    assert pd.isna(values[3])  # '-' became NaN
    assert values[4] == pytest.approx(11810.91)


def test_parse_chunk_index_is_tz_aware_utc():
    df = src._parse_chunk(SAMPLE_CSV)
    assert df.index.tz is not None
    # Berlin 2:00 AM Apr 1, 2026 = UTC 00:00 (CEST in effect).
    assert df.index[0] == pd.Timestamp("2026-04-01 00:00", tz="UTC")


def test_fetch_chunks_a_long_window():
    """A 365-day request must split into multiple ~90-day chunks. We
    capture what `_request` is called with, not return real data."""
    src._fetch_cached.cache_clear()
    calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def fake_request(module_id, start, end, region):
        calls.append((start, end))
        # Return a minimal valid body so `_parse_chunk` doesn't bail.
        return SAMPLE_CSV

    with patch.object(src, "_request", side_effect=fake_request):
        col = COLUMN_BY_NAME["fc_cons__grid_load"]
        start = pd.Timestamp("2025-01-01", tz="UTC")
        end = pd.Timestamp("2026-01-01", tz="UTC")
        src.fetch(col, start, end)

    # 365 days / 90 days per chunk → 5 chunks (4 full + 1 partial).
    assert 4 <= len(calls) <= 5, f"expected ~5 chunks, got {len(calls)}"
    # Chunks must be contiguous and inside the requested window.
    for (_s, e), (s2, _) in zip(calls, calls[1:], strict=False):
        assert e == s2, "chunk boundaries must touch"
    assert calls[0][0] == start
    assert calls[-1][1] == end


def test_fetch_returns_named_series_with_clipped_window():
    """`fetch` should return a tz-aware UTC Series clipped to [start, end)
    and named after the schema column."""
    src._fetch_cached.cache_clear()
    with patch.object(src, "_request", return_value=SAMPLE_CSV):
        col = COLUMN_BY_NAME["fc_cons__grid_load"]
        start = pd.Timestamp("2026-04-01", tz="UTC")
        end = pd.Timestamp("2026-04-02", tz="UTC")
        s = src.fetch(col, start, end)

    assert s.name == "fc_cons__grid_load"
    assert s.index.tz is not None
    assert s.index.min() >= start and s.index.max() < end
    # The '-' row should have produced a NaN, not been dropped.
    assert s.isna().any()


def test_unknown_column_raises():
    src._fetch_cached.cache_clear()
    fake_col = COLUMN_BY_NAME["actual_cons__grid_load"]  # not a downloadcenter column
    with pytest.raises(ValueError, match="no module ID"):
        src.fetch(fake_col, pd.Timestamp("2026-01-01", tz="UTC"),
                  pd.Timestamp("2026-01-02", tz="UTC"))
