"""Tests for the note on a last row whose session has not closed yet."""

from datetime import date, timedelta

import pandas as pd

from financetoolkit.cache import policy_model
from financetoolkit.mcp_server.registry_controller import _unclosed_session_note


def prices(last_day: date) -> pd.DataFrame:
    return pd.DataFrame(
        {"Close": [1.0, 2.0]},
        index=pd.period_range(end=last_day, periods=2, freq="D"),
    )


def test_a_last_row_of_an_unclosed_session_is_noted(monkeypatch):
    today = date.today()
    monkeypatch.setattr(
        policy_model, "get_last_closed_session", lambda now=None: today - timedelta(1)
    )

    note = _unclosed_session_note(prices(today))

    assert note is not None and str(today) in note


def test_a_closed_session_or_a_yearly_period_is_not_noted(monkeypatch):
    today = date.today()
    monkeypatch.setattr(
        policy_model, "get_last_closed_session", lambda now=None: today - timedelta(1)
    )

    assert _unclosed_session_note(prices(today - timedelta(1))) is None
    yearly = pd.DataFrame(
        {"Beta": [1.2]}, index=pd.period_range(end=today, periods=1, freq="Y")
    )
    assert _unclosed_session_note(yearly) is None
