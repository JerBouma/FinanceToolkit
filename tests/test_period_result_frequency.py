"""Period detection preserves the base frequency of multiplied periods."""

import pandas as pd
import pytest

from financetoolkit.helpers import period_of_result


@pytest.mark.parametrize(
    "frequency, expected",
    [("2D", "daily"), ("2W", "weekly"), ("2M", "monthly"), ("2Q", "quarterly"), ("2Y", "yearly")],
)
@pytest.mark.parametrize("on_columns", [False, True])
def test_multiplied_period_frequency(frequency, expected, on_columns):
    frame = pd.DataFrame({"value": [1.0, 2.0]}, index=pd.period_range("2020-01-01", periods=2, freq=frequency))
    assert period_of_result(frame.T if on_columns else frame) == expected
