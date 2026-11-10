"""Risk Diagnostics Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.econometrics import diagnostics_model

# Thresholds for asserting significance, or the lack of it, in the tests below.
HIGH_P_VALUE_THRESHOLD = 0.05
LOW_P_VALUE_THRESHOLD = 0.001
SIGNIFICANCE_LEVEL = 0.05
VARIANCE_RATIO_LOWER_BOUND = 0.8
VARIANCE_RATIO_UPPER_BOUND = 1.2


def test_get_arch_lm_test(recorder):
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.standard_normal(200) * 0.01)
    recorder.capture(diagnostics_model.get_arch_lm_test(returns).round(4))


def test_get_arch_lm_test_dataframe(recorder):
    rng = np.random.default_rng(42)
    returns_df = pd.DataFrame(
        {
            "AAPL": rng.standard_normal(200) * 0.01,
            "MSFT": rng.standard_normal(200) * 0.01,
        }
    )
    recorder.capture(diagnostics_model.get_arch_lm_test(returns_df).round(4))


def test_get_arch_lm_test_too_few_observations(recorder):
    returns = pd.Series([0.01, 0.02, 0.01])
    recorder.capture(diagnostics_model.get_arch_lm_test(returns, lags=5))


def test_get_jarque_bera_test(recorder):
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.standard_normal(200) * 0.01)
    recorder.capture(diagnostics_model.get_jarque_bera_test(returns).round(4))


def test_get_jarque_bera_test_dataframe(recorder):
    rng = np.random.default_rng(42)
    returns_df = pd.DataFrame(
        {
            "AAPL": rng.standard_normal(200) * 0.01,
            "MSFT": rng.standard_exponential(200) * 0.01,
        }
    )
    recorder.capture(diagnostics_model.get_jarque_bera_test(returns_df).round(4))


def test_get_ljung_box_test_white_noise(recorder):
    # White noise should fail to reject no autocorrelation, so a high p-value.
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.standard_normal(500) * 0.01)
    result = diagnostics_model.get_ljung_box_test(returns, lags=10).round(4)
    recorder.capture(result)
    recorder.capture(bool(result["P-Value"] > HIGH_P_VALUE_THRESHOLD))


def test_get_ljung_box_test_autocorrelated(recorder):
    # A strongly autocorrelated AR(1) should reject, so a low p-value.
    rng = np.random.default_rng(42)
    values = [0.0]
    for _ in range(500):
        values.append(0.8 * values[-1] + rng.standard_normal() * 0.01)
    returns = pd.Series(values[1:])

    result = diagnostics_model.get_ljung_box_test(returns, lags=10).round(4)
    recorder.capture(result)
    recorder.capture(bool(result["P-Value"] < LOW_P_VALUE_THRESHOLD))


def test_get_ljung_box_test_dataframe(recorder):
    rng = np.random.default_rng(42)
    returns_df = pd.DataFrame(
        {
            "AAPL": rng.standard_normal(200) * 0.01,
            "MSFT": rng.standard_normal(200) * 0.01,
        }
    )
    recorder.capture(diagnostics_model.get_ljung_box_test(returns_df).round(4))


def test_get_ljung_box_test_too_few_observations(recorder):
    returns = pd.Series([0.01, 0.02, 0.01])
    recorder.capture(diagnostics_model.get_ljung_box_test(returns, lags=10))


def test_get_variance_ratio_test_iid(recorder):
    # IID returns should give a Variance Ratio close to 1 and a high p-value.
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.standard_normal(1000) * 0.01)
    result = diagnostics_model.get_variance_ratio_test(returns, q=5).round(4)
    recorder.capture(result)
    recorder.capture(
        bool(
            VARIANCE_RATIO_LOWER_BOUND
            < result["Variance Ratio"]
            < VARIANCE_RATIO_UPPER_BOUND
        )
    )


def test_get_variance_ratio_test_momentum(recorder):
    # A momentum series should give a Variance Ratio > 1 and a low p-value.
    rng = np.random.default_rng(42)
    values = [0.0]
    for _ in range(1000):
        values.append(0.3 * values[-1] + rng.standard_normal() * 0.01)
    returns = pd.Series(values[1:])

    result = diagnostics_model.get_variance_ratio_test(returns, q=5).round(4)
    recorder.capture(result)
    recorder.capture(
        bool(result["Variance Ratio"] > 1 and result["P-Value"] < SIGNIFICANCE_LEVEL)
    )


def test_get_variance_ratio_test_mean_reversion(recorder):
    # A mean-reverting series should give a Variance Ratio < 1 and a low p-value.
    rng = np.random.default_rng(42)
    values = [0.0]
    for _ in range(1000):
        values.append(-0.3 * values[-1] + rng.standard_normal() * 0.01)
    returns = pd.Series(values[1:])

    result = diagnostics_model.get_variance_ratio_test(returns, q=5).round(4)
    recorder.capture(result)
    recorder.capture(
        bool(result["Variance Ratio"] < 1 and result["P-Value"] < SIGNIFICANCE_LEVEL)
    )


def test_get_variance_ratio_test_dataframe(recorder):
    rng = np.random.default_rng(42)
    returns_df = pd.DataFrame(
        {
            "AAPL": rng.standard_normal(200) * 0.01,
            "MSFT": rng.standard_normal(200) * 0.01,
        }
    )
    recorder.capture(
        diagnostics_model.get_variance_ratio_test(returns_df, q=4).round(4)
    )


def test_get_variance_ratio_test_too_few_observations(recorder):
    returns = pd.Series([0.01, 0.02, 0.01])
    recorder.capture(diagnostics_model.get_variance_ratio_test(returns, q=5))


def test_get_cusum_test_stable_mean(recorder):
    rng = np.random.default_rng(7)
    stable = pd.Series(rng.standard_normal(300) * 0.01)
    result = diagnostics_model.get_cusum_test(stable)
    assert not result["Reject Stability (5%)"]
    recorder.capture(result.round(4))


def test_get_cusum_test_mean_shift(recorder):
    rng = np.random.default_rng(7)
    shifted = pd.Series(
        np.concatenate(
            [
                rng.standard_normal(150) * 0.01,
                rng.standard_normal(150) * 0.01 + 0.05,
            ]
        )
    )
    result = diagnostics_model.get_cusum_test(shifted)
    assert result["Reject Stability (5%)"]
    recorder.capture(result.round(4))


def test_get_cusum_test_dataframe(recorder):
    rng = np.random.default_rng(42)
    returns_df = pd.DataFrame(
        {
            "AAPL": rng.standard_normal(200) * 0.01,
            "MSFT": rng.standard_normal(200) * 0.01,
        }
    )
    recorder.capture(diagnostics_model.get_cusum_test(returns_df).round(4))


def test_get_cusum_test_too_few_observations(recorder):
    returns = pd.Series([0.01, 0.02, 0.01])
    recorder.capture(diagnostics_model.get_cusum_test(returns))


def test_get_cusum_test_invalid_type():
    try:
        diagnostics_model.get_cusum_test(123)
        raise AssertionError("Expected TypeError")
    except TypeError:
        pass


def test_get_level_significance():
    months = pd.period_range("2002-03", periods=5, freq="M")
    rates = pd.Series([0.0528, 0.0516, 0.04, np.nan, 0.0513], index=months)

    significance = diagnostics_model.get_level_significance(rates)

    # The last month at least as high is 2002-04, the last at least as low 2002-05; of the
    # three earlier values one is below the latest (the missing month is left out).
    assert significance["Highest Since"] == "2002-04"
    assert significance["Lowest Since"] == "2002-05"
    assert significance["Percentile"] == pytest.approx(1 / 3)
    assert significance["Start Date"] == "2002-03"
    assert significance["Observations"] == 4  # noqa: PLR2004
    assert significance["Latest Date"] == "2002-07"
    assert significance["Latest Value"] == pytest.approx(0.0513)


def test_get_level_significance_ties_and_records():
    days = pd.period_range("2026-01-01", periods=4, freq="D")

    # An unchanged rate is in the middle of its ties, and was as high and as low the day before.
    flat = diagnostics_model.get_level_significance(pd.Series([0.02] * 4, index=days))
    assert flat["Percentile"] == pytest.approx(0.5)
    assert flat["Highest Since"] == flat["Lowest Since"] == "2026-01-03"

    # A record high has no earlier date as high.
    record = diagnostics_model.get_level_significance(
        pd.Series([0.01, 0.03, 0.02, 0.04], index=days)
    )
    assert record["Percentile"] == 1
    assert pd.isna(record["Highest Since"])
    assert record["Lowest Since"] == "2026-01-03"

    # A single observation has no history to compare with.
    single = diagnostics_model.get_level_significance(pd.Series([0.01], index=days[:1]))
    assert pd.isna(single["Percentile"]) and pd.isna(single["Highest Since"])

    with pytest.raises(ValueError, match="no observations"):
        diagnostics_model.get_level_significance(pd.Series([np.nan], index=days[:1]))


def test_get_streaks():
    months = pd.period_range("2022-01", periods=10, freq="M")
    inflation = pd.Series(
        [0.02, 0.01, -0.01, -0.02, np.nan, -0.01, 0.01, -0.03, 0.02, 0.03], index=months
    )

    streaks = diagnostics_model.get_streaks(inflation)

    # Positive since 2022-09 (two months), after one negative month; the longest earlier
    # positive run is the first one (the missing month is left out, so it does not break a run).
    assert streaks["Sign"] == "Positive"
    assert (streaks["Run Start"], streaks["Run Length"]) == ("2022-09", 2)
    assert streaks["Previous Sign"] == "Negative"
    assert (streaks["Previous Run Start"], streaks["Previous Run End"]) == (
        "2022-08",
        "2022-08",
    )
    assert streaks["Previous Run Length"] == 1
    assert (
        streaks["Longest Earlier Run Start"],
        streaks["Longest Earlier Run End"],
        streaks["Longest Earlier Run Length"],
    ) == ("2022-01", "2022-02", 2)
    assert streaks["Observations"] == 9  # noqa: PLR2004
    assert streaks["Latest Value"] == pytest.approx(0.03)

    # Negative the whole time except the run in 2022-03 to 2022-06 (three observations).
    negative = diagnostics_model.get_streaks(-inflation.iloc[:8])
    assert (negative["Sign"], negative["Run Start"], negative["Run Length"]) == (
        "Positive",
        "2022-08",
        1,
    )
    assert negative["Longest Earlier Run Length"] == 3  # noqa: PLR2004
    assert negative["Longest Earlier Run Start"] == "2022-03"


def test_get_streaks_zeros_ties_and_single_runs():
    days = pd.period_range("2026-01-01", periods=7, freq="D")

    # A zero is a run of its own; of two equally long earlier runs the latest is reported.
    zero = diagnostics_model.get_streaks(
        pd.Series([1.0, -1.0, 1.0, 0.0, 1.0, -1.0, 1.0], index=days)
    )
    assert zero["Previous Sign"] == "Negative"
    assert zero["Longest Earlier Run Start"] == "2026-01-05"
    assert zero["Longest Earlier Run Length"] == 1

    # A series that never changed sign has no previous or earlier run.
    single = diagnostics_model.get_streaks(pd.Series([0.0] * 3, index=days[:3]))
    assert (single["Sign"], single["Run Start"], single["Run Length"]) == (
        "Zero",
        "2026-01-01",
        3,
    )
    assert pd.isna(single["Previous Sign"]) and pd.isna(
        single["Longest Earlier Run Length"]
    )

    with pytest.raises(ValueError, match="no observations"):
        diagnostics_model.get_streaks(pd.Series([np.nan], index=days[:1]))


def test_get_level_significance_change_since_start():
    months = pd.period_range("2021-01", periods=3, freq="M")

    # US housing starts from January 2021 to their latest month.
    starts = diagnostics_model.get_level_significance(
        pd.Series([1624.0, 1500.0, 1221.0], index=months)
    )
    assert starts["Start Value"] == 1624  # noqa: PLR2004
    assert starts["Change Since Start"] == pytest.approx(1221 / 1624 - 1)

    # A rise from a negative value is positive; from zero there is no relative change.
    negative = diagnostics_model.get_level_significance(
        pd.Series([-0.02, 0.0, 0.01], index=months)
    )
    assert negative["Change Since Start"] == pytest.approx(1.5)
    zero = diagnostics_model.get_level_significance(
        pd.Series([0.0, 0.01, 0.02], index=months)
    )
    assert pd.isna(zero["Change Since Start"])


def test_get_level_significance_highest_and_lowest():
    days = pd.period_range("2026-02-27", periods=5, freq="D")

    # Brent from the close before the war: the peak, its date and the fall from it.
    brent = diagnostics_model.get_level_significance(
        pd.Series([72.48, 118.35, np.nan, 71.57, 92.17], index=days)
    )
    assert brent["Highest"] == pytest.approx(118.35)
    assert brent["Highest Date"] == "2026-02-28"
    assert brent["Change From Highest"] == pytest.approx(92.17 / 118.35 - 1)
    assert brent["Lowest"] == pytest.approx(71.57)
    assert brent["Lowest Date"] == "2026-03-02"
    assert brent["Change From Lowest"] == pytest.approx(92.17 / 71.57 - 1)

    # The latest value counts: at a new peak the change from the highest is zero, and of
    # equal values the first date is the one reached first.
    peak = diagnostics_model.get_level_significance(
        pd.Series([1.0, 3.0, 2.0, 3.0, 3.0], index=days)
    )
    assert peak["Change From Highest"] == 0
    assert peak["Highest Date"] == "2026-02-28"

    # A spread that went from negative to positive: changes relative to the absolute level,
    # NaN where the level is zero.
    spread = diagnostics_model.get_level_significance(
        pd.Series([-0.02, 0.0, -0.01, 0.01, 0.005], index=days)
    )
    assert spread["Change From Lowest"] == pytest.approx((0.005 + 0.02) / 0.02)
    assert spread["Change From Highest"] == pytest.approx(-0.5)

    zero = diagnostics_model.get_level_significance(
        pd.Series([0.0, -1.0, -0.5, -0.2, -0.1], index=days)
    )
    assert pd.isna(zero["Change From Highest"])


def test_get_level_significance_compares_with_a_level():
    levels = pd.Series(
        [10.0, 30.0, 20.0, 31.0, 25.0, 32.0, 33.0],
        index=pd.period_range("2020-01", periods=7, freq="M"),
    )

    result = diagnostics_model.get_level_significance(levels, level=30.0)

    # First exceeded in April, and above it without a break since June.
    assert result["Level"] == 30.0
    assert result["First Above Level"] == "2020-04"
    assert result["Above Level Since"] == "2020-06"
    assert pd.isna(
        diagnostics_model.get_level_significance(levels.iloc[:5], level=30.0)[
            "Above Level Since"
        ]
    )
    assert "Level" not in diagnostics_model.get_level_significance(levels)


def test_window_change_by_year_ranks_the_same_months_of_every_year():
    months = pd.period_range("2021-01", "2024-06", freq="M")
    values = pd.Series(100.0, index=months)
    values[pd.Period("2021-10", "M")] = 105
    values[pd.Period("2022-10", "M")] = 103
    values[pd.Period("2023-10", "M")] = 104

    window = diagnostics_model.get_window_change_by_year(values, 2, 10)

    assert window["Change"].tolist()[:3] == pytest.approx([0.05, 0.03, 0.04])
    assert window["Rank"].tolist()[:3] == [1, 3, 2]
    # 2023's 4% is the largest since 2021; 2021 has no earlier year as large.
    assert pd.isna(window.loc[2021, "Largest Since"])
    assert window.loc[2023, "Largest Since"] == 2021
    # October 2024 is not there yet.
    assert pd.isna(window.loc[2024, "Change"])

    # A window over the turn of the year is labelled by the year it ends.
    turn = diagnostics_model.get_window_change_by_year(values, 11, 2)
    assert turn.index.tolist() == [2021, 2022, 2023, 2024]
    assert pd.isna(turn.loc[2021, "Change"])

    with pytest.raises(ValueError, match="between 1 and 12"):
        diagnostics_model.get_window_change_by_year(values, 0, 10)
