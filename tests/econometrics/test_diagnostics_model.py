"""Risk Diagnostics Model Tests"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from financetoolkit.econometrics.diagnostics import diagnostics_model

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


def test_get_level_significance_counts_and_dates_the_observations_on_either_side():
    weeks = pd.PeriodIndex(
        pd.date_range("2019-01-05", periods=8, freq="W-SAT"), freq="D"
    )
    claims = pd.Series(
        [210.0, 203.0, 199.0, 220.0, 203.0, 250.0, 260.0, 202.0], index=weeks
    )

    result = diagnostics_model.get_level_significance(claims, level=203.0)

    # At or below counts a value equal to the level; the latest week is counted, but the
    # last dates are the earlier weeks before it.
    assert result["Observations Above Level"] == 4
    assert result["Observations At Or Below Level"] == 4
    assert (
        result["Observations Above Level"] + result["Observations At Or Below Level"]
        == result["Observations"]
    )
    assert result["Last Above Level"] == str(weeks[6])
    assert result["Last At Or Below Level"] == str(weeks[4])

    # Never on one side before the latest observation: NaN, not the latest date.
    rising = pd.Series([1.0, 2.0, 3.0], index=weeks[:3])
    first_above = diagnostics_model.get_level_significance(rising, level=2.5)
    assert pd.isna(first_above["Last Above Level"])
    assert first_above["Last At Or Below Level"] == str(weeks[1])
    assert first_above["Observations Above Level"] == 1
    never = diagnostics_model.get_level_significance(rising, level=0.5)
    assert pd.isna(never["Last At Or Below Level"])
    assert never["Observations At Or Below Level"] == 0
    assert "Observations Above Level" not in diagnostics_model.get_level_significance(
        rising
    )


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


def test_level_significance_gives_the_annualised_change_and_the_largest_drawdown():
    months = pd.period_range("2000-01", "2010-01", freq="M")
    values = pd.Series(100.0, index=months)
    values[pd.Period("2005-01", "M") :] = 150.0
    values[pd.Period("2006-01", "M") : pd.Period("2007-12", "M")] = 90.0
    values[pd.Period("2010-01", "M")] = 200.0

    significance = diagnostics_model.get_level_significance(values)

    assert significance["Multiple Since Start"] == pytest.approx(2.0)
    # Doubling over ten years.
    assert significance["Annualised Change"] == pytest.approx(
        2 ** (1 / 10) - 1, abs=1e-4
    )
    assert significance["Largest Drawdown"] == pytest.approx(-0.4)
    # The level stays at its peak of 150 through 2005 and falls from December.
    assert significance["Drawdown Peak Date"] == "2005-12"
    assert significance["Drawdown Trough Date"] == "2006-01"
    # 150 again from January 2008.
    assert significance["Recovery Date"] == "2008-01"


def test_streaks_count_runs_above_and_below_a_level():
    years = pd.period_range("2000", "2009", freq="Y")
    inflation = pd.Series(
        [0.03, 0.01, 0.01, 0.015, 0.03, 0.025, 0.04, 0.01, 0.01, 0.03], index=years
    )

    streaks = diagnostics_model.get_streaks(inflation, level=0.02)

    assert streaks["Sign"] == "Above Level"
    assert streaks["Run Length"] == 1
    assert streaks["Previous Sign"] == "Below Level"
    assert streaks["Longest Earlier Run Length"] == 3


def test_level_significance_ranks_the_latest_value():
    months = pd.period_range("2026-01", "2026-06", freq="M")
    values = pd.Series([0.3, 0.5, 0.1, 0.5, 0.2, 0.4], index=months)

    # Two earlier months were higher.
    assert diagnostics_model.get_level_significance(values)["Rank"] == 3


def test_level_significance_in_points_for_a_rate():
    rates = pd.Series(
        [0.0529, 0.0510, 0.0540, 0.0494],
        index=pd.period_range("2026-09-30", periods=4, freq="D"),
    )

    relative = diagnostics_model.get_level_significance(rates)
    points = diagnostics_model.get_level_significance(rates, difference=True)

    assert relative["Change Since Start"] == pytest.approx(0.0494 / 0.0529 - 1)
    assert points["Change Since Start"] == pytest.approx(-0.0035)
    assert points["Change From Highest"] == pytest.approx(-0.0046)
    assert points["Change From Lowest"] == pytest.approx(0.0)
    assert points["Largest Drawdown"] == pytest.approx(-0.0046)
    assert pd.isna(points["Multiple Since Start"])
    assert pd.isna(points["Annualised Change"])


def test_window_change_by_year_in_points():
    months = pd.period_range("2024-01", "2025-12", freq="M")
    rates = pd.Series(np.linspace(0.05, 0.03, len(months)), index=months)

    window = diagnostics_model.get_window_change_by_year(rates, 2, 10, difference=True)

    assert window.loc[2025, "Change"] == pytest.approx(
        rates[pd.Period("2025-10", "M")] - rates[pd.Period("2025-02", "M")]
    )


def test_window_change_by_year_ranks_and_years_are_whole_numbers():
    months = pd.period_range("2021-01", "2024-06", freq="M")
    values = pd.Series(100.0, index=months)
    values[pd.Period("2021-10", "M")] = 105
    values[pd.Period("2023-10", "M")] = 104

    window = diagnostics_model.get_window_change_by_year(values, 2, 10)

    assert str(window["Rank"].dtype) == "Int64"
    assert str(window["Largest Since"].dtype) == "Int64"
    # 2024 has no change yet and 2021 no earlier year: missing, not a float year.
    assert window.loc[2024, "Rank"] is pd.NA
    assert window.loc[2021, "Largest Since"] is pd.NA
    assert window.loc[2023, "Largest Since"] == 2021


def test_get_cusum_test_of_a_constant_series_is_not_defined():
    # A constant series fitted on a constant leaves residuals of rounding noise, which must not
    # give a statistic of noise over noise.
    for level in [0.01, 1.0, -0.003]:
        result = diagnostics_model.get_cusum_test(pd.Series([level] * 30))
        assert np.isnan(result["CUSUM Statistic"])
        assert not result["Reject Stability (5%)"]


def test_get_level_significance_dates_the_drawdown_from_the_last_visit_to_the_peak():
    # The level returns to its peak of 110 before falling to 90, so the decline starts from
    # the second visit.
    values = pd.Series(
        [100, 110, 105, 110, 90, 95.0],
        index=pd.period_range("2024-01-01", periods=6, freq="D"),
    )
    result = diagnostics_model.get_level_significance(values)
    assert str(result["Drawdown Peak Date"]) == "2024-01-04"
    assert str(result["Drawdown Trough Date"]) == "2024-01-05"


def test_get_cusum_test_of_a_series_constant_up_to_rounding_is_not_defined():
    # The steps of a cumulated constant differ by rounding only (spread 4e-16).
    steps = pd.Series(np.diff(np.cumsum(np.full(40, 0.1))))
    result = diagnostics_model.get_cusum_test(steps)
    assert np.isnan(result["CUSUM Statistic"])
    assert not result["Reject Stability (5%)"]
    # Real returns with little spread keep their statistic.
    returns = pd.Series(np.random.default_rng(1).normal(0.001, 0.0001, 50))
    assert np.isfinite(diagnostics_model.get_cusum_test(returns)["CUSUM Statistic"])


def test_get_cusum_test_recursive_rejects_a_mean_shift_late_in_the_sample(recorder):
    rng = np.random.default_rng(7)
    stable = pd.Series(rng.standard_normal(300) * 0.01)
    shifted = pd.Series(
        np.concatenate(
            [rng.standard_normal(240) * 0.01, rng.standard_normal(60) * 0.01 + 0.02]
        )
    )
    assert not diagnostics_model.get_cusum_test(stable, method="recursive")[
        "Reject Stability (5%)"
    ]
    result = diagnostics_model.get_cusum_test(shifted, method="recursive")
    assert result["Reject Stability (5%)"]
    recorder.capture(result.round(4))


def test_get_cusum_test_recursive_follows_brown_durbin_and_evans():
    # The recursive residuals of a constant-mean model are the forecast errors of the mean of
    # the returns before each date, scaled by sqrt(1 + 1 / (t - 1)); their squares add up to
    # the sum of squared OLS residuals.
    rng = np.random.default_rng(3)
    returns = rng.standard_normal(50)
    recursive = np.array(
        [
            (returns[t] - returns[:t].mean()) / np.sqrt(1 + 1 / t)
            for t in range(1, len(returns))
        ]
    )
    np.testing.assert_allclose(
        np.sum(recursive**2), np.sum((returns - returns.mean()) ** 2), rtol=1e-12
    )
    sigma = np.sqrt(np.sum(recursive**2) / (len(returns) - 1))
    boundary = np.sqrt(49) + 2 * np.arange(1, 50) / np.sqrt(49)
    statistic = np.max(np.abs(np.cumsum(recursive) / sigma) / boundary)

    result = diagnostics_model.get_cusum_test(pd.Series(returns), method="recursive")

    assert result["CUSUM Statistic"] == pytest.approx(statistic, rel=1e-12)
    # The crossing probability of Brown, Durbin and Evans (1975) at their tabulated values.
    for level, critical_value in ((0.01, 1.143), (0.05, 0.948), (0.10, 0.850)):
        crossing = 2 * (
            1
            - stats.norm.cdf(3 * critical_value)
            + np.exp(-4 * critical_value**2) * stats.norm.cdf(critical_value)
        )
        assert crossing == pytest.approx(level, abs=1e-4)


def test_get_cusum_test_chu_stinchcombe_white_by_hand():
    returns = pd.Series(
        [0.01, -0.02, 0.015, 0.03, 0.02, -0.01, 0.025, 0.02, 0.03, 0.01, 0.02, 0.015]
    )
    levels = np.concatenate([[0.0], np.cumsum(np.log1p(returns.to_numpy()))])
    t = len(returns)
    sigma = np.sqrt(np.mean(np.diff(levels) ** 2))
    statistics = [(levels[t] - levels[n]) / (sigma * np.sqrt(t - n)) for n in range(t)]
    reference = int(np.argmax(statistics))

    result = diagnostics_model.get_cusum_test(returns, method="chu_stinchcombe_white")

    assert result["CUSUM Statistic"] == pytest.approx(max(statistics), rel=1e-12)
    assert result["Critical Value 5%"] == pytest.approx(
        np.sqrt(4.6 + np.log(t - reference)), rel=1e-12
    )
    assert result["Observations"] == t


def test_get_cusum_test_chu_stinchcombe_white_detects_a_rise_without_look_ahead(
    recorder,
):
    rng = np.random.default_rng(11)
    returns = pd.Series(
        np.concatenate([rng.normal(0, 0.01, 200), rng.normal(0.008, 0.01, 100)]),
        index=pd.bdate_range("2020-01-01", periods=300),
    )
    path = diagnostics_model.get_cusum_test(
        returns, method="chu_stinchcombe_white", time_series=True
    )
    summary = diagnostics_model.get_cusum_test(returns, method="chu_stinchcombe_white")

    assert summary["Reject Stability (5%)"]
    assert summary["CUSUM Statistic"] == path["CUSUM Statistic"].iloc[-1]
    # The first nine dates have too few returns for a verdict.
    assert path["CUSUM Statistic"].iloc[:9].isna().all()
    # A date's statistic only uses the returns up to it.
    truncated = diagnostics_model.get_cusum_test(
        returns.iloc[:150], method="chu_stinchcombe_white", time_series=True
    )
    pd.testing.assert_frame_equal(truncated, path.iloc[:150])
    # The test is one-sided: the same fall is not a rise.
    assert not diagnostics_model.get_cusum_test(
        -returns, method="chu_stinchcombe_white"
    )["Reject Stability (5%)"]
    recorder.capture(path.iloc[::25].astype(float).round(4))


def test_get_cusum_test_time_series_needs_chu_stinchcombe_white():
    with pytest.raises(ValueError, match="chu_stinchcombe_white"):
        diagnostics_model.get_cusum_test(
            pd.Series(np.ones(20)), method="recursive", time_series=True
        )
    with pytest.raises(ValueError, match="method"):
        diagnostics_model.get_cusum_test(pd.Series(np.ones(20)), method="mosum")


def test_get_ljung_box_test_weighted_by_hand(recorder):
    rng = np.random.default_rng(5)
    values = rng.standard_normal(400)
    values[1:] += 0.2 * values[:-1]
    n, lags = len(values), 6
    demeaned = values - values.mean()
    autocorrelations = np.array(
        [
            demeaned[k:] @ demeaned[:-k] / (demeaned @ demeaned)
            for k in range(1, lags + 1)
        ]
    )
    k = np.arange(1, lags + 1)
    statistic = (
        n * (n + 2) * np.sum((lags - k + 1) / lags * autocorrelations**2 / (n - k))
    )
    shape = 0.75 * (lags + 1) ** 2 * lags / (2 * lags**2 + 3 * lags + 1)
    scale = 2 / 3 * (2 * lags**2 + 3 * lags + 1) / (lags * (lags + 1))

    result = diagnostics_model.get_ljung_box_test(
        pd.Series(values), lags=lags, weighted=True
    )

    assert result["Weighted Ljung-Box Statistic"] == pytest.approx(statistic, rel=1e-12)
    assert result["P-Value"] == pytest.approx(
        stats.gamma.sf(statistic, shape, scale=scale), rel=1e-10
    )
    assert result["P-Value"] < LOW_P_VALUE_THRESHOLD
    recorder.capture(result.round(4))


def test_get_arch_lm_test_weighted_gamma_of_rugarch():
    # rugarch's weighted ARCH test of the third lag after a GARCH(1, 1) (lags=3, model_df=2)
    # has the Gamma shape 0.5 and scale 2 printed in its vignette: a chi-squared with 1
    # degree of freedom of n times the third squared autocorrelation.
    rng = np.random.default_rng(9)
    values = rng.standard_normal(500)
    squared = (values - values.mean()) ** 2
    demeaned = squared - squared.mean()
    third = demeaned[3:] @ demeaned[:-3] / (demeaned @ demeaned)

    result = diagnostics_model.get_arch_lm_test(
        pd.Series(values), lags=3, weighted=True, model_df=2
    )

    assert result["Weighted ARCH-LM Statistic"] == pytest.approx(
        500 * third**2, rel=1e-12
    )
    assert result["P-Value"] == pytest.approx(
        stats.chi2.sf(500 * third**2, 1), rel=1e-10
    )
    with pytest.raises(ValueError, match="model_df"):
        diagnostics_model.get_arch_lm_test(
            pd.Series(values), lags=3, weighted=True, model_df=3
        )


def test_get_arch_lm_test_weighted_detects_volatility_clustering(recorder):
    rng = np.random.default_rng(42)
    calm = pd.Series(rng.standard_normal(500) * 0.01)
    variance = np.empty(500)
    clustered = np.empty(500)
    variance[0], clustered[0] = 1e-4, 0.0
    for t in range(1, 500):
        variance[t] = 1e-6 + 0.2 * clustered[t - 1] ** 2 + 0.75 * variance[t - 1]
        clustered[t] = np.sqrt(variance[t]) * rng.standard_normal()

    assert (
        diagnostics_model.get_arch_lm_test(calm, weighted=True)["P-Value"]
        > HIGH_P_VALUE_THRESHOLD
    )
    result = diagnostics_model.get_arch_lm_test(pd.Series(clustered), weighted=True)
    assert result["P-Value"] < LOW_P_VALUE_THRESHOLD
    recorder.capture(result.round(4))


def _gjr_garch(gamma: float, n: int = 3000, seed: int = 1):
    """Residuals and standardized residuals of a GJR-GARCH(1, 1) with leverage gamma."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    variance = np.empty(n)
    residuals = np.empty(n)
    variance[0] = 1e-4
    residuals[0] = np.sqrt(variance[0]) * z[0]
    for t in range(1, n):
        shock = residuals[t - 1]
        variance[t] = (
            2e-6 + (0.05 + gamma * (shock < 0)) * shock**2 + 0.85 * variance[t - 1]
        )
        residuals[t] = np.sqrt(variance[t]) * z[t]
    return residuals, residuals / np.sqrt(variance)


def test_get_sign_bias_test_detects_leverage(recorder):
    # Standardized with the symmetric variance, a GJR process leaves the leverage in the
    # squared standardized residuals; standardized with its own variance it does not.
    residuals, standardized = _gjr_garch(0.15)
    symmetric = residuals / np.sqrt(
        pd.Series(residuals**2).ewm(alpha=0.06).mean().shift(1).bfill().to_numpy()
    )
    leverage = diagnostics_model.get_sign_bias_test(
        pd.Series(residuals), pd.Series(symmetric)
    )
    own = diagnostics_model.get_sign_bias_test(
        pd.Series(residuals), pd.Series(standardized)
    )

    assert leverage["Negative Size Bias t-Statistic"] < -2
    assert leverage["Joint P-Value"] < LOW_P_VALUE_THRESHOLD
    assert own["Joint P-Value"] > HIGH_P_VALUE_THRESHOLD
    recorder.capture(leverage.round(4))


def test_get_sign_bias_test_by_hand():
    residuals, standardized = _gjr_garch(0.1, n=400, seed=4)
    shocks, squared = residuals[:-1], standardized[1:] ** 2
    negative = (shocks < 0).astype(float)
    design = np.column_stack(
        [np.ones(len(shocks)), negative, negative * shocks, (1 - negative) * shocks]
    )
    coefficients, *_ = np.linalg.lstsq(design, squared, rcond=None)
    errors = squared - design @ coefficients
    covariance = errors @ errors / (len(squared) - 4) * np.linalg.inv(design.T @ design)
    t_statistics = coefficients / np.sqrt(np.diag(covariance))
    r_squared = 1 - errors @ errors / np.sum((squared - squared.mean()) ** 2)

    result = diagnostics_model.get_sign_bias_test(
        pd.Series(residuals), pd.Series(standardized)
    )

    np.testing.assert_allclose(
        result[
            [
                "Sign Bias t-Statistic",
                "Negative Size Bias t-Statistic",
                "Positive Size Bias t-Statistic",
            ]
        ].to_numpy(dtype=float),
        t_statistics[1:],
        rtol=1e-9,
    )
    assert result["Joint Statistic"] == pytest.approx(399 * r_squared, rel=1e-9)
    assert result["Observations"] == 399


def test_get_pearson_goodness_of_fit_test_by_hand():
    # Residuals at the middle of 20 equal-probability bins fill every bin equally.
    even = pd.Series(stats.norm.ppf((np.arange(200) + 0.5) / 200))
    assert diagnostics_model.get_pearson_goodness_of_fit_test(even)[
        "Pearson Statistic"
    ] == pytest.approx(0, abs=1e-12)

    # Every residual below the median: two full bins and two empty ones out of four, and the
    # empty bins above the last occupied one count as well.
    low = pd.Series(stats.norm.ppf((np.arange(100) + 0.5) / 200))
    result = diagnostics_model.get_pearson_goodness_of_fit_test(low, bins=4)
    assert result["Pearson Statistic"] == pytest.approx(100.0)
    assert result["P-Value"] == pytest.approx(stats.chi2.sf(100.0, 3))


def test_get_pearson_goodness_of_fit_test_tells_the_distributions_apart(recorder):
    rng = np.random.default_rng(2)
    # Student-t draws with 4 degrees of freedom, scaled to a unit variance.
    draws = pd.Series(rng.standard_t(4, 3000) / np.sqrt(2))

    normal = diagnostics_model.get_pearson_goodness_of_fit_test(draws)
    student = diagnostics_model.get_pearson_goodness_of_fit_test(
        draws, distribution="t", parameters=(4.0,)
    )

    assert normal["P-Value"] < LOW_P_VALUE_THRESHOLD
    assert student["P-Value"] > HIGH_P_VALUE_THRESHOLD
    recorder.capture(pd.DataFrame({"Normal": normal, "Student-t": student}).round(4))


@pytest.mark.parametrize(
    ("statistic", "p_value"),
    # The 10%, 5% and 1% quantiles of the Cramér-von Mises limit (Anderson and Darling, 1952).
    [(0.34730, 0.10), (0.46136, 0.05), (0.74346, 0.01)],
)
def test_nyblom_p_value_of_one_parameter_is_the_cramer_von_mises_limit(
    statistic, p_value
):
    # The quantiles are tabulated to five decimals, which moves the p-value by up to 3e-6.
    assert diagnostics_model.get_nyblom_p_value(statistic, 1) == pytest.approx(
        p_value, abs=1e-5
    )


def test_nyblom_p_value_of_several_parameters_matches_a_simulation():
    rng = np.random.default_rng(0)
    weights = 1 / (np.arange(1, 401) ** 2 * np.pi**2)
    # Σ_j ∫ B_j^2 = Σ_n Σ_j Z_nj^2 / (n^2 π^2) for three bridges.
    draws = (rng.standard_normal((100000, 400)) ** 2 @ weights).reshape(-1, 1)
    draws = draws[:99999].reshape(33333, 3).sum(axis=1)

    for statistic in (0.6, 1.01, 1.6):
        assert diagnostics_model.get_nyblom_p_value(statistic, 3) == pytest.approx(
            (draws > statistic).mean(), abs=4e-3
        )
    # Hansen's (1992) simulated 5% critical values are within 1% of the limiting ones.
    for parameters, critical in ((1, 0.470), (2, 0.749), (3, 1.01), (5, 1.47)):
        assert diagnostics_model.get_nyblom_p_value(
            critical, parameters
        ) == pytest.approx(0.05, abs=3e-3)


def test_nyblom_test_detects_a_change_in_the_parameters():
    rng = np.random.default_rng(1)
    stable = rng.standard_normal((1000, 2))
    # The scores of a mean that moves from -0.3 to 0.3 halfway (the fit sits at 0).
    shifted = stable + np.r_[np.full(500, -0.3), np.full(500, 0.3)][:, None] * [1, 0]

    # The scores of a fit sum to zero, so both sets are centred.
    calm = diagnostics_model.get_nyblom_test(
        stable - stable.mean(axis=0), ["Mu", "Sigma"]
    )
    broken = diagnostics_model.get_nyblom_test(
        shifted - shifted.mean(axis=0), ["Mu", "Sigma"]
    )

    assert calm["Joint P-Value"] > 0.05
    assert broken["Mu P-Value"] < 1e-3
    assert broken["Sigma P-Value"] > 0.01
    # The individual statistic is Σ S_t^2 / (T V), V the sum of the squared scores.
    scores = shifted[:, 0] - shifted[:, 0].mean()
    assert broken["Mu Statistic"] == pytest.approx(
        np.sum(np.cumsum(scores) ** 2) / (1000 * np.sum(scores**2)), rel=1e-12
    )
    assert broken["Individual Critical Value (5%)"] == pytest.approx(0.4614, abs=1e-4)


def test_gmm_moment_test_matches_t_tests_of_the_moments():
    rng = np.random.default_rng(2)
    z = pd.Series(rng.standard_t(10, 1500) * np.sqrt(8 / 10))

    result = diagnostics_model.get_gmm_moment_test(z, 1, "t", (10.0,))

    # The moments of the standardized Student-t with 10 degrees of freedom: E[z^4] = 3 + 6 / 6
    # and E[z^8] = 105 (v - 2)^4 / ((v - 4)(v - 6)(v - 8)) = 1,120.
    values = z.to_numpy()[1:]
    assert result["Mean t-Statistic"] == pytest.approx(
        values.mean() * np.sqrt(1499), rel=1e-12
    )
    assert result["Kurtosis t-Statistic"] == pytest.approx(
        np.mean(values**4 - 4) / np.sqrt((1120 - 16) / 1499), rel=1e-6
    )
    # One lagged condition: its variance under the null is E[(z^2 - 1)^2]^2.
    squared = z.to_numpy() ** 2 - 1
    assert result["Q2 Statistic"] == pytest.approx(
        1499
        * np.mean(squared[1:] * squared[:-1]) ** 2
        / np.mean(squared[1:] ** 2) ** 2,
        rel=1e-12,
    )
    assert result["Joint P-Value"] == pytest.approx(
        stats.chi2.sf(result["Joint Statistic"], 7), rel=1e-12
    )
    # A Student-t with a Shape of 3.5 has no kurtosis: the Kurtosis and Q4 conditions are
    # left out, and the moments it has no 4th and 6th moment for take their sample variance.
    without = diagnostics_model.get_gmm_moment_test(z, 1, "t", (3.5,))
    assert np.isnan(without["Kurtosis t-Statistic"]) and np.isnan(
        without["Q4 Statistic"]
    )
    assert without["Skewness t-Statistic"] == pytest.approx(
        stats.ttest_1samp(values**3, 0).statistic, rel=1e-12
    )
    assert without["Joint P-Value"] == pytest.approx(
        stats.chi2.sf(without["Joint Statistic"], 5), rel=1e-12
    )


def test_gmm_moment_test_takes_the_moment_covariance_from_the_distribution():
    rng = np.random.default_rng(8)
    z = rng.standard_normal(800)
    result = diagnostics_model.get_gmm_moment_test(pd.Series(z), lags=1)

    # Normal moments: Var z = 1, Cov(z, z^3) = 3, Var(z^2 - 1) = 2, Cov(z^2 - 1, z^4 - 3) = 12,
    # Var z^3 = 15 and Var(z^4 - 3) = 96; the lagged products keep E[a c]^2 of the sample.
    centred = np.column_stack([z, z**2 - 1, z**3, z**4 - 3])
    conditions = np.column_stack([centred[1:], centred[1:, 1:] * centred[:-1, 1:]])
    products = centred[1:, 1:].T @ centred[1:, 1:] / 799
    covariance = np.zeros((7, 7))
    covariance[:4, :4] = [[1, 0, 3, 0], [0, 2, 0, 12], [3, 0, 15, 0], [0, 12, 0, 96]]
    covariance[4:, 4:] = products**2
    average = conditions.mean(axis=0)

    assert result["Joint Statistic"] == pytest.approx(
        799 * average @ np.linalg.solve(covariance, average), rel=1e-10
    )
    assert result["Kurtosis t-Statistic"] == pytest.approx(
        average[3] / np.sqrt(96 / 799), rel=1e-12
    )

    # A series without variation has nothing to test.
    constant = diagnostics_model.get_gmm_moment_test(pd.Series(np.full(300, 0.3)))
    assert constant.drop(["Lags", "Observations"]).isna().all()


def test_gmm_moment_test_keeps_its_size_under_the_null():
    rng = np.random.default_rng(5)
    results = pd.DataFrame(
        [
            diagnostics_model.get_gmm_moment_test(pd.Series(rng.standard_normal(2500)))
            for _ in range(300)
        ]
    )

    # The sample covariance of all seven conditions rejects about 20% of the time at 5%;
    # the covariance under the null about 5% (5.3% over 2,000 replications).
    assert (results["Joint P-Value"] < 0.05).mean() < 0.09
    assert results["Joint Statistic"].mean() == pytest.approx(7, abs=0.8)
    for group in ("Q2", "Q3", "Q4"):
        assert (results[f"{group} P-Value"] < 0.05).mean() < 0.09


def test_ljung_box_test_is_nan_without_variation():
    # Around a constant non-zero value the deviations from the mean are rounding noise with
    # autocorrelations close to 1, which would reject at any level.
    constant = pd.Series(np.full(250, 0.0123))

    for weighted in (False, True):
        result = diagnostics_model.get_ljung_box_test(constant, 10, weighted=weighted)
        assert result.isna().all()
