"""CoVaR Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.risk.dependence import covar_model

QUANTILE_REGRESSION_TOLERANCE = 0.2
CRASH_PROBABILITY = 0.05
TAIL_LINKED_SLOPE_MINIMUM = 0.2


def test_quantile_regression_matches_ols_at_median(recorder):
    rng = np.random.default_rng(2)
    n = 1000
    x = rng.standard_normal(n)
    y = 2 + 3 * x + rng.standard_normal(n)

    intercept, slope = covar_model._quantile_regression(y, x, 0.5)  # noqa: SLF001
    assert abs(intercept - 2) < QUANTILE_REGRESSION_TOLERANCE
    assert abs(slope - 3) < QUANTILE_REGRESSION_TOLERANCE
    recorder.capture((round(intercept, 4), round(slope, 4)))


def test_get_covar_tail_linked(recorder):
    rng = np.random.default_rng(6)
    n = 2000

    crash_factor = rng.standard_normal(n)
    crash_factor[rng.random(n) < CRASH_PROBABILITY] -= 5
    conditioning = pd.Series(crash_factor * 0.01 + rng.standard_normal(n) * 0.01)
    returns = pd.Series(crash_factor * 0.008 + rng.standard_normal(n) * 0.015)

    result = covar_model.get_covar(returns, conditioning, alpha=0.05)
    unconditional_var = np.percentile(returns, 5)

    # With genuine tail linkage, CoVaR should be materially worse than plain VaR.
    assert result["CoVaR"] < unconditional_var
    assert result["Quantile Regression Slope"] > TAIL_LINKED_SLOPE_MINIMUM
    recorder.capture(result.round(4))


def test_get_covar_independent(recorder):
    rng = np.random.default_rng(6)
    n = 2000
    returns = pd.Series(rng.standard_normal(n) * 0.02)
    conditioning = pd.Series(rng.standard_normal(n) * 0.02)
    result = covar_model.get_covar(returns, conditioning, alpha=0.05)
    recorder.capture(result.round(4))


def test_get_covar_too_few_observations(recorder):
    returns = pd.Series([0.01, -0.02, 0.03])
    conditioning = pd.Series([0.02, -0.01, 0.02])
    recorder.capture(covar_model.get_covar(returns, conditioning, alpha=0.05))


def test_get_covar_invalid_type():
    try:
        covar_model.get_covar(1, pd.Series([1.0, 2.0]), alpha=0.05)  # type: ignore
        raise AssertionError("Expected TypeError")
    except TypeError:
        pass


def _daily_pair(seed=4, start="2021-01-01", end="2022-12-31"):
    """Daily stock and market returns on the weekdays between start and end."""
    rng = np.random.default_rng(seed)
    days = pd.period_range(start, end, freq="D")
    days = days[days.dayofweek < 5]  # noqa: PLR2004
    market = pd.Series(rng.normal(0.0004, 0.01, len(days)), index=days)
    stock = 1.2 * market + pd.Series(rng.normal(0, 0.01, len(days)), index=days)
    return stock.rename("Stock"), market.rename("Market")


def test_get_marginal_expected_shortfall_by_hand():
    stock, market = _daily_pair()
    mes = covar_model.get_marginal_expected_shortfall(stock, market, 0.05, "yearly")

    # The average return of the stock on the days the market is at or below its 5% quantile.
    for year in mes.index:
        in_year = stock.index.asfreq("Y") == year
        tail = market[in_year] <= np.percentile(market[in_year], 5)
        assert tail.sum() >= 13  # noqa: PLR2004
        assert np.isclose(mes[year], stock[in_year][tail].mean())
        assert mes[year] < 0


def test_get_marginal_expected_shortfall_rolling_and_short_periods():
    stock, market = _daily_pair()
    rolling = covar_model.get_marginal_expected_shortfall(
        stock, market, 0.05, "quarterly", rolling=4
    )
    yearly = covar_model.get_marginal_expected_shortfall(stock, market, 0.05, "yearly")

    # Four quarters ending in the fourth quarter are the year, and the window starts once it spans four quarters.  # noqa: E501
    assert rolling.index[0] == pd.Period("2021Q4", "Q")
    assert np.isclose(rolling[pd.Period("2022Q4", "Q")], yearly[pd.Period("2022", "Y")])

    # A month holds about 21 days, more than 1 / alpha at 5% but fewer than at 1%.
    assert (
        covar_model.get_marginal_expected_shortfall(stock, market, 0.05, "monthly")
        .notna()
        .all()
    )
    assert (
        covar_model.get_marginal_expected_shortfall(stock, market, 0.01, "monthly")
        .isna()
        .all()
    )


def test_get_marginal_expected_shortfall_of_a_dataframe_keeps_each_history():
    stock, market = _daily_pair()
    late = stock.copy()
    late[late.index < pd.Period("2022-01-01", "D")] = np.nan
    mes = covar_model.get_marginal_expected_shortfall(
        pd.DataFrame({"Stock": stock, "Late": late}), market
    )

    assert np.isnan(mes.loc[pd.Period("2021", "Y"), "Late"])
    assert np.isclose(
        mes.loc[pd.Period("2022", "Y"), "Late"],
        mes.loc[pd.Period("2022", "Y"), "Stock"],
    )


def test_get_market_beta_matches_least_squares():
    stock, market = _daily_pair()
    beta = covar_model.get_market_beta(stock, market, "yearly")

    for year in beta.index:
        in_year = stock.index.asfreq("Y") == year
        slope = np.polyfit(market[in_year], stock[in_year], 1)[0]
        assert np.isclose(beta[year], slope)


def test_long_run_marginal_expected_shortfall_and_srisk():
    # A beta of 1 falls with the market, a beta of 2 by 1 - 0.6^2 = 64%.
    lrmes = covar_model.get_long_run_marginal_expected_shortfall(
        pd.Series([0.0, 1.0, 2.0]), decline=0.4
    )
    assert np.allclose(lrmes, [0.0, -0.4, -0.64])

    # 8% of 1,000 of debt is 80; the equity of 100 keeps 36 after a fall of 64%, of which 92% counts.
    srisk = covar_model.get_srisk(1000.0, 100.0, -0.64, capital_ratio=0.08)
    assert np.isclose(srisk, 80 - 0.92 * 36)


def test_get_srisk_share_adds_only_the_shortfalls():
    srisk = pd.DataFrame(
        {"A": [30.0, -5.0, np.nan], "B": [10.0, -1.0, 4.0], "C": [-50.0, -2.0, np.nan]}
    )
    share = covar_model.get_srisk_share(srisk)

    assert np.allclose(share.iloc[0], [0.75, 0.25, 0.0])
    # Without any shortfall there is nothing to share.
    assert share.iloc[1].isna().all()
    assert share.loc[2, "B"] == 1.0
    assert share.iloc[2][["A", "C"]].isna().all()


def test_get_marginal_expected_shortfall_invalid_type():
    with pytest.raises(TypeError):
        covar_model.get_marginal_expected_shortfall([0.01], pd.Series([0.01]))  # type: ignore
