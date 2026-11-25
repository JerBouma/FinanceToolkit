"""Performance Controller Tests""" ""

import copy
from functools import partial

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from financetoolkit.performance import performance_model
from financetoolkit.performance.performance_controller import Performance
from financetoolkit.risk import risk_model
from financetoolkit.utilities.error_model import InvalidArgumentError


def test_collect_all_metrics(recorder, performance_module):
    recorder.capture(performance_module.collect_all_metrics())
    recorder.capture(performance_module.collect_all_metrics(growth=True))
    recorder.capture(performance_module.collect_all_metrics(growth=True, lag=[1, 2, 3]))


def test_get_beta(recorder, performance_module):
    recorder.capture(performance_module.get_beta())
    recorder.capture(performance_module.get_beta(growth=True))
    recorder.capture(performance_module.get_beta(growth=True, lag=[1, 2, 3]))
    recorder.capture(performance_module.get_beta(period="monthly", rolling=6))


def test_get_capital_asset_pricing_model(recorder, performance_module):
    recorder.capture(performance_module.get_capital_asset_pricing_model())
    recorder.capture(performance_module.get_capital_asset_pricing_model(growth=True))
    recorder.capture(
        performance_module.get_capital_asset_pricing_model(growth=True, lag=[1, 2, 3])
    )
    recorder.capture(
        performance_module.get_capital_asset_pricing_model(period="monthly", rolling=6)
    )


def test_get_factor_asset_correlations(recorder, performance_module, live_mode):
    if not live_mode:
        pytest.skip("Downloads Fama-French data from the internet")
    recorder.capture(
        performance_module.get_factor_asset_correlations().round(1).iloc[:10]
    )
    recorder.capture(
        performance_module.get_factor_asset_correlations(period="monthly")
        .round(1)
        .iloc[5:10]
    )
    recorder.capture(
        performance_module.get_factor_asset_correlations(
            factors_to_calculate=["HML", "Mkt-RF"]
        )
        .round(1)
        .iloc[5:10]
    )


def test_get_factor_correlations(recorder, performance_module, live_mode):
    if not live_mode:
        pytest.skip("Downloads Fama-French data from the internet")
    recorder.capture(performance_module.get_factor_correlations().round(1))
    recorder.capture(
        performance_module.get_factor_correlations(period="monthly").round(1)
    )
    recorder.capture(
        performance_module.get_factor_correlations(
            factors_to_calculate=["HML", "Mkt-RF"]
        )
        .round(1)
        .iloc[5:10]
    )
    recorder.capture(
        performance_module.get_factor_correlations(exclude_risk_free=False)
        .round(1)
        .iloc[5:10]
    )


def test_get_fama_and_french_model(recorder, performance_module, live_mode):
    if not live_mode:
        pytest.skip("Downloads Fama-French data from the internet")
    recorder.capture(
        performance_module.get_fama_and_french_model().round(2).sort_index(axis=1)
    )
    recorder.capture(
        performance_module.get_fama_and_french_model(period="monthly")
        .round(2)
        .sort_index(axis=1)
    )
    recorder.capture(
        performance_module.get_fama_and_french_model(method="multi")
        .round(2)
        .sort_index(axis=1)
    )
    recorder.capture(
        performance_module.get_fama_and_french_model(method="simple")
        .round(2)
        .sort_index(axis=1)
    )

    result1, result2 = performance_module.get_fama_and_french_model(
        include_daily_residuals=True
    )
    recorder.capture(result1.round(2).sort_index(axis=1))
    recorder.capture(result2.round(2).sort_index(axis=1))

    recorder.capture(
        performance_module.get_fama_and_french_model(growth=True)
        .round(2)
        .sort_index(axis=1)
    )
    recorder.capture(
        performance_module.get_fama_and_french_model(growth=True, lag=[1, 2, 3])
        .round(2)
        .sort_index(axis=1)
    )


def test_get_carhart_four_factor_model(recorder, performance_module, live_mode):
    if not live_mode:
        pytest.skip("Downloads Fama-French and momentum data from the internet")
    recorder.capture(
        performance_module.get_carhart_four_factor_model().round(2).sort_index(axis=1)
    )
    recorder.capture(
        performance_module.get_carhart_four_factor_model(period="monthly")
        .round(2)
        .sort_index(axis=1)
    )
    recorder.capture(
        performance_module.get_carhart_four_factor_model(growth=True)
        .round(2)
        .sort_index(axis=1)
    )


def test_get_alpha(recorder, performance_module):
    recorder.capture(performance_module.get_alpha())
    recorder.capture(performance_module.get_alpha(period="quarterly"))
    recorder.capture(performance_module.get_alpha(growth=True))
    recorder.capture(performance_module.get_alpha(growth=True, lag=[1, 2, 3]))
    recorder.capture(performance_module.get_alpha(show_full_results=True))
    recorder.capture(performance_module.get_alpha(period="monthly", rolling=6))
    recorder.capture(performance_module.get_alpha(method="jensen"))
    recorder.capture(performance_module.get_alpha(method="jensen", period="quarterly"))
    recorder.capture(performance_module.get_alpha(method="jensen", growth=True))
    recorder.capture(
        performance_module.get_alpha(method="jensen", growth=True, lag=[1, 2, 3])
    )
    recorder.capture(
        performance_module.get_alpha(method="jensen", period="monthly", rolling=6)
    )
    with pytest.raises(InvalidArgumentError, match="'jensen'"):
        performance_module.get_alpha(method="capm")


def test_get_treynor_ratio(recorder, performance_module):
    recorder.capture(performance_module.get_treynor_ratio())
    recorder.capture(performance_module.get_treynor_ratio(period="quarterly"))
    recorder.capture(performance_module.get_treynor_ratio(growth=True))
    recorder.capture(performance_module.get_treynor_ratio(growth=True, lag=[1, 2, 3]))
    recorder.capture(performance_module.get_treynor_ratio(period="monthly", rolling=6))


def test_get_sharpe_ratio(recorder, performance_module):
    recorder.capture(round(performance_module.get_sharpe_ratio(), 2))
    recorder.capture(performance_module.get_sharpe_ratio(period="quarterly"))
    # Ten years would not fit in the three years of data, so the window is in quarters.
    recorder.capture(performance_module.get_sharpe_ratio(period="quarterly", rolling=4))
    recorder.capture(performance_module.get_sharpe_ratio(growth=True))
    recorder.capture(performance_module.get_sharpe_ratio(growth=True, lag=[1, 2, 3]))


def test_get_sharpe_ratio_probabilistic(recorder, performance_module):
    probabilistic = partial(performance_module.get_sharpe_ratio, method="probabilistic")

    recorder.capture(round(probabilistic(), 4))
    recorder.capture(probabilistic(period="quarterly"))
    recorder.capture(probabilistic(period="monthly", rolling=6))
    recorder.capture(probabilistic(benchmark_sharpe_ratio=-0.5))
    recorder.capture(probabilistic(growth=True))
    recorder.capture(probabilistic(growth=True, lag=[1, 2, 3]))


def test_get_sharpe_ratio_deflated(recorder, performance_module):
    deflated = partial(performance_module.get_sharpe_ratio, method="deflated")

    recorder.capture(round(deflated(), 4))
    recorder.capture(deflated(period="quarterly"))
    recorder.capture(deflated(period="monthly", rolling=6))
    recorder.capture(deflated(n_trials=1))
    recorder.capture(deflated(n_trials=50))
    recorder.capture(deflated(growth=True))
    recorder.capture(deflated(growth=True, lag=[1, 2, 3]))

    # More trials should never raise the probability, all else held fixed.
    fewer_trials = deflated(n_trials=1)
    more_trials = deflated(n_trials=50)
    assert (more_trials.fillna(0) <= fewer_trials.fillna(0) + 1e-9).all().all()


def test_get_sharpe_ratio_lo(recorder, performance_module):
    lo = partial(performance_module.get_sharpe_ratio, method="lo")

    recorder.capture(lo())
    recorder.capture(lo(period="quarterly", lags=3))
    recorder.capture(lo(period="monthly", rolling=36))
    # Rolling windows below 30 observations have no ratio.
    assert lo(period="monthly", rolling=12).isna().all().all()

    # Without autocorrelation lags, Lo's factor is SQRT(q) and the ratio the annualized one.
    pd.testing.assert_frame_equal(
        lo(lags=0, rounding=8),
        performance_module.get_sharpe_ratio(annualize=True, rounding=8),
    )


def test_get_sharpe_ratio_standard_error(recorder, performance_module):
    recorder.capture(performance_module.get_sharpe_ratio_standard_error())
    recorder.capture(
        performance_module.get_sharpe_ratio_standard_error(method="non_normal")
    )
    recorder.capture(
        performance_module.get_sharpe_ratio_standard_error(
            period="quarterly", method="gmm"
        )
    )
    with pytest.raises(InvalidArgumentError, match="'non_normal'"):
        performance_module.get_sharpe_ratio_standard_error(method="other")


def test_get_minimum_track_record_length(recorder, performance_module):
    recorder.capture(performance_module.get_minimum_track_record_length())
    recorder.capture(
        performance_module.get_minimum_track_record_length(
            period="quarterly", confidence_level=0.9
        )
    )
    recorder.capture(
        performance_module.get_minimum_track_record_length(
            period="monthly", rolling=12, benchmark_sharpe_ratio=-0.1
        )
    )


def test_get_coskewness(recorder, performance_module):
    recorder.capture(performance_module.get_coskewness())
    recorder.capture(performance_module.get_coskewness(method="residual"))
    recorder.capture(performance_module.get_coskewness(period="quarterly"))


def test_get_cokurtosis(recorder, performance_module):
    recorder.capture(performance_module.get_cokurtosis())
    recorder.capture(performance_module.get_cokurtosis(method="residual"))
    recorder.capture(performance_module.get_cokurtosis(period="quarterly"))


def test_get_sortino_ratio(recorder, performance_module):
    recorder.capture(performance_module.get_sortino_ratio())
    recorder.capture(performance_module.get_sortino_ratio(period="quarterly"))
    recorder.capture(performance_module.get_sortino_ratio(growth=True))
    recorder.capture(performance_module.get_sortino_ratio(growth=True, lag=[1, 2, 3]))
    recorder.capture(performance_module.get_sortino_ratio(period="monthly", rolling=6))


def test_get_ulcer_performance_index(recorder, performance_module):
    recorder.capture(performance_module.get_ulcer_performance_index())
    recorder.capture(performance_module.get_ulcer_performance_index(period="quarterly"))
    recorder.capture(performance_module.get_ulcer_performance_index(growth=True))
    recorder.capture(
        performance_module.get_ulcer_performance_index(growth=True, lag=[1, 2, 3])
    )


def test_get_safety_first_ratio(recorder, performance_module):
    recorder.capture(performance_module.get_safety_first_ratio())
    recorder.capture(
        performance_module.get_safety_first_ratio(minimum_acceptable_return=0.02)
    )
    recorder.capture(performance_module.get_safety_first_ratio(period="quarterly"))
    recorder.capture(performance_module.get_safety_first_ratio(growth=True))
    recorder.capture(
        performance_module.get_safety_first_ratio(
            period="monthly", rolling=6, minimum_acceptable_return=0.02
        )
    )


def test_get_upside_potential_ratio(recorder, performance_module):
    recorder.capture(performance_module.get_upside_potential_ratio())
    recorder.capture(performance_module.get_upside_potential_ratio(period="quarterly"))
    recorder.capture(performance_module.get_upside_potential_ratio(growth=True))
    recorder.capture(
        performance_module.get_upside_potential_ratio(period="monthly", rolling=6)
    )


def test_get_pain_ratio(recorder, performance_module):
    recorder.capture(performance_module.get_pain_ratio())
    recorder.capture(performance_module.get_pain_ratio(period="quarterly"))
    recorder.capture(performance_module.get_pain_ratio(growth=True, lag=[1, 2]))


def test_get_m2_ratio(recorder, performance_module):
    recorder.capture(performance_module.get_m2_ratio())
    recorder.capture(performance_module.get_m2_ratio(period="quarterly"))
    recorder.capture(performance_module.get_m2_ratio(growth=True))
    recorder.capture(performance_module.get_m2_ratio(growth=True, lag=[1, 2, 3]))
    recorder.capture(performance_module.get_m2_ratio(period="monthly", rolling=6))


def test_get_tracking_error(recorder, performance_module):
    recorder.capture(performance_module.get_tracking_error())
    recorder.capture(performance_module.get_tracking_error(period="quarterly"))
    recorder.capture(performance_module.get_tracking_error(growth=True))
    recorder.capture(performance_module.get_tracking_error(growth=True, lag=[1, 2, 3]))
    recorder.capture(performance_module.get_tracking_error(period="monthly", rolling=6))


def test_get_information_ratio(recorder, performance_module):
    recorder.capture(performance_module.get_information_ratio())
    recorder.capture(performance_module.get_information_ratio(period="quarterly"))
    recorder.capture(performance_module.get_information_ratio(growth=True))
    recorder.capture(
        performance_module.get_information_ratio(growth=True, lag=[1, 2, 3])
    )
    recorder.capture(
        performance_module.get_information_ratio(period="monthly", rolling=6)
    )


def test_get_compound_growth_rate(recorder, performance_module):
    recorder.capture(performance_module.get_compound_growth_rate())
    recorder.capture(performance_module.get_compound_growth_rate(rounding=10))


def test_get_appraisal_ratio(recorder, performance_module):
    recorder.capture(performance_module.get_appraisal_ratio())
    recorder.capture(performance_module.get_appraisal_ratio(period="quarterly"))
    recorder.capture(performance_module.get_appraisal_ratio(growth=True))
    recorder.capture(performance_module.get_appraisal_ratio(growth=True, lag=[1, 2, 3]))
    recorder.capture(
        performance_module.get_appraisal_ratio(period="monthly", rolling=6)
    )


def test_get_fama_decomposition(recorder, performance_module):
    recorder.capture(performance_module.get_fama_decomposition())
    recorder.capture(performance_module.get_fama_decomposition(period="quarterly"))
    recorder.capture(performance_module.get_fama_decomposition(growth=True))
    recorder.capture(
        performance_module.get_fama_decomposition(period="monthly", rolling=6)
    )

    # Selectivity + Diversification must reconstruct Jensen's Alpha for every ticker.
    fama_decomposition = performance_module.get_fama_decomposition()
    jensens_alpha = performance_module.get_alpha(method="jensen")
    reconstructed = (
        fama_decomposition.xs("Selectivity", level=1, axis=1)
        + fama_decomposition.xs("Diversification", level=1, axis=1)
    ).round(2)
    assert (reconstructed == jensens_alpha.round(2)).all().all()


def test_get_sharpe_ratio_adjusted(recorder, performance_module):
    adjusted = partial(performance_module.get_sharpe_ratio, method="adjusted")

    recorder.capture(adjusted())
    recorder.capture(adjusted(period="quarterly"))
    recorder.capture(adjusted(period="monthly", rolling=6))
    recorder.capture(adjusted(growth=True))
    recorder.capture(adjusted(growth=True, lag=[1, 2, 3]))


def test_get_starr_ratio(recorder, performance_module):
    recorder.capture(performance_module.get_starr_ratio())
    recorder.capture(performance_module.get_starr_ratio(period="quarterly"))
    recorder.capture(performance_module.get_starr_ratio(within_period=False))
    recorder.capture(performance_module.get_starr_ratio(alpha=0.1))
    recorder.capture(performance_module.get_starr_ratio(growth=True))
    recorder.capture(performance_module.get_starr_ratio(growth=True, lag=[1, 2, 3]))


def test_get_rachev_ratio(recorder, performance_module):
    recorder.capture(performance_module.get_rachev_ratio())
    recorder.capture(performance_module.get_rachev_ratio(period="quarterly"))
    recorder.capture(performance_module.get_rachev_ratio(within_period=False))
    recorder.capture(performance_module.get_rachev_ratio(alpha=0.1))
    recorder.capture(performance_module.get_rachev_ratio(growth=True))
    recorder.capture(performance_module.get_rachev_ratio(growth=True, lag=[1, 2, 3]))


def test_get_market_timing_model(recorder, performance_module):
    recorder.capture(performance_module.get_market_timing_model())
    recorder.capture(performance_module.get_market_timing_model(period="quarterly"))
    recorder.capture(performance_module.get_market_timing_model(growth=True))
    recorder.capture(
        performance_module.get_market_timing_model(method="henriksson_merton")
    )
    recorder.capture(
        performance_module.get_market_timing_model(
            method="henriksson_merton", period="quarterly"
        )
    )
    recorder.capture(
        performance_module.get_market_timing_model(
            method="henriksson_merton", growth=True
        )
    )


def test_get_returns_cumulative_keeps_the_first_period(performance_module):
    # A date range starting after the first observation, so its first period has a
    # return of its own (the data's first day only has a zero return).
    performance = copy.copy(performance_module)
    performance._start_date = "2020-06-15"

    period_returns = performance.get_returns(period="yearly", rounding=10)
    cumulative = performance.get_returns(period="yearly", cumulative=True, rounding=10)

    # Growth of 1 from the start of the date range: the first period shows its own return.
    assert period_returns.index[0] == pd.Period("2020", "Y")
    pd.testing.assert_frame_equal(
        cumulative, (1 + period_returns).cumprod().round(10), check_freq=False
    )
    assert (cumulative.iloc[0] != 1).all()

    excess = performance.get_excess_return(period="yearly", rounding=10)
    cumulative_excess = performance.get_excess_return(
        period="yearly", cumulative=True, rounding=10
    )
    pd.testing.assert_frame_equal(
        cumulative_excess, (1 + excess).cumprod().round(10), check_freq=False
    )
    assert (cumulative_excess.iloc[0] != 1).all()


def test_get_sharpe_ratio_deflated_uses_daily_trials(performance_module):
    probabilistic = performance_module.get_sharpe_ratio(
        method="probabilistic", rounding=10
    )
    deflated = performance_module.get_sharpe_ratio(method="deflated", rounding=10)

    # Trials on the same (daily) scale as the Sharpe ratio give a probability strictly
    # between 0 and the probabilistic Sharpe ratio, not a deflated ratio of always 0.
    assert (deflated.to_numpy() > 0).any()
    assert (deflated.to_numpy() <= probabilistic.to_numpy() + 1e-12).all()


def _intraday_performance():
    rng = np.random.default_rng(1)
    # Ten trading days, without the weekends a ticker trading on trading days does not have.
    days = pd.bdate_range("2024-01-01", periods=10).to_period("D")
    columns = pd.MultiIndex.from_product([["Return"], ["A", "Benchmark"]])
    daily = pd.DataFrame(
        rng.normal(0, 0.01, (len(days), 2)), index=days, columns=columns
    )
    # Seven hourly bars per day.
    stamps = [
        pd.Timestamp(str(day)) + pd.Timedelta(hours=10 + hour)
        for day in days
        for hour in range(7)
    ]
    intraday = pd.DataFrame(
        rng.normal(0.0005, 0.002, (len(stamps), 2)),
        index=pd.PeriodIndex(stamps, freq="min"),
        columns=columns,
    )
    return Performance(
        tickers=["A"],
        historical_data={
            "intraday": intraday,
            "daily": daily,
            "weekly": daily,
            "monthly": daily,
            "quarterly": daily,
            "yearly": daily,
        },
        risk_free_rate_data={"daily": pd.Series(0.04, index=days)},
        intraday_period="1hour",
        rounding=None,
    )


def test_get_sharpe_ratio_annualizes_intraday_bars():
    performance = _intraday_performance()

    for period in ("intraday", "daily"):
        ratio = performance.get_sharpe_ratio(period=period, rounding=10)
        annualized = performance.get_sharpe_ratio(
            period=period, annualize=True, rounding=10
        )

        # 252 trading days of 7 bars each.
        assert not ratio.empty
        assert (annualized / ratio).to_numpy() == pytest.approx(
            np.sqrt(252 * 7), rel=1e-6
        )


def test_intraday_excess_return_uses_the_per_bar_risk_free_rate():
    performance = _intraday_performance()

    within_day = performance._get_within_historical_data("daily")
    daily_rate = (1 + 0.04) ** (1 / 252) - 1
    bar_rate = (1 + daily_rate) ** (1 / 7) - 1

    # Ten days of seven bars each, grouped per day.
    assert within_day.index.get_level_values(0).nunique() == 10
    np.testing.assert_allclose(
        within_day["Excess Return"]["A"].to_numpy(),
        within_day["Return"]["A"].to_numpy() - bar_rate,
    )

    sharpe_ratio = performance.get_sharpe_ratio(period="daily", rounding=10)
    excess = within_day["Excess Return"]["A"]
    expected = excess.groupby(level=0).mean() / excess.groupby(level=0).std()
    np.testing.assert_allclose(
        sharpe_ratio["A"].to_numpy(), expected.round(10).to_numpy()
    )
    assert not performance.get_sortino_ratio(period="daily").empty


def test_within_period_daily_without_intraday_data_raises(performance_module):
    with pytest.raises(ValueError, match="Intraday data is required"):
        performance_module._get_within_historical_data("daily")


def test_ratios_over_the_whole_period_hold_one_number_per_ticker(performance_module):
    # Over the whole period the ratio is indexed by ticker; slicing it by date left nothing.
    for ratio in (
        performance_module.get_omega_ratio(within_period=False),
        performance_module.get_gain_to_pain_ratio(within_period=False),
        performance_module.get_rachev_ratio(within_period=False),
    ):
        assert not ratio.empty
        assert ratio.notna().all()


def test_get_beta_implied_return(performance_module):
    daily = performance_module.get_beta_implied_return(beta_window=60, rounding=10)
    yearly = performance_module.get_beta_implied_return(
        period="yearly", beta_window=60, rounding=10
    )

    for result in (daily, yearly):
        assert list(result.columns.get_level_values(0).unique()) == [
            "Beta",
            "Benchmark Return",
            "Implied Return",
            "Actual Return",
            "Residual",
            "Residual Value",
            "Residual Percentile",
            "Residual Largest Since",
        ]
        explained = result.dropna(subset=[("Beta", result["Beta"].columns[0])])
        assert not explained.empty
        assert (
            explained["Residual"]
            - (explained["Actual Return"] - explained["Implied Return"])
        ).abs().max().max() == pytest.approx(0.0, abs=1e-8)

    # The actual returns are the Toolkit's own period returns.
    returns = performance_module.get_returns(period="yearly", rounding=10)
    common = yearly.index.intersection(returns.index)
    assert yearly.loc[common, "Actual Return"].to_numpy() == pytest.approx(
        returns.loc[common, yearly["Actual Return"].columns].to_numpy(), nan_ok=True
    )


def test_the_kelly_criterion_per_period_over_the_history_and_rolling(
    performance_module,
):
    per_period = performance_module.get_kelly_criterion(rounding=8)
    half_kelly = performance_module.get_kelly_criterion(fraction=0.5, rounding=8)

    assert not per_period.empty
    pd.testing.assert_frame_equal(half_kelly, per_period / 2, atol=1e-8)
    assert not performance_module.get_kelly_criterion(within_period=False).empty
    assert (
        not performance_module.get_kelly_criterion(
            period="daily", rolling=20, method="continuous"
        )
        .dropna(how="all")
        .empty
    )


def test_the_return_statistics_over_the_history_and_per_period(performance_module):
    statistics = performance_module.get_return_statistics()

    assert "Profit Factor" in statistics.index
    assert statistics.loc["Win Rate"].between(0, 1).all()

    per_period = performance_module.get_return_statistics(within_period=True)

    assert "Payoff Ratio" in per_period.columns.get_level_values(0)


@pytest.mark.parametrize(
    "method", ["blume", "vasicek", "dimson", "scholes_williams", "upside", "downside"]
)
def test_get_beta_with_another_estimator(performance_module, method):
    ordinary = performance_module.get_beta()
    estimated = performance_module.get_beta(method=method)

    assert estimated.shape == ordinary.shape
    assert estimated.notna().all().all()

    if method == "blume":
        pd.testing.assert_frame_equal(
            estimated, (0.67 * ordinary + 0.33).round(4), check_exact=False, atol=1e-4
        )

    rolling = performance_module.get_beta(method=method, period="monthly", rolling=6)
    assert not rolling.dropna(how="all").empty


def test_morningstar_measures_use_the_monthly_geometric_excess_returns(
    performance_module,
):
    monthly_returns = performance_module._get_column(
        "monthly", "Return", within_period=False
    )
    excess_returns = (1 + monthly_returns).div(
        1 + performance_module._risk_free_rate_data["monthly"][monthly_returns.columns]
    ) - 1
    ticker = excess_returns.columns[0]

    mrar = performance_module.get_morningstar_risk_adjusted_return(
        within_period=False, rounding=10
    )
    risk = performance_module.get_morningstar_risk(within_period=False, rounding=10)

    values = excess_returns[ticker].dropna()
    expected = np.mean((1 + values) ** -2) ** -6 - 1
    assert mrar[ticker] == pytest.approx(expected, abs=1e-9)
    assert risk[ticker] == pytest.approx(
        np.prod(1 + values) ** (12 / len(values)) - 1 - expected, abs=1e-9
    )

    yearly = performance_module.get_morningstar_risk_adjusted_return()
    assert not yearly.empty
    assert performance_module.get_morningstar_risk(rolling=12).notna().any().all()


def test_morningstar_measures_need_a_yearly_or_quarterly_period(performance_module):
    with pytest.raises(ValueError, match="yearly or quarterly"):
        performance_module._get_morningstar_risk_adjusted_return(
            2, "monthly", True, None
        )


def _toolkit_with_historical(historical):
    from financetoolkit import Toolkit

    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        historical=historical,
        balance=pd.read_pickle("tests/datasets/balance_dataset.pickle"),
        income=pd.read_pickle("tests/datasets/income_dataset.pickle"),
        cash=pd.read_pickle("tests/datasets/cash_dataset.pickle"),
        convert_currency=False,
        start_date="2019-12-31",
        end_date="2023-01-01",
        sleep_timer=False,
    )
    toolkit._daily_risk_free_rate = pd.read_pickle(
        "tests/datasets/risk_free_rate.pickle"
    )
    toolkit._daily_treasury_data = pd.read_pickle("tests/datasets/treasury_data.pickle")

    return toolkit


DRAWDOWN_AND_TAIL_RATIOS = [
    "get_ulcer_performance_index",
    "get_calmar_ratio",
    "get_sterling_ratio",
    "get_burke_ratio",
    "get_starr_ratio",
    "get_rachev_ratio",
]


def test_a_later_listed_ticker_keeps_the_periods_of_the_others(performance_module):
    historical = pd.read_pickle("tests/datasets/historical_dataset.pickle")
    before_listing = historical.index < pd.Period("2021-06-01", "D")
    msft_columns = [column for column in historical.columns if column[1] == "MSFT"]
    historical.loc[before_listing, msft_columns] = np.nan
    later_listed = _toolkit_with_historical(historical).performance

    for method in DRAWDOWN_AND_TAIL_RATIOS:
        result = getattr(later_listed, method)()
        reference = getattr(performance_module, method)()

        # AAPL's data is unchanged, so neither are its ratios, also before MSFT lists.
        pd.testing.assert_series_equal(result["AAPL"], reference["AAPL"], obj=method)
        assert result.loc[:"2020", "MSFT"].isna().all(), method


def test_a_constant_price_gives_nan_instead_of_dropping_other_tickers(
    performance_module,
):
    historical = pd.read_pickle("tests/datasets/historical_dataset.pickle")
    for column in ["Open", "High", "Low", "Close", "Adj Close"]:
        historical[(column, "MSFT")] = 100.0
    historical[("Return", "MSFT")] = 0.0
    historical[("Cumulative Return", "MSFT")] = 1.0
    constant = _toolkit_with_historical(historical).performance

    calmar = constant.get_calmar_ratio()
    pd.testing.assert_series_equal(
        calmar["AAPL"], performance_module.get_calmar_ratio()["AAPL"]
    )
    # No drawdown and no return: 0 / 0 is undefined.
    assert calmar["MSFT"].isna().all()
    # A beta of 0 leaves the Treynor ratio undefined, not minus infinity.
    assert constant.get_treynor_ratio()["MSFT"].isna().all()


@pytest.mark.parametrize(
    "method",
    [
        "get_calmar_ratio",
        "get_burke_ratio",
        "get_ulcer_performance_index",
        "get_sortino_ratio",
        "get_omega_ratio",
        "get_gain_to_pain_ratio",
        "get_upside_potential_ratio",
        "get_pain_ratio",
    ],
)
def test_weeks_without_a_drawdown_or_a_loss_are_nan_not_infinite(
    performance_module, method
):
    result = getattr(performance_module, method)(period="weekly")

    assert not np.isinf(result.to_numpy(dtype=float)).any()
    assert result.notna().any().all()


def _daily_returns_in(performance_module, ticker: str, year: str) -> pd.Series:
    """The daily returns of one ticker within one calendar year."""
    daily = performance_module._historical_data["daily"]["Return"][ticker]
    return daily[daily.index.year == int(year)]


def _year_return(performance_module, ticker: str, year: str) -> float:
    return performance_module._historical_data["yearly"]["Return"].loc[year, ticker]


def _drawdowns(returns: pd.Series) -> pd.Series:
    wealth = (1 + returns.fillna(0)).cumprod()
    return wealth / wealth.cummax().clip(lower=1) - 1


def test_get_calmar_ratio_divides_the_year_by_its_maximum_drawdown(
    performance_module,
):
    result = performance_module.get_calmar_ratio(rounding=None)
    drawdowns = _drawdowns(_daily_returns_in(performance_module, "AAPL", "2021"))

    assert result.loc["2021", "AAPL"] == pytest.approx(
        _year_return(performance_module, "AAPL", "2021") / abs(drawdowns.min()),
        abs=5e-5,
    )


def test_get_sterling_ratio_divides_the_year_by_its_average_largest_drawdown(
    performance_module,
):
    daily = _daily_returns_in(performance_module, "AAPL", "2020")
    year_return = _year_return(performance_module, "AAPL", "2020")

    for largest, adjustment in ((3, 0.1), (3, 0.0), (5, 0.1), (1, 0.1)):
        result = performance_module.get_sterling_ratio(
            adjustment=adjustment, largest_drawdowns=largest, rounding=None
        )
        # Bacon's D_Lar: the mean depth of the deepest drawdowns from peak to trough.
        table = risk_model.get_drawdown_table(daily.to_frame("AAPL"), top=largest)
        assert result.loc["2020", "AAPL"] == pytest.approx(
            year_return / (abs(table["Drawdown"].mean()) + adjustment), abs=5e-5
        )

    # One drawdown is the deepest: the maximum drawdown of the year.
    assert performance_module.get_sterling_ratio(
        largest_drawdowns=1, adjustment=0.0, rounding=None
    ).loc["2020", "AAPL"] == pytest.approx(
        performance_module.get_calmar_ratio(rounding=None).loc["2020", "AAPL"],
        abs=5e-5,
    )


def test_get_burke_ratio_has_the_sign_of_the_excess_return(performance_module):
    result = performance_module.get_burke_ratio(rounding=None)
    excess_return = performance_module.get_excess_return(period="yearly")

    assert not result.empty
    common = result.index.intersection(excess_return.index)
    assert (
        (
            np.sign(result.loc[common, ["AAPL", "MSFT"]])
            == np.sign(excess_return.loc[common, ["AAPL", "MSFT"]])
        )
        .all()
        .all()
    )


def test_get_kappa_ratio_scales_the_mean_excess_return_by_the_partial_moment(
    performance_module,
):
    within = performance_module._get_column(
        "yearly", "Excess Return", within_period=True
    )
    excess = within.loc["2021", "AAPL"]

    for order in (2, 3):
        result = performance_module.get_kappa_ratio(order=order, rounding=None)
        lower_partial_moment = (excess.clip(upper=0).abs() ** order).mean()

        assert result.loc["2021", "AAPL"] == pytest.approx(
            excess.mean() / lower_partial_moment ** (1 / order), abs=5e-5
        )


def test_get_covariance_matrix_is_symmetric_with_the_variances_on_the_diagonal(
    performance_module,
):
    result = performance_module.get_covariance_matrix(period="monthly", rounding=10)
    monthly = performance_module._historical_data["monthly"]["Return"].loc[
        performance_module._start_date : performance_module._end_date
    ]

    pd.testing.assert_frame_equal(result, result.T)
    for ticker in result.columns:
        assert result.loc[ticker, ticker] == pytest.approx(
            monthly[ticker].var(), abs=1e-9
        )


def test_get_beta_implied_return_values_the_residual_at_the_previous_close(
    performance_module, ratios_module
):
    daily = performance_module.get_beta_implied_return(beta_window=60, rounding=10)
    market_cap = ratios_module.get_market_cap(show_daily=True, rounding=10)

    # The Residual times the market cap of the trading day before, never the same day's.
    previous = market_cap.shift(1).reindex(daily.index)
    expected = daily["Residual"] * previous[daily["Residual"].columns]
    valued = expected.notna()
    assert valued.to_numpy().any()
    # The Residual is rounded to 10 decimals here, which on a market cap of about 2e12
    # leaves differences of a few hundred.
    assert daily["Residual Value"][valued].to_numpy() == pytest.approx(
        expected[valued].to_numpy(), rel=1e-8, abs=1e3, nan_ok=True
    )

    # The significance of the Residual is get_move_significance on the residuals alone.
    from financetoolkit.risk import risk_model

    for ticker in daily["Residual"].columns:
        significance = risk_model.get_move_significance(
            daily[("Residual", ticker)], window=252
        )
        assert daily[("Residual Percentile", ticker)].to_numpy() == pytest.approx(
            significance["Percentile"].to_numpy(), abs=1e-9, nan_ok=True
        )
        pd.testing.assert_series_equal(
            daily[("Residual Largest Since", ticker)],
            significance["Largest Since"],
            check_names=False,
        )


def test_get_beta_implied_return_without_a_market_cap(performance_module):
    performance = copy.copy(performance_module)
    performance._market_cap_function = None
    performance._data_cache = {}

    result = performance.get_beta_implied_return(beta_window=60)

    # Without shares (an ETF, an index) the value is empty, never 0.
    assert result["Residual Value"].isna().all().all()
    assert result["Residual"].notna().any().any()


def _weekend_performance(rate=0.04, tickers=("BTC", "SPY")):
    """A stock (trading days) and a cryptocurrency (every calendar day) at a flat rate."""
    rng = np.random.default_rng(2)
    days = pd.period_range("2021-01-01", "2022-06-30", freq="D")
    weekday = days.dayofweek < 5  # noqa: PLR2004
    returns = pd.DataFrame(
        {
            "BTC": rng.normal(0.001, 0.03, len(days)),
            "SPY": np.where(weekday, rng.normal(0.0005, 0.01, len(days)), np.nan),
            "Benchmark": np.where(weekday, rng.normal(0.0005, 0.01, len(days)), np.nan),
        },
        index=days,
    )
    returns.iloc[0] = np.nan
    # Without the cryptocurrency the data only holds the trading days.
    returns = returns[[*tickers, "Benchmark"]]
    returns = returns[returns.notna().any(axis=1) | (returns.index == days[0])]
    prices = (1 + returns.fillna(0)).cumprod() * 100
    daily = pd.concat({"Return": returns, "Adj Close": prices, "Close": prices}, axis=1)
    historical_data = {"daily": daily, "intraday": pd.DataFrame()}
    # The return of a period runs from the last price of the period before to its last price.
    for period, frequency in (
        ("weekly", "W"),
        ("monthly", "M"),
        ("quarterly", "Q"),
        ("yearly", "Y"),
    ):
        period_prices = prices.groupby(returns.index.asfreq(frequency)).last()
        historical_data[period] = pd.concat(
            {
                "Return": period_prices / period_prices.shift() - 1,
                "Adj Close": period_prices,
                "Close": period_prices,
            },
            axis=1,
        )
    return Performance(
        tickers=list(tickers),
        historical_data=historical_data,
        risk_free_rate_data={"daily": pd.Series(rate, index=days)},
        rounding=None,
    )


def test_a_partial_period_earns_the_risk_free_rate_of_its_days(performance_module):
    annualized_rate = performance_module._annualized_risk_free_rate_data["daily"]
    returns = performance_module._historical_data["daily"]["Return"]
    yearly_rate = performance_module._risk_free_rate_data["yearly"]

    # Every year, including a partial one, compounds the annualized rate quoted on each
    # date a ticker has a return over the calendar days since its date before, as the
    # yearly return compounds the daily returns.
    for ticker in returns.columns:
        dates = returns[ticker].dropna().index
        first = returns.index.get_loc(dates[0])
        # The first return accrues from the date of the first price, here the first date.
        previous = returns.index[first - 1].ordinal if first else dates[0].ordinal - 1
        days = np.diff(np.append(previous, dates.asi8))
        rate = annualized_rate.sort_index().reindex(dates, method="ffill").bfill()
        growth = pd.Series((1 + rate.to_numpy()) ** (days / 365), index=dates)
        for year in yearly_rate.index:
            expected = growth[dates.asfreq("Y") == year].prod() - 1
            assert yearly_rate.loc[year, ticker] == pytest.approx(expected, rel=1e-12)

    performance = _weekend_performance()
    # The 181 days of the first half of 2022 at a flat 4% earn about half of it, not the
    # full year's rate, for the stock and the cryptocurrency alike.
    rate_2022 = performance._risk_free_rate_data["yearly"].loc["2022"]
    assert rate_2022.to_numpy() == pytest.approx(1.04 ** (181 / 365) - 1, rel=1e-9)

    excess = performance.get_excess_return(period="yearly")
    returns_2022 = performance._historical_data["yearly"]["Return"].loc["2022"]
    assert excess.loc["2022", "SPY"] == pytest.approx(
        returns_2022["SPY"] - rate_2022["SPY"], rel=1e-9
    )


def test_a_ticker_trading_on_weekends_accrues_the_rate_per_calendar_day():
    performance = _weekend_performance()
    daily = performance._get_within_historical_data("yearly")
    risk_free = (daily["Return"] - daily["Excess Return"]).droplevel(0)

    # The cryptocurrency earns the calendar days since its previous return, a year of
    # them the annual rate; the stock a trading day with every return, whichever other
    # tickers trade on weekends.
    btc = risk_free["BTC"].dropna()
    assert btc.to_numpy() == pytest.approx(1.04 ** (1 / 365) - 1, rel=1e-9)
    assert (1 + btc.loc["2021-01-02":"2022-01-01"]).prod() == pytest.approx(
        1.04, rel=1e-12
    )
    spy = risk_free["SPY"].dropna()
    assert spy.to_numpy() == pytest.approx(1.04 ** (1 / 252) - 1, rel=1e-9)

    # Each ticker is annualized with its own observations in a year.
    ratio = performance.get_sharpe_ratio()
    annualized = performance.get_sharpe_ratio(annualize=True)
    assert (annualized["BTC"] / ratio["BTC"]).to_numpy() == pytest.approx(
        np.sqrt(365), rel=1e-9
    )
    assert (annualized["SPY"] / ratio["SPY"]).to_numpy() == pytest.approx(
        np.sqrt(252), rel=1e-9
    )


def test_stock_returns_earn_the_trading_day_rate(performance_module):
    daily = performance_module._get_within_historical_data("yearly")
    risk_free = (daily["Return"] - daily["Excess Return"]).droplevel(0)
    annual_rate = performance_module._annualized_risk_free_rate_data["daily"]
    expected = (1 + annual_rate.reindex(risk_free.index).ffill()) ** (1 / 252) - 1

    for ticker in risk_free.columns:
        observed = risk_free[ticker].notna()
        assert risk_free.loc[observed, ticker].to_numpy() == pytest.approx(
            expected[observed].to_numpy(), rel=1e-9
        )


def test_the_rolling_appraisal_ratio_only_uses_earlier_residuals(performance_module):
    full = performance_module.get_appraisal_ratio(
        period="monthly", rolling=12, rounding=10
    )

    # The same calculation on the data up to 2021 gives the same 2021 ratios.
    performance = copy.copy(performance_module)
    performance._historical_data = {
        period: data.loc[:"2021-12-31"] if period in ("daily", "monthly") else data
        for period, data in performance_module._historical_data.items()
    }
    performance._data_cache = {}
    performance.__dict__.pop("_daily_returns", None)
    performance._risk_free_rate_data = type(performance_module._risk_free_rate_data)(
        performance._get_risk_free_rate
    )
    performance._end_date = "2021-12-31"
    truncated = performance.get_appraisal_ratio(
        period="monthly", rolling=12, rounding=10
    )

    pd.testing.assert_frame_equal(
        full.loc["2021-01":"2021-12"], truncated.loc["2021-01":"2021-12"]
    )


def test_the_rolling_fama_decomposition_keeps_total_risk_above_beta(
    performance_module,
):
    rolling = performance_module.get_fama_decomposition(
        period="monthly", rolling=12, rounding=10
    )
    returns = performance_module._historical_data["monthly"]["Return"]
    risk_free = performance_module._risk_free_rate_data["monthly"]["Benchmark"]
    market_premium = (returns["Benchmark"] - risk_free).reindex(rolling.index)

    # Diversification = (Total Risk Beta - Beta) * (Benchmark Return - Risk Free Rate)
    # with a Total Risk Beta of at least |Beta|, so it has the sign of the market premium
    # whenever Beta is positive.
    for ticker in ("AAPL", "MSFT"):
        diversification = rolling[(ticker, "Diversification")]
        valid = diversification.notna() & (market_premium.abs() > 0)
        assert (
            np.sign(diversification[valid]) * np.sign(market_premium[valid]) >= 0
        ).all()


def test_the_ulcer_performance_index_measures_drawdowns_from_the_running_peak(
    performance_module,
):
    default = performance_module.get_ulcer_performance_index(rounding=10)
    running_peak = performance_module.get_ulcer_performance_index(
        rolling=None, rounding=10
    )
    trailing = performance_module.get_ulcer_performance_index(rolling=14, rounding=10)

    pd.testing.assert_frame_equal(default, running_peak)
    assert not np.allclose(default.to_numpy(), trailing.to_numpy())


def test_the_deflated_sharpe_ratio_counts_non_overlapping_trials(performance_module):
    excess = performance_module._get_column("yearly", "Excess Return", True)
    window = int(excess.groupby(level=0).size().median())
    trials = excess.droplevel(0).count() // window

    default = performance_module.get_sharpe_ratio(method="deflated", rounding=10)
    for ticker, n_trials in trials.items():
        explicit = performance_module.get_sharpe_ratio(
            method="deflated", n_trials=int(n_trials), rounding=10
        )
        pd.testing.assert_series_equal(default[ticker], explicit[ticker])


def test_the_deflated_sharpe_ratio_counts_the_returns_of_each_ticker():
    performance = _weekend_performance()
    excess = performance._get_column("yearly", "Excess Return", True)
    default = performance.get_sharpe_ratio(method="deflated", rounding=10)

    # The window of each ticker is the median number of its own returns in a year, not of
    # the rows, which also hold the weekends only the cryptocurrency trades on.
    for ticker in ("BTC", "SPY"):
        window = int(excess[ticker].groupby(level=0).count().median())
        explicit = performance.get_sharpe_ratio(
            method="deflated", trials_window=window, rounding=10
        )
        assert default[ticker].notna().all()
        pd.testing.assert_series_equal(default[ticker], explicit[ticker])


def test_the_deflated_sharpe_ratio_of_a_stock_ignores_a_cryptocurrency_next_to_it():
    performance = _weekend_performance()
    default = performance.get_sharpe_ratio(method="deflated", rounding=None)

    # The weekends only the cryptocurrency trades on leave the stock without a return on
    # those rows, which neither empties its trials nor changes its ratio.
    alone = _weekend_performance(tickers=("SPY",)).get_sharpe_ratio(
        method="deflated", rounding=None
    )
    assert default.notna().all().all()
    pd.testing.assert_series_equal(default["SPY"], alone["SPY"])

    # Bailey and Lopez de Prado (2014) on the own returns of each ticker: the trials are
    # the Sharpe ratios of every window of the median number of returns in a year, and N
    # the non-overlapping windows those returns hold.
    excess = performance._get_column("yearly", "Excess Return", True)
    euler_mascheroni = 0.5772156649015329
    for ticker in ("BTC", "SPY"):
        history = excess[ticker].droplevel(0).dropna().to_numpy()
        window = int(excess[ticker].groupby(level=0).count().median())
        trials = np.array(
            [
                history[end - window : end].mean()
                / history[end - window : end].std(ddof=1)
                for end in range(window, len(history) + 1)
            ]
        )
        n_trials = len(history) // window
        expected_maximum = np.sqrt(trials.var(ddof=1)) * (
            (1 - euler_mascheroni) * stats.norm.ppf(1 - 1 / n_trials)
            + euler_mascheroni * stats.norm.ppf(1 - 1 / (n_trials * np.e))
        )
        for year in default.index:
            returns = excess.loc[year, ticker].dropna().to_numpy()
            sharpe_ratio = returns.mean() / returns.std(ddof=1)
            skewness = stats.skew(returns, bias=False)
            kurtosis = stats.kurtosis(returns, fisher=False, bias=False)
            expected = stats.norm.cdf(
                (sharpe_ratio - expected_maximum)
                * np.sqrt(len(returns) - 1)
                / np.sqrt(
                    1 - skewness * sharpe_ratio + (kurtosis - 1) / 4 * sharpe_ratio**2
                )
            )
            assert default.loc[year, ticker] == pytest.approx(expected, abs=1e-10)


def test_a_stock_earns_the_risk_free_rate_of_its_own_dates_next_to_a_cryptocurrency():
    mixed = _weekend_performance()
    alone = _weekend_performance(tickers=("SPY",))

    # The dates the cryptocurrency adds (the weekends) change neither the rate a stock
    # earns per return nor the period a weekend at the turn of a month accrues in.
    for period in ("daily", "weekly", "monthly", "quarterly", "yearly"):
        for column in ("SPY", "Benchmark"):
            pd.testing.assert_series_equal(
                mixed._risk_free_rate_data[period][column].dropna(),
                alone._risk_free_rate_data[period][column].dropna(),
            )

    # A month at a flat 4% earns its calendar days from the last trading day of the month
    # before to its own last trading day: May runs from Friday 2021-04-30 to Monday 2021-05-31
    # (31 days) and June from that Monday to Wednesday 2021-06-30 (30 days).
    monthly = mixed._risk_free_rate_data["monthly"]
    assert monthly.loc["2021-05", "SPY"] == pytest.approx(1.04 ** (31 / 365) - 1)
    assert monthly.loc["2021-06", "SPY"] == pytest.approx(1.04 ** (30 / 365) - 1)
    assert monthly.loc["2021-05", "BTC"] == pytest.approx(1.04 ** (31 / 365) - 1)
    assert monthly.loc["2021-07", "SPY"] == pytest.approx(1.04 ** (30 / 365) - 1)
    assert monthly.loc["2021-07", "BTC"] == pytest.approx(1.04 ** (31 / 365) - 1)


@pytest.mark.parametrize(("period", "rolling"), [("monthly", 12), ("daily", 63)])
def test_the_rolling_measures_of_a_stock_ignore_a_cryptocurrency_next_to_it(
    period, rolling
):
    mixed = _weekend_performance()
    alone = _weekend_performance(tickers=("SPY",))

    def spy(result):
        return result["SPY"].dropna()

    for method in ("standard", "lo", "adjusted", "probabilistic", "deflated"):
        pd.testing.assert_series_equal(
            spy(mixed.get_sharpe_ratio(period=period, rolling=rolling, method=method)),
            spy(alone.get_sharpe_ratio(period=period, rolling=rolling, method=method)),
        )
    pd.testing.assert_series_equal(
        spy(mixed.get_alpha(period=period, rolling=rolling, method="jensen")),
        spy(alone.get_alpha(period=period, rolling=rolling, method="jensen")),
        rtol=1e-9,
    )
    for name in (
        "get_sortino_ratio",
        "get_treynor_ratio",
        "get_capital_asset_pricing_model",
        "get_m2_ratio",
        "get_appraisal_ratio",
        "get_minimum_track_record_length",
        "get_safety_first_ratio",
        "get_upside_potential_ratio",
    ):
        pd.testing.assert_series_equal(
            spy(getattr(mixed, name)(period=period, rolling=rolling)),
            spy(getattr(alone, name)(period=period, rolling=rolling)),
            rtol=1e-9,
        )
    pd.testing.assert_series_equal(
        spy(mixed.get_omega_ratio(period=period, rolling=rolling)),
        spy(alone.get_omega_ratio(period=period, rolling=rolling)),
    )
    fama_decomposition = mixed.get_fama_decomposition(period=period, rolling=rolling)
    assert fama_decomposition["SPY"].notna().any().all()
    pd.testing.assert_frame_equal(
        fama_decomposition["SPY"].dropna(),
        alone.get_fama_decomposition(period=period, rolling=rolling)["SPY"].dropna(),
        rtol=1e-9,
    )
    pd.testing.assert_series_equal(
        spy(
            mixed.get_kelly_criterion(
                period=period, rolling=rolling, method="continuous"
            )
        ),
        spy(
            alone.get_kelly_criterion(
                period=period, rolling=rolling, method="continuous"
            )
        ),
    )


@pytest.mark.parametrize("period", ["weekly", "monthly", "quarterly", "yearly"])
def test_the_excess_return_of_a_stock_ignores_a_cryptocurrency_next_to_it(period):
    mixed = _weekend_performance()
    alone = _weekend_performance(tickers=("SPY",))

    for cumulative in (False, True):
        expected = alone.get_excess_return(period=period, cumulative=cumulative)
        result = mixed.get_excess_return(period=period, cumulative=cumulative)
        pd.testing.assert_series_equal(
            result["SPY"].dropna(), expected["SPY"].dropna(), rtol=1e-12
        )
    for name in (
        "get_sortino_ratio",
        "get_burke_ratio",
        "get_starr_ratio",
        "get_safety_first_ratio",
        "get_upside_potential_ratio",
        "get_pain_ratio",
    ):
        pd.testing.assert_series_equal(
            getattr(mixed, name)(period=period)["SPY"].dropna(),
            getattr(alone, name)(period=period)["SPY"].dropna(),
            rtol=1e-12,
        )


def test_the_rolling_appraisal_ratio_starts_after_one_window(performance_module):
    ratio = performance_module.get_appraisal_ratio(
        period="monthly", rolling=12, rounding=10
    )
    returns = performance_module._historical_data["monthly"]["Return"]
    risk_free = performance_module._risk_free_rate_data["monthly"]
    first_return = returns["AAPL"].first_valid_index()

    # The residuals of each window come from the regression over that window, so the
    # first ratio follows the first 12 returns rather than two windows of them.
    assert ratio["AAPL"].first_valid_index() == first_return + 11

    end = ratio.index[-1]
    window = returns.loc[end - 11 : end]
    y = window["AAPL"] - risk_free["AAPL"].reindex(window.index)
    x = window["Benchmark"] - risk_free["Benchmark"].reindex(window.index)
    slope, intercept = np.polyfit(x, y, 1)
    residual_standard_deviation = np.sqrt(((y - slope * x - intercept) ** 2).sum() / 11)
    alpha = performance_module.get_alpha(
        method="jensen", period="monthly", rolling=12, rounding=10
    )
    assert ratio.loc[end, "AAPL"] == pytest.approx(
        alpha.loc[end, "AAPL"] / residual_standard_deviation, abs=1e-8
    )


def _later_listing_performance(tickers):
    """Two stocks at a flat rate, "IPO" listed only from 2022 and "SPY" from 2021."""
    rng = np.random.default_rng(3)
    days = pd.period_range("2021-01-01", "2022-12-30", freq="B").asfreq("D")
    returns = pd.DataFrame(
        {
            "IPO": np.where(
                days >= pd.Period("2022-01-03", "D"),
                rng.normal(0.0005, 0.02, len(days)),
                np.nan,
            ),
            "SPY": rng.normal(0.0005, 0.01, len(days)),
            "Benchmark": rng.normal(0.0005, 0.01, len(days)),
        },
        index=days,
    )
    returns.iloc[0] = np.nan
    returns.loc[pd.Period("2022-01-03", "D"), "IPO"] = np.nan
    returns = returns[[*tickers, "Benchmark"]]
    prices = (1 + returns.fillna(0)).cumprod() * 100
    prices["IPO"] = prices["IPO"].where(days >= pd.Period("2022-01-03", "D"))
    daily = pd.concat({"Return": returns, "Adj Close": prices, "Close": prices}, axis=1)
    historical_data = {"daily": daily, "intraday": pd.DataFrame()}
    period_prices = prices.groupby(days.asfreq("M")).last()
    historical_data["monthly"] = pd.concat(
        {
            "Return": period_prices / period_prices.shift() - 1,
            "Adj Close": period_prices,
            "Close": period_prices,
        },
        axis=1,
    )
    return Performance(
        tickers=list(tickers),
        historical_data=historical_data,
        risk_free_rate_data={"daily": pd.Series(0.04, index=days)},
        rounding=None,
    )


def test_the_rolling_measures_of_a_ticker_ignore_a_later_listing_before_it():
    later_first = _later_listing_performance(("IPO", "SPY"))
    earlier_first = _later_listing_performance(("SPY", "IPO"))

    # The risk free rate of the months before the listing of the first ticker stays in date order, so a rolling window over twelve rows covers twelve consecutive months.  # noqa: E501
    assert later_first._risk_free_rate_data["monthly"].index.is_monotonic_increasing
    m2_ratio = later_first.get_m2_ratio(period="monthly", rolling=12)
    assert m2_ratio.loc["2021-12":"2022-12", "SPY"].notna().all()
    pd.testing.assert_frame_equal(
        m2_ratio,
        earlier_first.get_m2_ratio(period="monthly", rolling=12)[m2_ratio.columns],
    )


def test_get_safety_first_ratio_compounds_the_threshold_per_return(performance_module):
    result = performance_module.get_safety_first_ratio(
        minimum_acceptable_return=0.02, rounding=10
    )
    returns = _daily_returns_in(performance_module, "AAPL", "2021").dropna()

    # Roy (1952) on the daily returns of the year, with the annual threshold compounded
    # down to a trading day.
    threshold = 1.02 ** (1 / 252) - 1
    assert result.loc["2021", "AAPL"] == pytest.approx(
        (returns.mean() - threshold) / returns.std(ddof=1), abs=1e-9
    )


def test_get_safety_first_ratio_compounds_the_threshold_per_calendar_day():
    performance = _weekend_performance()
    result = performance.get_safety_first_ratio(minimum_acceptable_return=0.05)
    returns = performance._get_column("yearly", "Return", True)["BTC"].loc["2022"]
    returns = returns.dropna()

    assert result.loc["2022", "BTC"] == pytest.approx(
        (returns.mean() - (1.05 ** (1 / 365) - 1)) / returns.std(ddof=1), rel=1e-9
    )


def test_get_upside_potential_ratio_divides_the_gains_by_the_downside_deviation(
    performance_module,
):
    result = performance_module.get_upside_potential_ratio(rounding=10)
    within = performance_module._get_column(
        "yearly", "Excess Return", within_period=True
    )
    excess = within.loc["2022", "MSFT"].dropna().to_numpy()

    # PerformanceAnalytics' UpsidePotentialRatio(method="full") with the risk-free rate
    # as the MAR: both partial moments over every observation.
    upside_potential = excess[excess > 0].sum() / len(excess)
    downside_deviation = np.sqrt((excess[excess < 0] ** 2).sum() / len(excess))
    assert result.loc["2022", "MSFT"] == pytest.approx(
        upside_potential / downside_deviation, abs=1e-9
    )


def test_get_pain_ratio_divides_the_excess_return_by_the_mean_drawdown(
    performance_module,
):
    result = performance_module.get_pain_ratio(rounding=10)
    drawdowns = _drawdowns(
        _daily_returns_in(performance_module, "AAPL", "2021").dropna()
    )
    excess_return = (
        _year_return(performance_module, "AAPL", "2021")
        - performance_module._risk_free_rate_data["yearly"].loc["2021", "AAPL"]
    )

    # Bacon (2013): the Pain Index is the mean absolute drawdown over every day.
    assert result.loc["2021", "AAPL"] == pytest.approx(
        excess_return / drawdowns.abs().mean(), abs=1e-9
    )


def test_get_sharpe_ratio_smart(recorder, performance_module):
    smart = partial(performance_module.get_sharpe_ratio, method="smart")

    recorder.capture(smart())
    recorder.capture(smart(annualize=True))
    recorder.capture(smart(period="monthly", rolling=12))

    # The penalty is at least 1, so the smart ratio is never larger in size.
    standard = performance_module.get_sharpe_ratio(rounding=10)
    assert (smart(rounding=10).abs() <= standard.abs() + 1e-9).all().all()


def test_probabilistic_and_deflated_sharpe_ratios_can_be_annualized(performance_module):
    # A probability does not depend on the time scale of the Sharpe ratios.
    for method in ("probabilistic", "deflated"):
        annualized = performance_module.get_sharpe_ratio(method=method, annualize=True)
        assert not annualized.empty
        pd.testing.assert_frame_equal(
            annualized, performance_module.get_sharpe_ratio(method=method)
        )

    # An annualized benchmark is compared with the annualized Sharpe ratio.
    observations_per_year = 252
    pd.testing.assert_frame_equal(
        performance_module.get_sharpe_ratio(
            method="probabilistic", annualize=True, benchmark_sharpe_ratio=1.0
        ),
        performance_module.get_sharpe_ratio(
            method="probabilistic",
            benchmark_sharpe_ratio=1.0 / np.sqrt(observations_per_year),
        ),
    )


def test_get_sortino_ratio_variants(recorder, performance_module):
    sortino_ratio = performance_module.get_sortino_ratio

    recorder.capture(sortino_ratio(method="smart"))
    recorder.capture(sortino_ratio(method="adjusted", annualize=True))
    recorder.capture(sortino_ratio(method="probabilistic"))
    recorder.capture(sortino_ratio(method="probabilistic_adjusted"))
    recorder.capture(sortino_ratio(method="smart", period="monthly", rolling=12))

    pd.testing.assert_frame_equal(
        sortino_ratio(method="adjusted", rounding=10),
        (sortino_ratio(rounding=12) / np.sqrt(2)).round(10),
    )
    with pytest.raises(InvalidArgumentError, match="'smart'"):
        sortino_ratio(method="other")


def test_get_calmar_and_sterling_ratio_annualized(recorder, performance_module):
    calmar_ratio = performance_module.get_calmar_ratio(method="annualized")
    recorder.capture(calmar_ratio)
    recorder.capture(
        performance_module.get_sterling_ratio(method="annualized", adjustment=0.05)
    )
    # With one drawdown, D_Lar is the maximum drawdown: PerformanceAnalytics' SterlingRatio,
    # which is the Calmar ratio with the adjustment added to the drawdown.
    pd.testing.assert_series_equal(
        performance_module.get_sterling_ratio(
            method="annualized", adjustment=0.0, largest_drawdowns=1, rounding=None
        ),
        performance_module.get_calmar_ratio(method="annualized", rounding=None),
        check_names=False,
    )

    # The default per-period ratios are unchanged by the method parameter.
    pd.testing.assert_frame_equal(
        performance_module.get_calmar_ratio(),
        performance_module.get_calmar_ratio(method="standard"),
    )

    returns = performance_module._historical_data["daily"]["Return"]["AAPL"]
    wealth = (1 + returns).cumprod()
    maximum_drawdown = (wealth / wealth.cummax() - 1).min()
    annualized_return = np.prod(1 + returns) ** (252 / len(returns)) - 1
    assert calmar_ratio["AAPL"] == pytest.approx(
        annualized_return / abs(maximum_drawdown), abs=1e-4
    )


def test_get_alpha_annualized(recorder, performance_module):
    recorder.capture(performance_module.get_alpha(method="annualized"))
    recorder.capture(
        performance_module.get_alpha(method="annualized", show_full_results=True)
    )


def test_get_burke_ratio_modified(recorder, performance_module):
    recorder.capture(performance_module.get_burke_ratio(modified=True))
    recorder.capture(
        performance_module.get_burke_ratio(modified=True, within_period=False)
    )


def test_get_treynor_ratio_modified(recorder, performance_module):
    recorder.capture(performance_module.get_treynor_ratio(modified=True))
    recorder.capture(
        performance_module.get_treynor_ratio(modified=True, period="monthly", rolling=6)
    )


def test_get_appraisal_ratio_methods(recorder, performance_module):
    recorder.capture(performance_module.get_appraisal_ratio(method="modified"))
    recorder.capture(performance_module.get_appraisal_ratio(method="alternative"))
    recorder.capture(
        performance_module.get_appraisal_ratio(
            method="alternative", period="monthly", rolling=6
        )
    )


def test_get_m2_ratio_sortino_and_excess(recorder, performance_module):
    recorder.capture(performance_module.get_m2_ratio(method="sortino"))
    recorder.capture(performance_module.get_m2_ratio(excess="geometric"))
    recorder.capture(
        performance_module.get_m2_ratio(method="sortino", excess="arithmetic")
    )
    # The M2 for Sortino has no rolling form.
    assert performance_module.get_m2_ratio(method="sortino", rolling=6).empty


def test_get_gain_to_pain_ratio_omega_sharpe(recorder, performance_module):
    omega_sharpe_ratio = performance_module.get_gain_to_pain_ratio(
        minimum_acceptable_return=0.001, rounding=10
    )
    recorder.capture(omega_sharpe_ratio)

    # The Omega-Sharpe ratio is the Omega ratio minus 1.
    omega_ratio = performance_module.get_omega_ratio(
        minimum_acceptable_return=0.001, rounding=10
    )
    np.testing.assert_allclose(
        omega_sharpe_ratio.to_numpy(), omega_ratio.to_numpy() - 1, atol=1e-9
    )


def test_get_d_ratio(recorder, performance_module):
    recorder.capture(performance_module.get_d_ratio())
    recorder.capture(performance_module.get_d_ratio(period="quarterly"))
    recorder.capture(performance_module.get_d_ratio(within_period=False))


def test_get_prospect_ratio(recorder, performance_module):
    recorder.capture(performance_module.get_prospect_ratio())
    recorder.capture(performance_module.get_prospect_ratio(period="quarterly"))
    recorder.capture(
        performance_module.get_prospect_ratio(
            minimum_acceptable_return=0.001, loss_aversion=2
        )
    )


def test_get_omega_excess_return(recorder, performance_module):
    recorder.capture(performance_module.get_omega_excess_return())
    recorder.capture(performance_module.get_omega_excess_return(period="quarterly"))
    recorder.capture(
        performance_module.get_omega_excess_return(minimum_acceptable_return=0.001)
    )


def test_get_return_statistics(recorder, performance_module):
    recorder.capture(performance_module.get_return_statistics())
    recorder.capture(performance_module.get_return_statistics(period="monthly"))
    recorder.capture(
        performance_module.get_return_statistics(
            within_period=True, confidence_level=0.99
        )
    )


def test_get_capture_ratio_methods(recorder, performance_module):
    recorder.capture(performance_module.get_upside_capture_ratio(method="annualized"))
    recorder.capture(performance_module.get_downside_capture_ratio(method="arithmetic"))
    recorder.capture(
        performance_module.get_upside_capture_ratio(period="monthly", rolling=12)
    )
    recorder.capture(
        performance_module.get_downside_capture_ratio(
            period="monthly", rolling=12, method="annualized"
        )
    )


def test_get_capture_ratios(recorder, performance_module):
    recorder.capture(performance_module.get_capture_ratios())
    recorder.capture(performance_module.get_capture_ratios(method="annualized"))

    # The table holds the single capture ratios and the up and down market win rates.
    table = performance_module.get_capture_ratios()
    pd.testing.assert_series_equal(
        table[("AAPL", "Upside Capture Ratio")],
        performance_module.get_upside_capture_ratio()["AAPL"].loc[table.index],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        table[("MSFT", "Down Percent")],
        performance_module.get_win_rate(market="down")["MSFT"].loc[table.index],
        check_names=False,
    )


def test_get_win_rate_market(recorder, performance_module):
    recorder.capture(performance_module.get_win_rate(market="up"))
    recorder.capture(performance_module.get_win_rate(market="down"))


def test_get_outperformance_probability(recorder, performance_module):
    recorder.capture(performance_module.get_outperformance_probability())
    recorder.capture(
        performance_module.get_outperformance_probability(
            period="quarterly", windows=[1, 4]
        )
    )
    recorder.capture(
        performance_module.get_outperformance_probability(relative=False, windows=[12])
    )


def test_get_bull_bear_beta(recorder, performance_module):
    recorder.capture(performance_module.get_bull_bear_beta())
    recorder.capture(performance_module.get_bull_bear_beta(market="return"))
    recorder.capture(performance_module.get_bull_bear_beta(period="quarterly"))


def test_get_beta_fragility(recorder, performance_module):
    recorder.capture(performance_module.get_beta_fragility())
    recorder.capture(performance_module.get_beta_fragility(period="quarterly"))


def test_get_stability_of_returns(recorder, performance_module):
    recorder.capture(performance_module.get_stability_of_returns())
    recorder.capture(performance_module.get_stability_of_returns(within_period=False))


def test_get_r_squared(recorder, performance_module):
    recorder.capture(performance_module.get_r_squared())
    recorder.capture(performance_module.get_r_squared(period="monthly", rolling=12))


def test_get_risk_return_ratio(recorder, performance_module):
    recorder.capture(performance_module.get_risk_return_ratio())
    recorder.capture(performance_module.get_risk_return_ratio(annualize=True))
    recorder.capture(
        performance_module.get_risk_return_ratio(
            period="monthly", rolling=12, annualize=True
        )
    )


def test_get_risk_decomposition(recorder, performance_module):
    risks = performance_module.get_risk_decomposition()
    recorder.capture(risks)
    recorder.capture(performance_module.get_risk_decomposition(period="quarterly"))

    # The regression splits the variance exactly, so the total risk is the volatility.
    total = risks.xs("Total Risk", level=1, axis=1)
    systematic = risks.xs("Systematic Risk", level=1, axis=1)
    specific = risks.xs("Specific Risk", level=1, axis=1)
    np.testing.assert_allclose(total, np.sqrt(systematic**2 + specific**2), atol=2e-4)


def test_get_capital_market_line(recorder, performance_module):
    recorder.capture(performance_module.get_capital_market_line())
    recorder.capture(performance_module.get_capital_market_line(period="quarterly"))


def test_get_returns_log_and_unsmoothed(recorder, performance_module):
    recorder.capture(performance_module.get_returns(logarithmic=True))
    recorder.capture(
        performance_module.get_returns(period="daily", logarithmic=True).iloc[:20]
    )
    recorder.capture(performance_module.get_returns(period="monthly", unsmoothed=True))

    # The cumulative log return is the sum of the log returns.
    log_returns = performance_module.get_returns(
        period="monthly", logarithmic=True, rounding=12
    )
    cumulative = performance_module.get_returns(
        period="monthly", logarithmic=True, cumulative=True, rounding=12
    )
    pd.testing.assert_frame_equal(
        log_returns.cumsum().round(8), cumulative.round(8), check_freq=False
    )


def test_get_serenity_index(recorder, performance_module):
    recorder.capture(performance_module.get_serenity_index())
    recorder.capture(performance_module.get_serenity_index(period="quarterly"))
    recorder.capture(performance_module.get_serenity_index(alpha=0.1))


def test_get_risk_measure_ratio(recorder, performance_module):
    recorder.capture(performance_module.get_risk_measure_ratio())
    recorder.capture(performance_module.get_risk_measure_ratio(risk_measure="cdar"))
    recorder.capture(
        performance_module.get_risk_measure_ratio(
            period="monthly", risk_measure="mad", within_period=False
        )
    )

    for risk_measure in performance_model.RISK_MEASURES:
        assert not performance_module.get_risk_measure_ratio(
            risk_measure=risk_measure
        ).empty

    # An unknown risk measure raises, naming the risk measures.
    with pytest.raises(InvalidArgumentError, match="risk_measure must be one of"):
        performance_module.get_risk_measure_ratio(risk_measure="other")


def test_get_probabilistic_momentum(recorder, performance_module):
    recorder.capture(performance_module.get_probabilistic_momentum())
    recorder.capture(performance_module.get_probabilistic_momentum(period="quarterly"))
    recorder.capture(
        performance_module.get_probabilistic_momentum(period="daily", rolling=60).iloc[
            -20:
        ]
    )


def test_get_trailing_returns(recorder, performance_module):
    trailing_returns = performance_module.get_trailing_returns()
    recorder.capture(trailing_returns)

    # The year to date return runs from the last close of the previous year.
    prices = performance_module._historical_data["daily"]["Adj Close"]["AAPL"]
    assert trailing_returns.loc["Year to Date", "AAPL"] == pytest.approx(
        prices.loc[:"2022-12-31"].iloc[-1] / prices.loc[:"2021-12-31"].iloc[-1] - 1,
        abs=1e-4,
    )


def test_get_calendar_returns(recorder, performance_module):
    calendar_returns = performance_module.get_calendar_returns(rounding=10)
    recorder.capture(performance_module.get_calendar_returns())

    # The months and the year equal the monthly and yearly returns.
    monthly = performance_module.get_returns(period="monthly", rounding=10)
    yearly = performance_module.get_returns(period="yearly", rounding=10)
    assert calendar_returns.loc["2021", ("MSFT", "Mar")] == pytest.approx(
        monthly.loc["2021-03", "MSFT"], abs=1e-9
    )
    assert calendar_returns.loc["2022", ("AAPL", "Year")] == pytest.approx(
        yearly.loc["2022", "AAPL"], abs=1e-9
    )


def test_get_bootstrap_statistics(recorder, performance_module):
    recorder.capture(performance_module.get_bootstrap_statistics(n_bootstrap=200))
    shuffled = performance_module.get_bootstrap_statistics(
        n_bootstrap=200, method="shuffle"
    )
    recorder.capture(shuffled)

    # Shuffling keeps the returns, so only the drawdown varies across the simulations.
    assert shuffled.loc["Cumulative Return", ("AAPL", "Std")] == 0
    assert shuffled.loc["Maximum Drawdown", ("AAPL", "Std")] > 0

    # The same seed gives the same result, another seed another one.
    pd.testing.assert_frame_equal(
        performance_module.get_bootstrap_statistics(n_bootstrap=50, seed=1),
        performance_module.get_bootstrap_statistics(n_bootstrap=50, seed=1),
    )
    assert not performance_module.get_bootstrap_statistics(
        n_bootstrap=50, seed=1
    ).equals(performance_module.get_bootstrap_statistics(n_bootstrap=50, seed=2))


def test_get_forecast_cone(recorder, performance_module):
    recorder.capture(performance_module.get_forecast_cone(horizon=10, simulations=200))
    recorder.capture(
        performance_module.get_forecast_cone(
            horizon=5, standard_deviations=[1.0], simulations=200
        )
    )


def test_get_probability_of_backtest_overfitting(recorder, performance_module):
    recorder.capture(performance_module.get_probability_of_backtest_overfitting())
    recorder.capture(
        performance_module.get_probability_of_backtest_overfitting(blocks=8)
    )

    # An odd number of blocks gives an empty result through handle_errors.
    assert performance_module.get_probability_of_backtest_overfitting(blocks=7).empty


def test_get_smoothing_index(recorder, performance_module):
    recorder.capture(performance_module.get_smoothing_index())
    recorder.capture(performance_module.get_smoothing_index(period="weekly", order=1))


def test_rolling_factor_models(recorder, performance_module, monkeypatch):
    # Synthetic daily factors on the Toolkit's dates, so the rolling regressions run offline.
    dates = performance_module._historical_data["daily"].index
    rng = np.random.default_rng(0)
    factors = pd.DataFrame(
        rng.normal(0, 0.01, (len(dates), 6)),
        index=dates,
        columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
    )
    factors["RF"] = 0.0001

    performance = copy.copy(performance_module)
    performance._fama_and_french_datasets = {"United States": factors}
    monkeypatch.setattr(
        performance_model,
        "obtain_fama_and_french_three_factor_dataset",
        lambda region: factors[["Mkt-RF", "SMB", "HML", "RF"]],
    )
    monkeypatch.setattr(
        performance_model,
        "obtain_carhart_momentum_dataset",
        lambda region: factors[["RMW"]].rename(columns={"RMW": "Mom"}),
    )

    fama_and_french = performance.get_fama_and_french_model(rolling=126)
    recorder.capture(fama_and_french.iloc[-10:])
    recorder.capture(performance.get_carhart_four_factor_model(rolling=126).iloc[-10:])

    # The last window equals a regression on the last 126 days.
    returns = performance._historical_data["daily"]["Return"]["AAPL"].iloc[-126:]
    regressors = np.column_stack(
        [np.ones(126), factors[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].iloc[-126:]]
    )
    coefficients = np.linalg.lstsq(regressors, returns - 0.0001, rcond=None)[0]
    np.testing.assert_allclose(
        fama_and_french["AAPL"].iloc[-1][["Intercept", "Mkt-RF Slope"]].to_numpy(),
        coefficients[:2],
        atol=1e-4,
    )

    with pytest.raises(InvalidArgumentError, match="method='multi'"):
        performance.get_fama_and_french_model(rolling=126, method="simple")


def _hedged_performance(with_portfolio_returns=True, excess_return_block=False):
    """
    A stock, its inverse fund and the Portfolio of the two in equal weights, rebalanced daily: the
    Portfolio's returns are nearly flat while each holding moves with the market.
    """
    rng = np.random.default_rng(7)
    days = pd.bdate_range("2021-01-01", "2022-12-30").to_period("D")
    market = rng.normal(0.0004, 0.012, len(days))
    returns = pd.DataFrame(
        {
            "SPY": market + rng.normal(0, 0.001, len(days)),
            "SH": -market + rng.normal(0, 0.001, len(days)),
            "Benchmark": market,
        },
        index=days,
    )
    returns["Portfolio"] = 0.5 * returns["SPY"] + 0.5 * returns["SH"]
    returns.iloc[0] = np.nan
    if not with_portfolio_returns:
        returns = returns.drop(columns="Portfolio")
    prices = (1 + returns.fillna(0)).cumprod() * 100
    daily = pd.concat({"Return": returns, "Adj Close": prices, "Close": prices}, axis=1)
    if excess_return_block:
        # Historical data supplied with an "Excess Return" block of its own, as older Portfolio data has.
        daily = pd.concat(
            [daily, pd.concat({"Excess Return": returns}, axis=1)], axis=1
        )
    historical_data = {"daily": daily, "intraday": pd.DataFrame()}
    period_prices = prices.groupby(days.asfreq("Y")).last()
    historical_data["yearly"] = pd.concat(
        {
            "Return": period_prices / period_prices.shift() - 1,
            "Adj Close": period_prices,
            "Close": period_prices,
        },
        axis=1,
    )
    performance = Performance(
        tickers=["SPY", "SH", "Portfolio"],
        historical_data=historical_data,
        risk_free_rate_data={"daily": pd.Series(0.0, index=days)},
        rounding=None,
    )
    weights = pd.DataFrame(
        {"SPY": 0.5, "SH": 0.5}, index=pd.period_range("2021", "2022", freq="Y")
    )
    performance._portfolio_weights = {"yearly": weights}
    return performance, returns


@pytest.mark.parametrize("excess_return_block", [False, True])
def test_the_performance_of_the_portfolio_comes_from_its_own_returns(
    excess_return_block,
):
    performance, returns = _hedged_performance(excess_return_block=excess_return_block)
    portfolio = returns["Portfolio"].loc["2022"]
    benchmark = returns["Benchmark"].loc["2022"]

    # With a risk free rate of 0 the excess returns are the returns themselves.
    sharpe_ratio = performance.get_sharpe_ratio(period="yearly")
    assert list(sharpe_ratio.columns) == ["SPY", "SH", "Portfolio"]
    assert sharpe_ratio.loc["2022", "Portfolio"] == pytest.approx(
        portfolio.mean() / portfolio.std()
    )
    beta = performance.get_beta(period="yearly")
    assert beta.loc["2022", "Portfolio"] == pytest.approx(
        np.cov(portfolio, benchmark)[0, 1] / benchmark.var()
    )
    active = portfolio - benchmark
    assert performance.get_information_ratio(period="yearly").loc[
        "2022", "Portfolio"
    ] == pytest.approx(active.mean() / active.std())
    assert performance.get_sortino_ratio(period="yearly").loc[
        "2022", "Portfolio"
    ] == pytest.approx(
        portfolio.mean() / np.sqrt((portfolio.clip(upper=0) ** 2).mean())
    )

    # The holdings move with and against the market, the Portfolio of the two hardly at all,
    # and its Sharpe ratio is not the average of theirs.
    assert abs(beta.loc["2022", "Portfolio"]) < 0.01  # noqa: PLR2004
    assert sharpe_ratio.loc["2022", "Portfolio"] != pytest.approx(
        sharpe_ratio.loc["2022", ["SPY", "SH"]].mean(), abs=1e-3
    )

    # The methods without a weighted average measure the Portfolio as well, except the
    # backtest overfitting, where the Portfolio of the tickers is no candidate of its own.
    capture_ratios = performance.get_capture_ratios(period="yearly")
    assert "Portfolio" in capture_ratios.columns.get_level_values(0)
    without_portfolio, _ = _hedged_performance(with_portfolio_returns=False)
    pd.testing.assert_series_equal(
        performance.get_probability_of_backtest_overfitting(blocks=4),
        without_portfolio.get_probability_of_backtest_overfitting(blocks=4),
    )


def test_a_portfolio_without_returns_gets_the_weighted_average():
    performance, _ = _hedged_performance(with_portfolio_returns=False)

    sharpe_ratio = performance.get_sharpe_ratio(period="yearly")
    assert sharpe_ratio.loc["2022", "Portfolio"] == pytest.approx(
        sharpe_ratio.loc["2022", ["SPY", "SH"]].mean()
    )


def test_get_variability_weighted_return(recorder, performance_module):
    recorder.capture(performance_module.get_variability_weighted_return())
    recorder.capture(
        performance_module.get_variability_weighted_return(
            period="monthly", within_period=False, tau=0.2, maximum_deviation=2.0
        )
    )


def test_get_return_concentration(recorder, performance_module):
    recorder.capture(performance_module.get_return_concentration())
    recorder.capture(performance_module.get_return_concentration(within_period=True))


def test_get_strategy_risk(recorder, performance_module):
    recorder.capture(performance_module.get_strategy_risk())
    recorder.capture(
        performance_module.get_strategy_risk(period="monthly", target_sharpe_ratio=1.0)
    )
    recorder.capture(performance_module.get_strategy_risk(within_period=True))


def test_get_sharpe_ratio_difference_test(recorder, performance_module):
    recorder.capture(performance_module.get_sharpe_ratio_difference_test())
    recorder.capture(
        performance_module.get_sharpe_ratio_difference_test(
            period="quarterly", method="ledoit_wolf"
        )
    )
    recorder.capture(
        performance_module.get_sharpe_ratio_difference_test(
            method="ledoit_wolf", lags=0
        )
    )
    with pytest.raises(InvalidArgumentError, match="'memmel' or 'ledoit_wolf'"):
        performance_module.get_sharpe_ratio_difference_test(method="other")


def _toolkit_between(start_date, end_date):
    """The offline Toolkit over another date range."""
    from financetoolkit import Toolkit

    toolkit = Toolkit(
        tickers=["AAPL", "MSFT"],
        historical=pd.read_pickle("tests/datasets/historical_dataset.pickle"),
        convert_currency=False,
        start_date=start_date,
        end_date=end_date,
        sleep_timer=False,
    )
    toolkit._daily_risk_free_rate = pd.read_pickle(
        "tests/datasets/risk_free_rate.pickle"
    )
    toolkit._daily_treasury_data = pd.read_pickle("tests/datasets/treasury_data.pickle")

    return toolkit


def test_return_statistics_of_a_range_without_data_are_empty(caplog):
    # The test data ends in 2022, so a range in 2023 has no returns.
    after_the_data = _toolkit_between("2023-06-01", "2023-12-31").performance

    with caplog.at_level("WARNING"):
        assert after_the_data.get_return_statistics().empty
        assert after_the_data.get_strategy_risk().empty

    assert "There are no daily returns between 2023-06-01" in caplog.text


def test_bull_bear_beta_of_a_week_is_nan(performance_module):
    weekly = performance_module.get_bull_bear_beta(period="weekly")

    assert weekly.empty or weekly.isna().all().all()
    assert performance_module.get_bull_bear_beta().notna().all().all()


def test_a_matrix_of_too_few_returns_is_logged(performance_module, caplog):
    short_range = _toolkit_between("2022-11-15", "2022-12-30").performance

    with caplog.at_level("WARNING"):
        performance_module.get_covariance_matrix()
    assert "too few for a covariance" not in caplog.text

    with caplog.at_level("WARNING"):
        short_range.get_correlation_matrix()
        short_range.get_covariance_matrix()
    assert caplog.text.count("too few for a covariance") == 2  # noqa: PLR2004
    assert short_range.get_covariance_matrix(period="daily").notna().all().all()


@pytest.mark.parametrize(
    "method",
    ["get_sharpe_ratio", "get_beta", "get_covariance_matrix", "get_smoothing_index"],
)
def test_an_unknown_period_raises(performance_module, method):
    with pytest.raises(InvalidArgumentError, match="period must be one of"):
        getattr(performance_module, method)(period="hourly")


def test_an_empty_minimum_track_record_length_is_logged(caplog):
    # AAPL and MSFT fell from mid-November to the end of 2022: both Sharpe ratios are negative.
    falling = _toolkit_between("2022-11-15", "2022-12-30").performance

    with caplog.at_level("WARNING"):
        assert falling.get_minimum_track_record_length().empty

    assert "does not exceed the benchmark Sharpe ratio" in caplog.text


@pytest.mark.parametrize(
    "method",
    [
        "get_beta",
        "get_sharpe_ratio",
        "get_ulcer_performance_index",
        "get_morningstar_risk",
        "get_minimum_track_record_length",
    ],
)
def test_a_rolling_window_needs_two_periods(performance_module, method):
    for rolling in (1, 0, 2.5):
        with pytest.raises(InvalidArgumentError, match="rolling is a number"):
            getattr(performance_module, method)(rolling=rolling)


def test_a_rolling_window_longer_than_the_data_is_logged_once(
    performance_module, caplog
):
    with caplog.at_level("WARNING"):
        sharpe_ratio = performance_module.get_sharpe_ratio(
            period="monthly", rolling=1000
        )

    assert sharpe_ratio.empty or sharpe_ratio.isna().all().all()
    assert caplog.text.count("is at least as long as the") == 1


def test_an_unknown_metric_of_the_factor_analysis_raises(performance_module):
    with pytest.raises(InvalidArgumentError, match="not a metric of the Performance"):
        performance_module.get_factor_analysis(metrics="bogus")


def test_collect_all_metrics_logs_a_repeated_message_once(caplog):
    after_the_data = _toolkit_between("2023-06-01", "2023-12-31").performance

    with caplog.at_level("WARNING"):
        after_the_data.collect_all_metrics()

    assert caplog.text.count("There are no observations between 2023-06-01") == 1


def test_the_rolling_appraisal_ratio_needs_three_periods(performance_module):
    # A regression on two returns leaves no residual to scale the alpha by.
    with pytest.raises(InvalidArgumentError, match="at least 3"):
        performance_module.get_appraisal_ratio(rolling=2)
    assert performance_module.get_appraisal_ratio(rolling=3).notna().any().any()


def test_the_growth_of_a_rolling_ratio_counts_the_periods_with_a_value(
    performance_module,
):
    # Rolling over two of the three yearly returns leaves two ratios, whose growth exists.
    growth = performance_module.get_sharpe_ratio(
        period="yearly", rolling=2, growth=True
    )

    assert growth.notna().any().any()


@pytest.mark.parametrize("period", ["weekly", "monthly"])
def test_bull_bear_beta_on_short_periods_says_why_it_is_empty(
    performance_module, period, caplog
):
    with caplog.at_level("WARNING", logger="financetoolkit"):
        result = performance_module.get_bull_bear_beta(period=period)

    assert result.empty
    assert caplog.text.count("need at least 20 days with returns on each side") == 1
    assert "Use quarterly or yearly periods." in caplog.text


@pytest.mark.parametrize("period", ["quarterly", "yearly"])
def test_bull_bear_beta_on_long_periods_advises_a_longer_date_range(
    performance_module, period, caplog, monkeypatch
):
    # Raising the minimum leaves every quarter and year short of days, as a short date range does.  # noqa: E501
    monkeypatch.setattr(
        "financetoolkit.performance.performance_controller.MINIMUM_BULL_BEAR_OBSERVATIONS",
        10_000,
    )
    with caplog.at_level("WARNING", logger="financetoolkit"):
        result = performance_module.get_bull_bear_beta(period=period)

    assert result.empty
    assert "Use a longer date range." in caplog.text
    assert "Use quarterly or yearly periods." not in caplog.text
