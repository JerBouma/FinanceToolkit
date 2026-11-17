"""Performance Controller Tests""" ""

import copy
from functools import partial

import numpy as np
import pandas as pd
import pytest

from financetoolkit.performance.performance_controller import Performance


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


def test_get_jensens_alpha(recorder, performance_module):
    recorder.capture(performance_module.get_jensens_alpha())
    recorder.capture(performance_module.get_jensens_alpha(period="quarterly"))
    recorder.capture(performance_module.get_jensens_alpha(growth=True))
    recorder.capture(performance_module.get_jensens_alpha(growth=True, lag=[1, 2, 3]))
    recorder.capture(performance_module.get_jensens_alpha(period="monthly", rolling=6))


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
    assert performance_module.get_sharpe_ratio_standard_error(method="other").empty


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
    jensens_alpha = performance_module.get_jensens_alpha()
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


def test_get_treynor_mazuy_model(recorder, performance_module):
    recorder.capture(performance_module.get_treynor_mazuy_model())
    recorder.capture(performance_module.get_treynor_mazuy_model(period="quarterly"))
    recorder.capture(performance_module.get_treynor_mazuy_model(growth=True))


def test_get_henriksson_merton_model(recorder, performance_module):
    recorder.capture(performance_module.get_henriksson_merton_model())
    recorder.capture(performance_module.get_henriksson_merton_model(period="quarterly"))
    recorder.capture(performance_module.get_henriksson_merton_model(growth=True))


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
        1 + performance_module._risk_free_rate_data["monthly"], axis=0
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


def test_get_sterling_ratio_divides_the_year_by_its_adjusted_average_drawdown(
    performance_module,
):
    drawdowns = _drawdowns(_daily_returns_in(performance_module, "MSFT", "2022"))
    year_return = _year_return(performance_module, "MSFT", "2022")

    for adjustment in (0.1, 0.0):
        result = performance_module.get_sterling_ratio(
            adjustment=adjustment, rounding=None
        )
        assert result.loc["2022", "MSFT"] == pytest.approx(
            year_return / (abs(drawdowns[drawdowns < 0].mean()) + adjustment),
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


def _weekend_performance(rate=0.04):
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
    prices = (1 + returns.fillna(0)).cumprod() * 100
    daily = pd.concat({"Return": returns, "Adj Close": prices, "Close": prices}, axis=1)
    yearly_prices = prices.groupby(days.asfreq("Y")).last()
    yearly = pd.concat(
        {
            "Return": yearly_prices / yearly_prices.shift() - 1,
            "Adj Close": yearly_prices,
            "Close": yearly_prices,
        },
        axis=1,
    )
    return Performance(
        tickers=["BTC", "SPY"],
        historical_data={"daily": daily, "yearly": yearly, "intraday": pd.DataFrame()},
        risk_free_rate_data={"daily": pd.Series(rate, index=days)},
        rounding=None,
    )


def test_a_partial_period_earns_the_risk_free_rate_of_its_days(performance_module):
    annualized_rate = performance_module._annualized_risk_free_rate_data["daily"]
    returns = performance_module._historical_data["daily"]["Return"]
    yearly_rate = performance_module._risk_free_rate_data["yearly"]

    # Every year, including a partial one, compounds the annualized rate quoted on each
    # date with a return over the calendar days since the date before, as the yearly
    # return compounds the daily returns.
    dates = returns.dropna(how="all").index
    first = returns.index.get_loc(dates[0])
    # The first date accrues from the date before it, or one day when the data start there.
    previous = returns.index[first - 1].ordinal if first else dates[0].ordinal - 1
    days = np.diff(np.append(previous, dates.asi8))
    rate = annualized_rate.sort_index().reindex(dates, method="ffill").bfill()
    growth = pd.Series((1 + rate.to_numpy()) ** (days / 365), index=dates)
    for year in yearly_rate.index:
        expected = growth[dates.asfreq("Y") == year].prod() - 1
        assert yearly_rate[year] == pytest.approx(expected, rel=1e-12)

    performance = _weekend_performance()
    # The 181 days of the first half of 2022 at a flat 4% earn about half of it, not the
    # full year's rate.
    rate_2022 = performance._risk_free_rate_data["yearly"]["2022"]
    assert rate_2022 == pytest.approx(1.04 ** (181 / 365) - 1, rel=1e-9)

    excess = performance.get_excess_return(period="yearly")
    returns_2022 = performance._historical_data["yearly"]["Return"].loc["2022"]
    assert excess.loc["2022", "SPY"] == pytest.approx(
        returns_2022["SPY"] - rate_2022, rel=1e-9
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
    performance.__dict__.pop("_risk_free_rate_per_date", None)
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
    risk_free = performance_module._risk_free_rate_data["monthly"]
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
    y = window["AAPL"] - risk_free.reindex(window.index)
    x = window["Benchmark"] - risk_free.reindex(window.index)
    slope, intercept = np.polyfit(x, y, 1)
    residual_standard_deviation = np.sqrt(((y - slope * x - intercept) ** 2).sum() / 11)
    alpha = performance_module.get_jensens_alpha(
        period="monthly", rolling=12, rounding=10
    )
    assert ratio.loc[end, "AAPL"] == pytest.approx(
        alpha.loc[end, "AAPL"] / residual_standard_deviation, abs=1e-8
    )
