"""Performance Controller Tests""" ""
# pylint: disable=missing-function-docstring

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
    recorder.capture(performance_module.get_sharpe_ratio(rolling=10))
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
    days = pd.period_range("2024-01-01", periods=10, freq="D")
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
