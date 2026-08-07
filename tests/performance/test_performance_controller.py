"""Performance Controller Tests""" ""
# pylint: disable=missing-function-docstring

from functools import partial

import pytest


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
