"""Models Controller Tests""" ""
# pylint: disable=missing-function-docstring


def test_get_dupont_analysis(recorder, models_module):
    recorder.capture(models_module.get_dupont_analysis())
    recorder.capture(models_module.get_dupont_analysis(growth=True))
    recorder.capture(models_module.get_dupont_analysis(growth=True, lag=[1, 2, 3]))


def test_get_extended_dupont_analysis(recorder, models_module):
    recorder.capture(models_module.get_extended_dupont_analysis())
    recorder.capture(models_module.get_extended_dupont_analysis(growth=True))
    recorder.capture(
        models_module.get_extended_dupont_analysis(growth=True, lag=[1, 2, 3])
    )


def test_get_enterprise_value_breakdown(recorder, models_module):
    recorder.capture(models_module.get_enterprise_value_breakdown())
    recorder.capture(models_module.get_enterprise_value_breakdown(growth=True))
    recorder.capture(
        models_module.get_enterprise_value_breakdown(growth=True, lag=[1, 2, 3])
    )


def test_get_weighted_average_cost_of_capital(recorder, models_module):
    recorder.capture(models_module.get_weighted_average_cost_of_capital().round(1))
    recorder.capture(
        models_module.get_weighted_average_cost_of_capital(growth=True).round(1)
    )
    recorder.capture(
        models_module.get_weighted_average_cost_of_capital(growth=True, lag=[1, 2, 3])
    )


def test_get_intrinsic_valuation(recorder, models_module):
    recorder.capture(
        models_module.get_intrinsic_valuation(
            growth_rate=0.05,
            perpetual_growth_rate=0.02,
            weighted_average_cost_of_capital=0.10,
        )
    )
    recorder.capture(
        models_module.get_intrinsic_valuation(
            growth_rate=[0.05, 0.08],
            perpetual_growth_rate=0.02,
            weighted_average_cost_of_capital=0.10,
        )
    )


def test_get_gorden_growth_model(recorder, models_module):
    recorder.capture(
        models_module.get_gorden_growth_model(rate_of_return=0.1, growth_rate=0.09)
    )
    recorder.capture(
        models_module.get_gorden_growth_model(
            rate_of_return=0.2, growth_rate=0.1, project_periods=10
        )
    )


def test_get_altman_z_score(recorder, models_module):
    recorder.capture(models_module.get_altman_z_score())
    recorder.capture(models_module.get_altman_z_score(growth=True))
    recorder.capture(models_module.get_altman_z_score(growth=True, lag=[1, 2, 3]))


def test_get_piotroski_score(recorder, models_module):
    recorder.capture(models_module.get_piotroski_score())


def test_get_present_value_of_growth_opportunities(recorder, models_module):
    recorder.capture(models_module.get_present_value_of_growth_opportunities())
    recorder.capture(
        models_module.get_present_value_of_growth_opportunities(calculate_daily=True)
    )


def test_get_economic_value_added(recorder, models_module):
    recorder.capture(models_module.get_economic_value_added())
    recorder.capture(models_module.get_economic_value_added(growth=True))


def test_get_beneish_m_score(recorder, models_module):
    recorder.capture(models_module.get_beneish_m_score())
    recorder.capture(models_module.get_beneish_m_score(growth=True))


def test_get_sustainable_growth_rate(recorder, models_module):
    recorder.capture(models_module.get_sustainable_growth_rate())
    recorder.capture(models_module.get_sustainable_growth_rate(growth=True))


def test_get_internal_growth_rate(recorder, models_module):
    recorder.capture(models_module.get_internal_growth_rate())
    recorder.capture(models_module.get_internal_growth_rate(growth=True))
