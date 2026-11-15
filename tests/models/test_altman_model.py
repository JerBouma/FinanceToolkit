"""Altman Model Tests""" ""
import pandas as pd
import pytest

from financetoolkit.models.distress import altman_model


def test_get_working_capital_to_total_assets_ratio(recorder):
    recorder.capture(
        altman_model.get_working_capital_to_total_assets_ratio(
            working_capital=pd.Series([100, 110, 120, 130, 80]),
            total_assets=pd.Series([500, 400, 300, 200, 100]),
        )
    )


def test_get_retained_earnings_to_total_assets_ratio(recorder):
    recorder.capture(
        altman_model.get_retained_earnings_to_total_assets_ratio(
            retained_earnings=pd.Series([100, 110, 120, 130, 80]),
            total_assets=pd.Series([500, 400, 300, 200, 100]),
        )
    )


def test_get_earnings_before_interest_and_taxes_to_total_assets_ratio(recorder):
    recorder.capture(
        altman_model.get_earnings_before_interest_and_taxes_to_total_assets_ratio(
            ebit=pd.Series([100, 110, 120, 130, 80]),
            total_assets=pd.Series([500, 400, 300, 200, 100]),
        )
    )


def test_get_market_value_of_equity_to_book_value_of_total_liabilities_ratio(recorder):
    recorder.capture(
        altman_model.get_market_value_of_equity_to_book_value_of_total_liabilities_ratio(
            market_value_of_equity=pd.Series([100, 110, 120, 130, 80]),
            total_liabilities=pd.Series([500, 400, 300, 200, 100]),
        )
    )


def test_get_sales_to_total_assets_ratio(recorder):
    recorder.capture(
        altman_model.get_sales_to_total_assets_ratio(
            sales=pd.Series([100, 110, 120, 130, 80]),
            total_assets=pd.Series([500, 400, 300, 200, 100]),
        )
    )


def test_get_altman_z_score(recorder):
    recorder.capture(
        altman_model.get_altman_z_score(
            working_capital_to_total_assets_ratio=pd.Series([0.2, 0.2, 0.4, 0.5, 0.6]),
            retained_earnings_to_total_assets_ratio=pd.Series(
                [0.2, 0.3, 0.3, 0.5, 0.6]
            ),
            earnings_before_interest_and_taxes_to_total_assets_ratio=pd.Series(
                [0.2, 0.3, 0.3, 0.5, 0.6]
            ),
            equity_to_total_liabilities_ratio=pd.Series([0.2, 0.3, 0.4, 0.3, 0.6]),
            sales_to_total_assets_ratio=pd.Series([0.2, 0.3, 0.4, 0.4, 0.6]),
        )
    )


def test_get_book_value_of_equity_to_book_value_of_total_liabilities_ratio(recorder):
    recorder.capture(
        altman_model.get_book_value_of_equity_to_book_value_of_total_liabilities_ratio(
            book_value_of_equity=pd.Series([100, 110, -120, 130, 80]),
            total_liabilities=pd.Series([500, 400, 300, 200, 100]),
        )
    )


def test_get_altman_z_score_models():
    ratios = {
        "working_capital_to_total_assets_ratio": pd.Series([0.2, -0.1]),
        "retained_earnings_to_total_assets_ratio": pd.Series([0.3, -0.2]),
        "earnings_before_interest_and_taxes_to_total_assets_ratio": pd.Series(
            [0.1, -0.05]
        ),
        "equity_to_total_liabilities_ratio": pd.Series([0.8, 0.1]),
        "sales_to_total_assets_ratio": pd.Series([1.1, 0.6]),
    }
    x1, x2, x3, x4, x5 = (series.iloc[0] for series in ratios.values())

    expected = {
        "original": 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5,
        "private": 0.717 * x1 + 0.847 * x2 + 3.107 * x3 + 0.420 * x4 + 0.998 * x5,
        "non-manufacturing": 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4,
        "emerging": 3.25 + 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4,
    }

    for model, value in expected.items():
        score = altman_model.get_altman_z_score(**ratios, model=model)
        assert abs(score.iloc[0] - value) < 1e-12

    # The four-variable models have no sales term, so they ignore it and work without it.
    without_sales = {
        k: v for k, v in ratios.items() if k != "sales_to_total_assets_ratio"
    }
    assert altman_model.get_altman_z_score(
        **without_sales, model="non-manufacturing"
    ).equals(altman_model.get_altman_z_score(**ratios, model="non-manufacturing"))

    with pytest.raises(ValueError, match="Sales to Total Assets"):
        altman_model.get_altman_z_score(**without_sales, model="private")

    with pytest.raises(ValueError, match="Invalid model"):
        altman_model.get_altman_z_score(**ratios, model="zeta")
