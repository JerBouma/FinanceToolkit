"""Fulmer Model Tests""" ""
import math

import pandas as pd
import pytest

from financetoolkit.models import fulmer_model

# pylint: disable=missing-function-docstring


def test_get_retained_earnings_to_total_assets_ratio(recorder):
    recorder.capture(
        fulmer_model.get_retained_earnings_to_total_assets_ratio(
            retained_earnings=pd.Series([100, 110, 120, 130, 80]),
            total_assets=pd.Series([500, 400, 300, 200, 100]),
        )
    )


def test_get_sales_to_total_assets_ratio(recorder):
    recorder.capture(
        fulmer_model.get_sales_to_total_assets_ratio(
            sales=pd.Series([100, 110, 120, 130, 80]),
            total_assets=pd.Series([500, 400, 300, 200, 100]),
        )
    )


def test_get_ebt_to_equity_ratio(recorder):
    recorder.capture(
        fulmer_model.get_ebt_to_equity_ratio(
            ebt=pd.Series([100, 110, 120, 130, 80]),
            total_equity=pd.Series([500, 400, 300, 200, 100]),
        )
    )


def test_get_cash_flow_to_total_liabilities_ratio(recorder):
    recorder.capture(
        fulmer_model.get_cash_flow_to_total_liabilities_ratio(
            cash_flow=pd.Series([100, 110, 120, 130, 80]),
            total_liabilities=pd.Series([500, 400, 300, 200, 100]),
        )
    )


def test_get_debt_to_total_assets_ratio(recorder):
    recorder.capture(
        fulmer_model.get_debt_to_total_assets_ratio(
            total_debt=pd.Series([100, 110, 120, 130, 80]),
            total_assets=pd.Series([500, 400, 300, 200, 100]),
        )
    )


def test_get_current_liabilities_to_total_assets_ratio(recorder):
    recorder.capture(
        fulmer_model.get_current_liabilities_to_total_assets_ratio(
            current_liabilities=pd.Series([100, 110, 120, 130, 80]),
            total_assets=pd.Series([500, 400, 300, 200, 100]),
        )
    )


def test_get_tangible_total_assets(recorder):
    recorder.capture(
        fulmer_model.get_tangible_total_assets(
            total_assets=pd.Series([1000, 900, 800]),
            goodwill_and_intangible_assets=pd.Series([100, 90, 80]),
        )
    )
    assert fulmer_model.get_tangible_total_assets(1000, 100) == 900


def test_get_log_of_tangible_total_assets(recorder):
    recorder.capture(
        fulmer_model.get_log_of_tangible_total_assets(
            tangible_total_assets=pd.Series([900, 450, 100]),
        )
    )
    assert round(fulmer_model.get_log_of_tangible_total_assets(900), 4) == round(
        math.log(900), 4
    )


def test_get_working_capital_to_total_liabilities_ratio(recorder):
    recorder.capture(
        fulmer_model.get_working_capital_to_total_liabilities_ratio(
            working_capital=pd.Series([100, 110, 120, 130, 80]),
            total_liabilities=pd.Series([500, 400, 300, 200, 100]),
        )
    )


def test_get_log_of_ebit_to_interest_expense_ratio(recorder):
    recorder.capture(
        fulmer_model.get_log_of_ebit_to_interest_expense_ratio(
            ebit=pd.Series([180, 10]),
            interest_expense=pd.Series([20, 15]),
        )
    )
    assert round(
        fulmer_model.get_log_of_ebit_to_interest_expense_ratio(180, 20), 4
    ) == round(math.log(9), 4)


def test_get_fulmer_h_score(recorder):
    recorder.capture(
        fulmer_model.get_fulmer_h_score(
            retained_earnings_to_total_assets_ratio=pd.Series([0.3, -0.1]),
            sales_to_total_assets_ratio=pd.Series([1.2, 0.6]),
            ebt_to_equity_ratio=pd.Series([0.3, -0.8]),
            cash_flow_to_total_liabilities_ratio=pd.Series([0.5, -0.0667]),
            debt_to_total_assets_ratio=pd.Series([0.25, 0.6]),
            current_liabilities_to_total_assets_ratio=pd.Series([0.2, 0.5]),
            log_of_tangible_total_assets=pd.Series([math.log(900), math.log(450)]),
            working_capital_to_total_liabilities_ratio=pd.Series([0.375, -0.0889]),
            log_of_ebit_to_interest_expense_ratio=pd.Series(
                [math.log(9), math.log(10 / 15)]
            ),
        )
    )
    # Synthetic ground truth: a healthy company above 0, a distressed one below.
    healthy = fulmer_model.get_fulmer_h_score(
        retained_earnings_to_total_assets_ratio=0.3,
        sales_to_total_assets_ratio=1.2,
        ebt_to_equity_ratio=0.3,
        cash_flow_to_total_liabilities_ratio=0.5,
        debt_to_total_assets_ratio=0.25,
        current_liabilities_to_total_assets_ratio=0.2,
        log_of_tangible_total_assets=math.log(900),
        working_capital_to_total_liabilities_ratio=0.375,
        log_of_ebit_to_interest_expense_ratio=math.log(9),
    )
    distressed = fulmer_model.get_fulmer_h_score(
        retained_earnings_to_total_assets_ratio=-0.1,
        sales_to_total_assets_ratio=0.6,
        ebt_to_equity_ratio=-0.8,
        cash_flow_to_total_liabilities_ratio=-0.0667,
        debt_to_total_assets_ratio=0.6,
        current_liabilities_to_total_assets_ratio=0.5,
        log_of_tangible_total_assets=math.log(450),
        working_capital_to_total_liabilities_ratio=-0.0889,
        log_of_ebit_to_interest_expense_ratio=math.log(10 / 15),
    )
    assert round(healthy, 4) == 3.2135
    assert healthy > 0
    assert round(distressed, 4) == -2.4942
    assert distressed < 0


def test_get_fulmer_h_score_type_error():
    with pytest.raises(TypeError):
        fulmer_model.get_fulmer_h_score(
            retained_earnings_to_total_assets_ratio="not a number",
            sales_to_total_assets_ratio=1.2,
            ebt_to_equity_ratio=0.3,
            cash_flow_to_total_liabilities_ratio=0.5,
            debt_to_total_assets_ratio=0.25,
            current_liabilities_to_total_assets_ratio=0.2,
            log_of_tangible_total_assets=math.log(900),
            working_capital_to_total_liabilities_ratio=0.375,
            log_of_ebit_to_interest_expense_ratio=math.log(9),
        )
