"""Ohlson Model Tests""" ""
import pandas as pd

from financetoolkit.models.distress import ohlson_model

COLUMNS = ["2020", "2021", "2022", "2023", "2024"]


def test_get_log_of_total_assets(recorder):
    recorder.capture(
        ohlson_model.get_log_of_total_assets(
            total_assets=pd.Series([500, 400, 300, 200, 100]),
        )
    )


def test_get_total_liabilities_to_total_assets_ratio(recorder):
    recorder.capture(
        ohlson_model.get_total_liabilities_to_total_assets_ratio(
            total_liabilities=pd.Series([100, 110, 120, 130, 80]),
            total_assets=pd.Series([500, 400, 300, 200, 100]),
        )
    )


def test_get_current_liabilities_to_current_assets_ratio(recorder):
    recorder.capture(
        ohlson_model.get_current_liabilities_to_current_assets_ratio(
            current_liabilities=pd.Series([200, 210, 220, 230, 240]),
            current_assets=pd.Series([300, 320, 340, 360, 380]),
        )
    )


def test_get_negative_equity_indicator(recorder):
    recorder.capture(
        ohlson_model.get_negative_equity_indicator(
            total_liabilities=pd.Series([100, 500, 120, 900, 80]),
            total_assets=pd.Series([500, 400, 300, 200, 100]),
        )
    )


def test_get_net_income_to_total_assets_ratio(recorder):
    recorder.capture(
        ohlson_model.get_net_income_to_total_assets_ratio(
            net_income=pd.Series([100, -110, 120, 130, -80]),
            total_assets=pd.Series([500, 400, 300, 200, 100]),
        )
    )


def test_get_funds_from_operations_to_total_liabilities_ratio(recorder):
    recorder.capture(
        ohlson_model.get_funds_from_operations_to_total_liabilities_ratio(
            funds_from_operations=pd.Series([120, 115, 118, 160, 150]),
            total_liabilities=pd.Series([100, 110, 120, 130, 80]),
        )
    )


def test_get_negative_income_indicator(recorder):
    recorder.capture(
        ohlson_model.get_negative_income_indicator(
            net_income=pd.DataFrame(
                [[-10, -20, 5, -30, -40]], index=["AAPL"], columns=COLUMNS
            ),
        )
    )


def test_get_change_in_net_income_ratio(recorder):
    recorder.capture(
        ohlson_model.get_change_in_net_income_ratio(
            net_income=pd.DataFrame(
                [[100, 110, 90, -20, 30]], index=["AAPL"], columns=COLUMNS
            ),
        )
    )


def test_get_ohlson_o_score(recorder):
    recorder.capture(
        ohlson_model.get_ohlson_o_score(
            log_of_total_assets=pd.Series([6.2, 6.0, 5.8, 5.6, 5.4]),
            total_liabilities_to_total_assets_ratio=pd.Series(
                [0.5, 0.6, 0.7, 0.8, 0.9]
            ),
            working_capital_to_total_assets_ratio=pd.Series(
                [0.2, 0.1, 0.0, -0.1, -0.2]
            ),
            current_liabilities_to_current_assets_ratio=pd.Series(
                [0.8, 0.9, 1.0, 1.1, 1.2]
            ),
            negative_equity_indicator=pd.Series([0, 0, 0, 1, 1]),
            net_income_to_total_assets_ratio=pd.Series([0.1, 0.05, 0.0, -0.05, -0.1]),
            funds_from_operations_to_total_liabilities_ratio=pd.Series(
                [0.4, 0.3, 0.2, 0.1, 0.0]
            ),
            negative_income_indicator=pd.Series([0, 0, 0, 1, 1]),
            change_in_net_income_ratio=pd.Series([0.1, 0.05, 0.0, -0.1, -0.2]),
        )
    )


def test_get_ohlson_bankruptcy_probability(recorder):
    recorder.capture(
        ohlson_model.get_ohlson_bankruptcy_probability(
            ohlson_o_score=pd.Series([-5.0, -2.0, 0.0, 2.0, 5.0]),
        )
    )


def test_log_of_total_assets_is_deflated_by_the_price_level_index():
    import numpy as np  # noqa: PLC0415

    total_assets = pd.DataFrame(
        [[1e9, 2e9]], index=["A"], columns=pd.PeriodIndex(["2020", "2021"], freq="Y")
    )
    index = pd.Series([600.0, 650.0], index=total_assets.columns)

    size = ohlson_model.get_log_of_total_assets(total_assets, index)

    # Ohlson (1980): SIZE = ln(Total Assets / GNP price-level index), 1968 = 100.
    assert size.iloc[0, 0] == np.log(1e9 / 600)
    assert size.iloc[0, 1] == np.log(2e9 / 650)
    assert ohlson_model.get_log_of_total_assets(1e9, 600.0) == np.log(1e9 / 600)
    # Without an index, the total assets are not deflated.
    assert ohlson_model.get_log_of_total_assets(1e9) == np.log(1e9)


def test_get_ohlson_o_score_keeps_a_missing_negative_equity_indicator_missing():
    ones = pd.Series([1.0, 1.0, 1.0])
    indicator = pd.Series([0.0, float("nan"), 1.0])

    result = ohlson_model.get_ohlson_o_score(
        log_of_total_assets=ones,
        total_liabilities_to_total_assets_ratio=ones,
        working_capital_to_total_assets_ratio=ones,
        current_liabilities_to_current_assets_ratio=ones,
        negative_equity_indicator=indicator,
        net_income_to_total_assets_ratio=ones,
        funds_from_operations_to_total_liabilities_ratio=ones,
        negative_income_indicator=pd.Series([0.0, 0.0, 0.0]),
        change_in_net_income_ratio=ones,
    )

    assert result.notna().tolist() == [True, False, True]
    # The indicator of 1 lowers the score by its coefficient of 1.72.
    assert round(result[0] - result[2], 2) == 1.72
