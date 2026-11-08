"""Damodaran Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.models import damodaran_model


def industry_sheet(total_label: str = "Total Market") -> pd.DataFrame:
    """An "Industry Averages" worksheet in Damodaran's layout, read without a header."""
    rows = [
        ["Date updated:", "2026-01-05", None, None, None, None, None, None],
        [
            "What is this data?",
            "Enterprise value multiples",
            None,
            None,
            None,
            None,
            None,
            None,
        ],
        [None, None, "Only positive EBITDA firms", None, "All firms", None, None, None],
        [
            "Industry Name",
            "Number of firms",
            "EV/EBITDA ",
            "EV/EBIT",
            "EV/EBITDA",
            "Average correlation with the market",
            2023.0,
            "Average (2022-2026)",
        ],
        ["Advertising", 52, 12.0, 20.9, 15.1, 0.21, 1.1, 1.2],
        ["Semiconductor", 66, 30.5, 40.2, 31.0, 0.35, 1.4, 1.5],
        [None, None, None, None, None, None, None, None],
        [total_label, 5994, 14.0, 22.0, 16.0, 0.30, 0.9, 0.8],
    ]
    return pd.DataFrame(rows)


def test_the_industry_sheet_is_read_from_its_header_row():
    data = damodaran_model.parse_industry_sheet(industry_sheet())

    assert data.index.tolist() == ["Advertising", "Semiconductor", "Total Market"]
    assert data.loc["Semiconductor", "Number of Firms"] == 66
    assert data.loc["Advertising", "Average correlation with the market"] == 0.21


def test_columns_that_share_a_name_carry_their_group():
    data = damodaran_model.parse_industry_sheet(industry_sheet())

    assert data.loc["Semiconductor", "EV/EBITDA (Only positive EBITDA firms)"] == 30.5
    assert data.loc["Semiconductor", "EV/EBITDA (All firms)"] == 31.0
    # A name that is not shared keeps its own name.
    assert "EV/EBIT" in data.columns


def test_the_yearly_beta_history_is_left_out():
    data = damodaran_model.parse_industry_sheet(industry_sheet())

    assert "2023" not in data.columns
    assert not any(column.startswith("Average (") for column in data.columns)


def test_the_european_grand_total_is_called_the_total_market():
    data = damodaran_model.parse_industry_sheet(industry_sheet("Grand Total"))

    assert data.loc["Total Market", "EV/EBIT"] == 22.0


def test_a_sheet_without_a_header_row_means_the_layout_changed():
    sheet = pd.DataFrame([["Notes", 1.0], ["Advertising", np.nan]])

    with pytest.raises(ValueError, match="layout changed"):
        damodaran_model.parse_industry_sheet(sheet)


@pytest.mark.parametrize(
    ("dataset", "region", "file_name"),
    [
        ("betas", "United States", "betas"),
        ("betas", "Europe", "betaEurope"),
        ("cost_of_capital", "Emerging Markets", "waccemerg"),
        ("price_to_earnings", "United States", "pedata"),
        ("price_to_earnings", "Global", "peGlobal"),
        (
            "enterprise_value_multiples",
            "Australia, New Zealand and Canada",
            "vebitdaRest",
        ),
    ],
)
def test_each_dataset_and_region_has_its_own_file(dataset, region, file_name):
    assert damodaran_model.get_industry_file_name(dataset, region) == file_name


@pytest.mark.parametrize(
    ("dataset", "region"), [("wacc", "Europe"), ("betas", "Germany")]
)
def test_an_unknown_dataset_or_region_is_refused(dataset, region):
    with pytest.raises(ValueError, match="must be one of"):
        damodaran_model.get_industry_file_name(dataset, region)
