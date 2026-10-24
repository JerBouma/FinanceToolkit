"""Style Analysis and Regional Factor Tests"""

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from financetoolkit.performance import performance_model

# pylint: disable=missing-function-docstring


def zipped(text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("factors.csv", text)
    return buffer.getvalue()


def test_regional_factor_files_are_read():
    text = (
        "This file was created using the 202608 Bloomberg database.\n\n"
        "Missing data are indicated by -99.99.\n\n\n\n"
        ",Mkt-RF,SMB,HML,RMW,CMA,RF\n"
        "19900702    ,0.69   ,-0.08   ,-0.33    ,0.21    ,0.00    ,0.03\n"
        "19900703    ,-99.99   ,0.10   ,0.20    ,0.30    ,0.40    ,0.03\n"
    )

    factors = performance_model.read_factor_file(zipped(text))

    assert list(factors.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert factors.loc[pd.Period("1990-07-02", "D"), "Mkt-RF"] == pytest.approx(0.0069)
    assert np.isnan(factors.loc[pd.Period("1990-07-03", "D"), "Mkt-RF"])


def test_united_states_factor_files_are_read():
    text = (
        "This file was created by using the 202608 CRSP database.\n"
        "The Tbill return is the simple daily rate.\n\n"
        ",Mkt-RF,SMB,HML,RF\n"
        "19630701,   -0.67,    0.00,   -0.33,    0.01\n\n"
        "Copyright 2026 Eugene F. Fama and Kenneth R. French\n"
    )

    factors = performance_model.read_factor_file(zipped(text))

    assert len(factors) == 1
    assert factors.iloc[0]["Mkt-RF"] == pytest.approx(-0.0067)


def test_factor_urls_per_region():
    assert performance_model.factor_url().endswith(
        "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    )
    assert performance_model.factor_url("Europe", "momentum").endswith(
        "Europe_Mom_Factor_Daily_CSV.zip"
    )
    with pytest.raises(ValueError, match="not published"):
        performance_model.factor_url("Mars")


def test_style_analysis_recovers_the_mix():
    rng = np.random.default_rng(0)
    styles = pd.DataFrame(
        rng.normal(0.006, 0.04, (120, 4)), columns=["Growth", "Value", "Small", "Bonds"]
    )
    fund = 0.5 * styles["Growth"] + 0.3 * styles["Value"] + 0.2 * styles["Bonds"]
    fund = fund + rng.normal(0.001, 0.003, 120)

    result = performance_model.get_style_analysis(fund, styles)

    assert result[["Growth", "Value", "Small", "Bonds"]].to_numpy() == pytest.approx(
        [0.5, 0.3, 0.0, 0.2], abs=0.03
    )
    assert result[["Growth", "Value", "Small", "Bonds"]].sum() == pytest.approx(1)
    assert result["R Squared"] > 0.95  # noqa: PLR2004
    assert result["Selection Return"] == pytest.approx(0.012, abs=0.006)


def test_style_analysis_on_the_toolkit(performance_module):
    result = performance_module.get_style_analysis("MSFT, Benchmark", period="weekly")

    assert list(result.index) == ["AAPL"]
    assert result.loc["AAPL", ["MSFT", "Benchmark"]].sum() == pytest.approx(1, abs=1e-3)

    rolling = performance_module.get_style_analysis(
        ["MSFT", "Benchmark"], period="weekly", window=52
    )
    assert ("AAPL", "R Squared") in rolling.columns
