"""Style Analysis and Regional Factor Tests"""

import copy
import io
import itertools
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


def test_factor_urls_use_the_published_spelling():
    # Only Asia Pacific ex Japan publishes its momentum file as "MOM".
    assert performance_model.factor_url("Asia Pacific ex Japan", "momentum").endswith(
        "/Asia_Pacific_ex_Japan_MOM_Factor_Daily_CSV.zip"
    )
    assert performance_model.factor_url("Japan", "three").endswith(
        "/Japan_3_Factors_Daily_CSV.zip"
    )
    assert performance_model.factor_url("United States", "momentum").endswith(
        "/F-F_Momentum_Factor_daily_CSV.zip"
    )


def exact_style_weights(fund: np.ndarray, styles: np.ndarray) -> np.ndarray:
    """The exact style weights: the best of the equality-constrained solutions on every
    set of styles with a positive weight (the problem is a convex quadratic programme, so the
    best feasible candidate is the optimum)."""
    covariance = np.cov(np.column_stack([fund, styles]), rowvar=False, ddof=1)
    style_covariance, fund_covariance = covariance[1:, 1:], covariance[1:, 0]
    count = styles.shape[1]
    best_objective, best_weights = np.inf, None

    for size in range(1, count + 1):
        for subset in itertools.combinations(range(count), size):
            chosen = list(subset)
            system = np.zeros((size + 1, size + 1))
            system[:size, :size] = 2 * style_covariance[np.ix_(chosen, chosen)]
            system[:size, size] = system[size, :size] = 1
            solution = np.linalg.solve(
                system, np.append(2 * fund_covariance[chosen], 1)
            )
            weights = np.zeros(count)
            weights[chosen] = solution[:size]
            if (weights < 0).any():
                continue
            objective = (
                weights @ style_covariance @ weights - 2 * weights @ fund_covariance
            )
            if objective < best_objective:
                best_objective, best_weights = objective, weights

    return best_weights


@pytest.mark.parametrize(
    "true_weights",
    [
        # Every weight positive: the bounds are inactive.
        [0.5, 0.3, 0.2],
        # A style the fund is short of: its weight sits at the lower bound of zero.
        [0.9, -0.2, 0.3],
    ],
)
def test_style_weights_match_the_exact_solution_on_daily_returns(true_weights):
    rng = np.random.default_rng(7)
    market = rng.normal(0.0004, 0.01, (504, 1))
    styles = pd.DataFrame(
        market + rng.normal(0, 0.005, (504, 3)), columns=["Growth", "Value", "Small"]
    )
    fund = styles @ np.array(true_weights) + rng.normal(0, 0.002, 504)

    result = performance_model.get_style_analysis(fund, styles, periods_per_year=252)

    assert result[["Growth", "Value", "Small"]].to_numpy() == pytest.approx(
        exact_style_weights(fund.to_numpy(), styles.to_numpy()), abs=1e-6
    )


def test_style_analysis_without_enough_observations_is_nan():
    rng = np.random.default_rng(1)
    styles = pd.DataFrame(rng.normal(0, 0.01, (3, 3)), columns=["A", "B", "C"])

    result = performance_model.get_style_analysis(styles["A"], styles)

    assert list(result.index) == [
        "A",
        "B",
        "C",
        "R Squared",
        "Selection Return",
        "Tracking Error",
    ]
    assert result.isna().all()


def test_rolling_style_analysis_measures_selection_out_of_sample():
    rng = np.random.default_rng(3)
    index = pd.period_range("2010-01", periods=90, freq="M")
    styles = pd.DataFrame(
        rng.normal(0.006, 0.04, (90, 3)), index=index, columns=["A", "B", "C"]
    )
    fund = 0.7 * styles["A"] + 0.3 * styles["C"] + rng.normal(0.001, 0.005, 90)
    # The fund lists late: its first windows have too few returns to estimate.
    fund.iloc[:40] = np.nan

    rolling = performance_model.get_rolling_style_analysis(fund, styles, window=36)

    assert list(rolling.index) == list(index[35:])
    # A window with no more returns than styles is NaN, the next one is estimated.
    assert rolling.loc[index[42]].isna().all()
    assert rolling.loc[index[43], ["A", "B", "C"]].notna().all()
    # The weights of a window are those of its own 36 months.
    window = slice(index[89 - 35], index[89])
    assert rolling.loc[index[89]].iloc[:6].to_numpy() == pytest.approx(
        performance_model.get_style_analysis(
            fund.loc[window], styles.loc[window]
        ).to_numpy()
    )
    # The out-of-sample selection return of a month uses the weights of the window that
    # ends a month earlier, and the first estimate has no earlier one.
    previous_weights = rolling.loc[index[59], ["A", "B", "C"]].to_numpy()
    assert rolling.loc[index[60], "Out-of-Sample Selection Return"] == pytest.approx(
        fund.loc[index[60]] - styles.loc[index[60]].to_numpy() @ previous_weights
    )
    assert np.isnan(rolling.loc[index[43], "Out-of-Sample Selection Return"])
    assert rolling.loc[index[44] :, "Out-of-Sample Selection Return"].notna().all()


def test_style_analysis_keeps_the_funds_with_enough_data(performance_module):
    module = copy.copy(performance_module)
    weekly = module._historical_data["weekly"].copy()
    # MSFT lists a week before the end: one return is too few for even one style.
    weekly.loc[weekly.index[:-1], ("Return", "MSFT")] = np.nan
    module._historical_data = {**module._historical_data, "weekly": weekly}

    result = module.get_style_analysis("Benchmark", period="weekly")

    assert result.loc["MSFT"].isna().all()
    assert result.loc["AAPL", "Benchmark"] == pytest.approx(1)

    rolling = module.get_style_analysis("Benchmark", period="weekly", window=10)
    assert rolling["MSFT"].isna().all().all()
    assert rolling[("AAPL", "R Squared")].notna().all()


def test_rolling_style_analysis_on_the_toolkit_starts_with_a_full_window(
    performance_module,
):
    returns = performance_module._historical_data["weekly"]["Return"].loc[
        "2019-12-31":"2023-01-01"
    ]

    rolling = performance_module.get_style_analysis(
        ["MSFT", "Benchmark"], period="weekly", window=52
    )

    # The first week has no return, so the first full window ends a week later.
    assert returns.iloc[0].isna().all()
    assert rolling.index[0] == returns.index[52]
    assert ("AAPL", "Out-of-Sample Selection Return") in rolling.columns
    assert np.isnan(rolling[("AAPL", "Out-of-Sample Selection Return")].iloc[0])
    assert rolling[("AAPL", "Out-of-Sample Selection Return")].iloc[1:].notna().all()
