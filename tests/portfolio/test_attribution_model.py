"""Portfolio Attribution Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.portfolio.construction.attribution import attribution_model
from financetoolkit.portfolio.construction.optimization import estimation_model


def test_brinson_effects_add_up_to_the_excess_return():
    result = attribution_model.get_brinson_fachler_attribution(
        portfolio_weights=pd.Series({"Tech": 0.6, "Health": 0.4}),
        portfolio_returns=pd.Series({"Tech": 0.12, "Health": 0.03}),
        benchmark_weights=pd.Series({"Tech": 0.5, "Health": 0.3, "Energy": 0.2}),
        benchmark_returns=pd.Series({"Tech": 0.10, "Health": 0.05, "Energy": -0.02}),
    )
    total = result.loc["Total"]

    assert total["Total Effect"] == pytest.approx(
        total["Portfolio Return"] - total["Benchmark Return"]
    )
    # Not holding Energy, which lagged, is a pure allocation effect.
    assert result.loc["Energy", "Selection Effect"] == 0
    assert result.loc["Energy", "Allocation Effect"] == pytest.approx(
        -0.2 * (-0.02 - total["Benchmark Return"])
    )


def test_money_weighted_return():
    one_year = pd.Series(
        [-100, 110], index=pd.to_datetime(["2020-01-01", "2020-12-31"])
    )
    assert attribution_model.get_money_weighted_return(one_year) == pytest.approx(0.1)

    with pytest.raises(ValueError, match="money invested and money received"):
        attribution_model.get_money_weighted_return(one_year.abs())


def test_stress_test_scales_up_the_assets_with_data():
    prices = pd.DataFrame(
        {"A": [100, 80, 90], "B": [np.nan, 50, 55]},
        index=pd.period_range("2020-02-18", periods=3, freq="D"),
    )
    returns = attribution_model.get_scenario_returns(
        prices, {"S": ("2020-02-18", "2020-02-20"), "T": ("2020-02-19", "2020-02-20")}
    )
    stress = attribution_model.get_stress_test(returns, pd.Series({"A": 0.5, "B": 0.5}))

    assert stress.loc["S", "Coverage"] == 0.5
    assert stress.loc["S", "Portfolio Return"] == pytest.approx(-0.1)
    assert stress.loc["T", "Portfolio Return"] == pytest.approx(0.1125)


def test_look_through_exposure():
    sectors = attribution_model.get_look_through_exposure(
        pd.Series({"QQQ": 0.5, "AAPL": 0.5}),
        {
            "QQQ": pd.Series({"Technology": 0.6, "Health": 0.4}),
            "AAPL": pd.Series({"Technology": 1.0}),
        },
    )
    assert sectors["Technology"] == pytest.approx(0.8)

    holdings = attribution_model.get_look_through_exposure(
        pd.Series({"QQQ": 0.5, "AAPL": 0.5}),
        {
            "QQQ": pd.Series({"NVDA": 0.1, "AAPL": 0.1}),
            "AAPL": pd.Series({"AAPL": 1.0}),
        },
    )
    assert holdings["AAPL"] == pytest.approx(0.55)
    assert holdings["Other"] == pytest.approx(0.4)


def test_money_weighted_return_of_a_short_holding_and_without_a_root():
    # Ten times the money in two weeks annualizes far beyond the old upper bound of 100.
    short = attribution_model.get_money_weighted_return(
        pd.Series([-100.0, 1000.0], index=["2024-01-01", "2024-01-15"])
    )
    assert short > 1e20

    with pytest.raises(ValueError, match="No annual rate"):
        attribution_model.get_money_weighted_return(
            pd.Series(
                [-100.0, 100.0, -100.0, 1.0],
                index=["2024-01-01", "2024-01-15", "2024-02-01", "2024-03-01"],
            )
        )


def test_look_through_exposure_skips_assets_without_weight():
    exposure = attribution_model.get_look_through_exposure(
        pd.Series({"AAPL": 1.0, "MSFT": 0.0}),
        {"AAPL": pd.Series({"Technology": 1.0})},
    )

    assert exposure.to_dict() == {"Technology": 1.0}


def test_money_weighted_return_with_more_than_one_solution_is_refused():
    # Invested, received and paid again: both 10.34% and 19.26% a year settle these flows.
    dates = pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"])
    flows = pd.Series([-100.0, 230.0, -132.0], index=dates)

    with pytest.raises(ValueError, match=r"2 annual rates .*10\.34%, 19\.26%"):
        attribution_model.get_money_weighted_return(flows)


def test_money_weighted_return_with_dividends_between_purchases():
    # The flows change sign many times (dividends between purchases) yet have one solution,
    # which the search over the grid finds as the direct solution does.
    dates = pd.date_range("2020-01-01", periods=9, freq="QS")
    flows = pd.Series(
        [-100.0, 1.0, -100.0, 1.0, -100.0, 1.0, -100.0, 1.0, 450.0], index=dates
    )
    years = ((dates - dates[0]).days / 365).to_numpy()

    rate = attribution_model.get_money_weighted_return(flows)

    assert np.sum(flows.to_numpy() / (1 + rate) ** years) == pytest.approx(0, abs=1e-8)
    assert 0 < rate < 1


def _two_sub_periods():
    """Two sub-periods of a portfolio and benchmark over the same three segments."""
    index = ["Q1", "Q2"]
    return (
        pd.DataFrame({"Tech": [0.6, 0.65], "Health": [0.4, 0.35]}, index=index),
        pd.DataFrame({"Tech": [0.12, -0.04], "Health": [0.03, 0.05]}, index=index),
        pd.DataFrame(
            {"Tech": [0.5, 0.52], "Health": [0.3, 0.29], "Energy": [0.2, 0.19]},
            index=index,
        ),
        pd.DataFrame(
            {"Tech": [0.10, -0.02], "Health": [0.05, 0.01], "Energy": [-0.02, 0.06]},
            index=index,
        ),
    )


@pytest.mark.parametrize("linking", ["carino", "menchero"])
def test_linked_effects_add_up_to_the_compounded_excess_return(linking):
    result = attribution_model.get_linked_brinson_fachler_attribution(
        *_two_sub_periods(), linking=linking
    )
    total = result.loc["Total"]
    effects = ["Allocation Effect", "Selection Effect", "Interaction Effect"]

    assert total["Total Effect"] == pytest.approx(
        total["Portfolio Return"] - total["Benchmark Return"], abs=1e-12
    )
    assert total[effects].sum() == pytest.approx(total["Total Effect"], abs=1e-12)
    assert result.loc["Tech", "Portfolio Weight"] == 0.6
    assert result.loc["Tech", "Portfolio Return"] == pytest.approx(1.12 * 0.96 - 1)


def test_linking_coefficients_by_hand():
    returns, benchmark = pd.Series([0.12, -0.03]), pd.Series([0.08, 0.01])
    total, total_benchmark = 1.12 * 0.97 - 1, 1.08 * 1.01 - 1

    # Carino (1999): k_t / K with logarithmic excess returns.
    k = (np.log(1 + total) - np.log(1 + total_benchmark)) / (total - total_benchmark)
    k_t = (np.log1p(returns) - np.log1p(benchmark)) / (returns - benchmark)
    carino = attribution_model.get_linking_coefficients(returns, benchmark, "carino")
    assert carino.to_numpy() == pytest.approx((k_t / k).to_numpy(), abs=1e-14)

    # Menchero (2000): M plus the smallest corrections that close the gap.
    m = ((total - total_benchmark) / 2) / (
        (1 + total) ** 0.5 - (1 + total_benchmark) ** 0.5
    )
    excess = returns - benchmark
    alpha = (total - total_benchmark - m * excess.sum()) / (excess**2).sum() * excess
    menchero = attribution_model.get_linking_coefficients(
        returns, benchmark, "menchero"
    )
    assert menchero.to_numpy() == pytest.approx((m + alpha).to_numpy(), abs=1e-14)

    for coefficients in (carino, menchero):
        assert (coefficients * excess).sum() == pytest.approx(
            total - total_benchmark, abs=1e-15
        )


def test_linking_coefficients_with_equal_returns():
    same = pd.Series([0.1, -0.05])

    # The limits: k_t = 1 / (1 + R_t), K = 1 / (1 + R) and M = (1 + R)^((T - 1) / T).
    carino = attribution_model.get_linking_coefficients(same, same, "carino")
    assert carino.to_numpy() == pytest.approx([1.045 / 1.1, 1.045 / 0.95])
    menchero = attribution_model.get_linking_coefficients(same, same, "menchero")
    assert menchero.to_numpy() == pytest.approx([1.045**0.5] * 2)

    with pytest.raises(ValueError, match="linking must be one of"):
        attribution_model.get_linking_coefficients(same, same, "frongello")


@pytest.mark.parametrize("linking", ["carino", "menchero"])
def test_one_linked_sub_period_is_the_single_period_attribution(linking):
    weights, returns, benchmark_weights, benchmark_returns = _two_sub_periods()
    linked = attribution_model.get_linked_brinson_fachler_attribution(
        weights.iloc[:1],
        returns.iloc[:1],
        benchmark_weights.iloc[:1],
        benchmark_returns.iloc[:1],
        linking=linking,
    )
    single = attribution_model.get_brinson_fachler_attribution(
        weights.iloc[0],
        returns.iloc[0],
        benchmark_weights.iloc[0],
        benchmark_returns.iloc[0],
    )

    pd.testing.assert_frame_equal(linked, single.loc[linked.index], atol=1e-14)


@pytest.mark.parametrize(
    ("returns", "benchmark"),
    [([1.0, -0.5], [0.0, 0.0]), ([0.25, -0.2, 0.1], [0.0, 0.0, 0.1])],
)
def test_menchero_linking_adds_up_when_the_compounded_returns_are_equal(
    returns, benchmark
):
    returns, benchmark = pd.Series(returns), pd.Series(benchmark)
    coefficients = attribution_model.get_linking_coefficients(
        returns, benchmark, linking="menchero"
    )

    assert (coefficients * (returns - benchmark)).sum() == pytest.approx(0, abs=1e-14)


def test_carino_linking_refuses_a_return_of_minus_one_hundred_percent():
    with pytest.raises(ValueError, match="menchero"):
        attribution_model.get_linking_coefficients(
            pd.Series([-1.0, 0.5]), pd.Series([0.1, 0.1]), linking="carino"
        )


def test_a_sector_held_short_keeps_its_own_return():
    # 130/30: Tech held long, Consumer held short. The short sector's own return enters
    # the portfolio return; the benchmark's Consumer return would give 1.3 x 0.5 - 0.3 x 0.1.
    result = attribution_model.get_brinson_fachler_attribution(
        portfolio_weights=pd.Series({"Tech": 1.3, "Consumer": -0.3}),
        portfolio_returns=pd.Series({"Tech": 0.5, "Consumer": 1.0}),
        benchmark_weights=pd.Series({"Tech": 0.6, "Consumer": 0.4}),
        benchmark_returns=pd.Series({"Tech": 0.4, "Consumer": 0.1}),
    )
    total = result.loc["Total"]

    assert total["Portfolio Return"] == pytest.approx(1.3 * 0.5 - 0.3 * 1.0)
    assert result.loc["Consumer", "Portfolio Return"] == pytest.approx(1.0)
    assert total["Total Effect"] == pytest.approx(
        total["Portfolio Return"] - total["Benchmark Return"]
    )
    assert result.loc["Consumer", "Selection Effect"] == pytest.approx(0.4 * 0.9)


@pytest.mark.parametrize("linking", ["carino", "menchero"])
def test_linked_effects_add_up_with_a_sector_held_short(linking):
    index = ["Q1", "Q2"]
    weights = pd.DataFrame({"Tech": [1.3, 1.4], "Consumer": [-0.3, -0.4]}, index=index)
    returns = pd.DataFrame({"Tech": [0.1, -0.05], "Consumer": [0.2, 0.08]}, index=index)
    benchmark_weights = pd.DataFrame(
        {"Tech": [0.6, 0.62], "Consumer": [0.4, 0.38]}, index=index
    )
    benchmark_returns = pd.DataFrame(
        {"Tech": [0.08, -0.02], "Consumer": [0.03, 0.04]}, index=index
    )

    result = attribution_model.get_linked_brinson_fachler_attribution(
        weights, returns, benchmark_weights, benchmark_returns, linking=linking
    )
    total = result.loc["Total"]
    portfolio = (1 + (weights * returns).sum(axis=1)).prod() - 1

    assert total["Portfolio Return"] == pytest.approx(portfolio)
    assert total["Total Effect"] == pytest.approx(
        total["Portfolio Return"] - total["Benchmark Return"], abs=1e-12
    )
    assert result.loc["Consumer", "Portfolio Return"] == pytest.approx(1.2 * 1.08 - 1)


def test_stress_test_coverage_is_the_share_of_the_gross_exposure():
    returns = pd.DataFrame(
        {"Long": [-0.5, -0.5, -0.2], "Short": [np.nan, 0.1, np.nan]},
        index=["Long only", "Both", "None"],
    )
    returns.loc["None", "Long"] = np.nan

    long_listed = attribution_model.get_stress_test(
        returns, pd.Series({"Long": 1.3, "Short": -0.3})
    )
    # Only the long side has prices: it stands in for the portfolio, at 1.3 of 1.6.
    assert long_listed.loc["Long only", "Coverage"] == pytest.approx(1.3 / 1.6)
    assert long_listed.loc["Long only", "Portfolio Return"] == pytest.approx(-0.5)
    assert long_listed.loc["Both", "Coverage"] == pytest.approx(1.0)
    assert long_listed.loc["Both", "Portfolio Return"] == pytest.approx(
        1.3 * -0.5 - 0.3 * 0.1
    )
    assert long_listed.loc["None", "Coverage"] == 0
    assert np.isnan(long_listed.loc["None", "Portfolio Return"])

    # Only the short side has prices: a negative net weight cannot be scaled up to the
    # portfolio, so the return is empty and the Coverage still lies between 0 and 1.
    short_listed = attribution_model.get_stress_test(
        pd.DataFrame({"Long": [np.nan], "Short": [0.1]}, index=["Short only"]),
        pd.Series({"Long": 1.3, "Short": -0.3}),
    )
    assert short_listed.loc["Short only", "Coverage"] == pytest.approx(0.3 / 1.6)
    assert np.isnan(short_listed.loc["Short only", "Portfolio Return"])


def test_a_sector_without_a_benchmark_return_keeps_its_weight():
    result = attribution_model.get_brinson_fachler_attribution(
        portfolio_weights=pd.Series({"Tech": 0.6, "Health": 0.4}),
        portfolio_returns=pd.Series({"Tech": 0.12, "Health": 0.03}),
        benchmark_weights=pd.Series({"Tech": 0.5, "Health": 0.3, "Energy": 0.2}),
        benchmark_returns=pd.Series({"Tech": 0.10, "Health": np.nan, "Energy": -0.02}),
    )

    # Its weight is not handed to the other sectors and its return is unknown, not zero.
    assert result.loc["Health", "Benchmark Weight"] == pytest.approx(0.3)
    assert result.loc["Tech", "Benchmark Weight"] == pytest.approx(0.5)
    assert np.isnan(result.loc["Health", "Benchmark Return"])
    assert result.loc["Health", "Portfolio Return"] == pytest.approx(0.03)
    assert result.loc["Total", "Portfolio Return"] == pytest.approx(
        0.6 * 0.12 + 0.4 * 0.03
    )
    # Without the benchmark's total return no allocation effect and no total is known.
    for column in [
        "Benchmark Return",
        "Allocation Effect",
        "Selection Effect",
        "Total Effect",
    ]:
        assert np.isnan(result.loc["Total", column]), column
    assert result["Allocation Effect"].isna().all()
    assert result[["Selection Effect", "Interaction Effect"]].loc["Health"].isna().all()
    # The selection within a sector with a benchmark return is still known.
    assert result.loc["Tech", "Selection Effect"] == pytest.approx(0.5 * 0.02)
    assert result.loc["Energy", "Selection Effect"] == 0


def test_a_linked_sector_without_a_benchmark_return_in_one_sub_period():
    weights, returns, benchmark_weights, benchmark_returns = _two_sub_periods()
    benchmark_returns.iloc[0, benchmark_returns.columns.get_loc("Energy")] = np.nan
    result = attribution_model.get_linked_brinson_fachler_attribution(
        weights, returns, benchmark_weights, benchmark_returns
    )

    # Energy's compounded benchmark return is unknown, the others compound as before.
    assert np.isnan(result.loc["Energy", "Benchmark Return"])
    assert result.loc["Tech", "Benchmark Return"] == pytest.approx(1.10 * 0.98 - 1)
    assert result.loc["Energy", "Benchmark Weight"] == pytest.approx(0.2)
    assert np.isnan(result.loc["Total", "Benchmark Return"])
    # Every linking coefficient depends on the compounded benchmark return.
    assert result.loc[:, "Allocation Effect":].isna().all().all()

    # The second sub-period alone has every return and is attributed in full.
    second = attribution_model.get_linked_brinson_fachler_attribution(
        weights.iloc[1:],
        returns.iloc[1:],
        benchmark_weights.iloc[1:],
        benchmark_returns.iloc[1:],
    )
    assert second.loc["Total"].notna().all()


def test_factor_attribution_matches_empyrical():
    rng = np.random.default_rng(42)
    factor_data = rng.normal(0.0004, 0.01, size=(500, 2))
    loadings = rng.uniform(0.2, 1.2, size=(2, 5))
    data = factor_data @ loadings + rng.normal(0.0002, 0.012, size=(500, 5))
    data = data - data.mean(axis=0) + np.array([0.10, 0.12, 0.14, 0.11, 0.13]) / 252
    returns = pd.DataFrame(data, columns=list("ABCDE"))
    noise = np.random.default_rng(7).normal(0.0001, 0.006, size=500)
    factors = pd.DataFrame(
        {"MKT": factor_data[:, 0], "SMB": factor_data[:, 1], "NOISE": noise}
    )
    weights = pd.DataFrame(
        np.tile([0.3, 0.2, 0.2, 0.2, 0.1], (500, 1)), columns=returns.columns
    )
    factor_loadings = estimation_model.get_factor_loadings(returns, factors).drop(
        columns="Intercept"
    )
    table = attribution_model.get_factor_attribution(
        returns, weights, factor_loadings, factors
    )
    # empyrical 0.7.3 perf_attrib with these weights and loadings, first two periods:
    # MKT, SMB, NOISE, Common, Specific and Total Return.
    assert np.allclose(
        table[
            ["MKT", "SMB", "NOISE", "Common Return", "Specific Return", "Total Return"]
        ]
        .iloc[:2]
        .to_numpy(),
        [
            [
                2.0212717366e-03,
                -6.2326103268e-03,
                3.7559700912e-06,
                -4.2075826202e-03,
                -6.6433077724e-03,
                -1.0850890393e-02,
            ],
            [
                4.6348636464e-03,
                6.1115749134e-03,
                6.619493316e-05,
                1.0812633493e-02,
                -3.2383881638e-03,
                7.5742453292e-03,
            ],
        ],
        atol=1e-12,
    )
    assert np.allclose(table["Total Return"], returns @ weights.iloc[0])
    assert np.allclose(
        table["Common Return"] + table["Specific Return"], table["Total Return"]
    )
    # Static weights have no timing.
    assert np.allclose(table["Timing Return"], 0)
    assert np.allclose(table["Tilt Return"], table["Common Return"])


def test_period_contributions_add_up_to_the_compounded_return():
    rng = np.random.default_rng(3)
    index = pd.period_range("2024-01-01", periods=90, freq="D")
    contributions = pd.DataFrame(
        rng.normal(0.0005, 0.004, size=(90, 3)), index=index, columns=list("ABC")
    )
    table = attribution_model.get_period_contributions(contributions, "M")
    portfolio = contributions.sum(axis=1)
    compounded = (1 + portfolio).groupby(index.asfreq("M")).prod() - 1
    assert np.allclose(table["Portfolio"], compounded)
    assert np.allclose(table[list("ABC")].sum(axis=1), compounded)
    # Within a month the first day's contribution is not carried by any growth.
    first = contributions.iloc[0]
    second = contributions.iloc[1] * (1 + portfolio.iloc[0])
    assert np.allclose(
        attribution_model.get_period_contributions(contributions.iloc[:2], "M").iloc[
            0, :3
        ],
        first + second,
    )
