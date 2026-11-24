"""Drawdown Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.risk import risk_model
from financetoolkit.risk.drawdown import drawdown_model

# Forty returns of a Student-t(4) draw. The reference values come from skfolio
# (get_drawdowns, max_drawdown, average_drawdown, edar) and riskfolio-lib (MDD, ADD, EDaR and
# UCI in their Rel and Abs versions), signs flipped to the Toolkit's convention.
REFERENCE_RETURNS = pd.Series(
    [0.0004, -0.0137, 0.0009, -0.0064, 0.0025, 0.0177, -0.0455, -0.0052, 0.0026, -0.0068,
     -0.0241, -0.007, 0.0001, -0.001, -0.0152, -0.0147, -0.0041, -0.0149, -0.0015, 0.0059,
     0.0036, -0.0195, 0.0084, -0.0079, -0.0059, 0.0107, -0.0016, -0.0031, -0.0039, -0.0035,
     0.0002, 0.0143, -0.0145, -0.0299, 0.0015, -0.0072, -0.0021, 0.0113, 0.0011, -0.0077]
)  # fmt: skip


@pytest.mark.parametrize(
    ("compounded", "maximum", "average", "entropic", "quantile", "ulcer"),
    [
        (True, -0.17687366647667402, -0.10716854413708654, -0.17634136922818716, -0.1731768147041965, 0.1198614354767762),
        (False, -0.19180000000000003, -0.11342, -0.19115150120124205, -0.18723000000000004, 0.1275076331048459),
    ],
)  # fmt: skip
def test_drawdown_measures_match_references(
    compounded, maximum, average, entropic, quantile, ulcer
):
    returns = REFERENCE_RETURNS

    assert np.isclose(
        drawdown_model.get_maximum_drawdown(returns, compounded), maximum, atol=1e-12
    )
    assert np.isclose(
        drawdown_model.get_average_drawdown(returns, "observation", compounded),
        average,
        atol=1e-12,
    )
    assert np.isclose(
        drawdown_model.get_entropic_drawdown_at_risk(returns, 0.05, compounded),
        entropic,
        atol=1e-12,
    )
    assert np.isclose(
        drawdown_model.get_drawdown_at_risk(returns, 0.05, compounded),
        quantile,
        atol=1e-12,
    )
    assert np.isclose(
        risk_model.get_ui(returns, None, compound_returns=compounded), ulcer, atol=1e-12
    )


def test_drawdown_measures_are_ordered():
    # DaR >= CDaR >= EDaR >= Maximum Drawdown, all negative.
    drawdown_at_risk = drawdown_model.get_drawdown_at_risk(REFERENCE_RETURNS)
    conditional = risk_model.get_conditional_drawdown_at_risk(REFERENCE_RETURNS, 0.05)
    entropic = drawdown_model.get_entropic_drawdown_at_risk(REFERENCE_RETURNS)
    maximum = drawdown_model.get_maximum_drawdown(REFERENCE_RETURNS)

    assert drawdown_at_risk >= conditional >= entropic >= maximum


def test_get_drawdown_samples():
    samples = np.array([[-0.1, 0.2, -0.5]])
    compounded = drawdown_model.get_drawdown_samples(samples)
    uncompounded = drawdown_model.get_drawdown_samples(samples, compound_returns=False)

    # The first loss is a drawdown from the starting wealth of 1 (or sum of 0).
    np.testing.assert_allclose(compounded, [[-0.1, 0, -0.5]])
    np.testing.assert_allclose(uncompounded, [[-0.1, 0, -0.5]])
    np.testing.assert_allclose(
        drawdown_model.get_drawdown_samples(
            np.array([[0.1, -0.1]]), compound_returns=False
        ),
        [[0, -0.1]],
    )


def test_get_average_drawdown_episode():
    # Two episodes, troughs -0.1 and -0.2 (compounded: 1.1 * 0.8 = 0.88 is 20% below 1.1).
    returns = pd.Series([-0.1, 0.2, 0.0, -0.2, -0.0])
    assert np.isclose(drawdown_model.get_average_drawdown(returns), -0.15)
    assert np.isclose(
        drawdown_model.get_average_drawdown(returns, "observation"),
        (-0.1 - 0.2 - 0.2) / 5,
    )
    assert drawdown_model.get_average_drawdown(pd.Series([0.01, 0.02])) == 0
    np.testing.assert_allclose(
        drawdown_model.get_drawdown_troughs(np.array([0, -0.1, -0.3, 0, -0.2, -0.1])),
        [-0.3, -0.2],
    )

    with pytest.raises(ValueError):
        drawdown_model.get_average_drawdown(returns, "peak")


def test_drawdown_measures_per_period_and_rolling(recorder):
    dates = pd.period_range("2024-01-01", periods=40, freq="D")
    frame = pd.DataFrame(
        {"A": REFERENCE_RETURNS.to_numpy(), "B": REFERENCE_RETURNS[::-1].to_numpy()},
        index=dates,
    )
    within = frame.set_axis(
        pd.MultiIndex.from_arrays([dates.asfreq("M"), dates]), axis=0
    )

    for function in [
        drawdown_model.get_maximum_drawdown,
        drawdown_model.get_drawdown_at_risk,
        drawdown_model.get_average_drawdown,
        drawdown_model.get_entropic_drawdown_at_risk,
    ]:
        recorder.capture(function(within).round(10))
        recorder.capture(function(frame, window_size=10).round(10))


def test_get_drawdown_statistics():
    # Wealth: 0.9, 1.08 (a new high), 0.972, 1.1664 (a new high), 0.93312 (not recovered).
    returns = pd.DataFrame(
        {"A": [np.nan, -0.1, 0.2, -0.1, 0.2, -0.2], "B": np.nan},
        index=pd.period_range("2024-01-01", periods=6, freq="D"),
    )
    statistics = drawdown_model.get_drawdown_statistics(returns)
    row = statistics.loc["A"]

    assert row["Drawdowns"] == 3
    assert np.isclose(row["Average Drawdown"], (-0.1 - 0.1 - 0.2) / 3)
    assert np.isclose(row["Drawdown Deviation"], np.sqrt((0.01 + 0.01 + 0.04) / 5))
    # To Trough is 1 for each; the two recovered drawdowns take 2 periods from peak to recovery.
    assert row["Average To Trough"] == 1
    assert row["Average Length"] == 2
    assert row["Average To Recovery"] == 1
    assert statistics.loc["B"].isna().all()


def test_a_path_back_at_exactly_its_old_high_ends_the_episode():
    # Closes of 100, 99, 95, 100, 98 and 96 are two drawdowns of -5% and -4%, though the
    # compounded wealth at the second 100 is 0.9999999999999999.
    closes = np.array([100.0, 99.0, 95.0, 100.0, 98.0, 96.0])
    samples = (closes[1:] / closes[:-1] - 1)[None, :]

    drawdowns = drawdown_model.get_drawdown_samples(samples)

    assert drawdowns[0, 2] == 0
    np.testing.assert_allclose(
        drawdown_model.get_drawdown_troughs(drawdowns[0]), [-0.05, -0.04]
    )
    assert drawdown_model.get_average_drawdown(
        pd.Series(samples[0]), method="episode"
    ) == pytest.approx(-0.045)


def test_get_conditional_expected_drawdown_averages_the_worst_path_drawdowns():
    # Goldberg and Mahmoud (2017): the Maximum Drawdown of every path of `window` returns, each
    # from a fresh wealth of 1, and the exact tail average of the deepest alpha share of them
    # (36 paths at alpha 0.2: the 7 deepest and 0.2 of the 8th).
    window, alpha = 5, 0.2
    values = REFERENCE_RETURNS.to_numpy()
    maximum_drawdowns = []

    for start in range(len(values) - window + 1):
        wealth = np.cumprod(1 + values[start : start + window])
        peak = np.maximum(np.maximum.accumulate(wealth), 1)
        maximum_drawdowns.append((wealth / peak - 1).min())

    deepest = np.sort(maximum_drawdowns)
    expected = (deepest[:7].sum() + 0.2 * deepest[7]) / 7.2

    assert drawdown_model.get_conditional_expected_drawdown(
        REFERENCE_RETURNS, window, alpha
    ) == pytest.approx(expected, abs=1e-15)

    # A path as long as the data leaves one path, its Maximum Drawdown.
    assert drawdown_model.get_conditional_expected_drawdown(
        REFERENCE_RETURNS, len(REFERENCE_RETURNS)
    ) == pytest.approx(drawdown_model.get_maximum_drawdown(REFERENCE_RETURNS))


def test_get_conditional_expected_drawdown_rolling_and_per_period():
    returns = pd.DataFrame(
        {"A": REFERENCE_RETURNS.to_numpy(), "B": REFERENCE_RETURNS.to_numpy()[::-1]},
        index=pd.date_range("2024-01-01", periods=40, freq="D"),
    )
    rolling = drawdown_model.get_conditional_expected_drawdown(
        returns, window=5, window_size=20
    )

    assert rolling.iloc[:19].isna().all().all()
    assert rolling.iloc[-1]["B"] == pytest.approx(
        drawdown_model.get_conditional_expected_drawdown(returns["B"].iloc[-20:], 5)
    )

    nested = returns.set_index(
        [returns.index.to_period("M"), returns.index], append=False
    )
    per_period = drawdown_model.get_conditional_expected_drawdown(nested, window=5)
    assert per_period.loc[pd.Period("2024-02", "M"), "A"] == pytest.approx(
        drawdown_model.get_conditional_expected_drawdown(returns["A"].iloc[31:], 5)
    )

    with pytest.raises(ValueError, match="rolling window"):
        drawdown_model.get_conditional_expected_drawdown(returns, 5, window_size=3)


def test_get_drawdown_state_and_recovery_measures():
    # Wealth: 0.9, 1.08 (a new high), 0.972, 1.1664 (a new high), 0.93312, 1.026432 (open).
    returns = pd.DataFrame(
        {
            "A": [np.nan, -0.1, 0.2, -0.1, 0.2, -0.2, 0.1],
            "C": [np.nan, 0.1] * 3 + [0.1],
        },
        index=pd.period_range("2024-01-01", periods=7, freq="D"),
    )
    statistics = drawdown_model.get_drawdown_statistics(returns)
    row = statistics.loc["A"]

    # Below the peak on four of six returns.
    assert row["Coverage"] == pytest.approx(4 / 6)
    assert row["Active Drawdown"] == pytest.approx(1.026432 / 1.1664 - 1)
    assert row["Active Duration"] == 2
    assert row["Active Recovery Duration"] == 1
    assert row["Active Recovery Return"] == pytest.approx(0.1)
    assert row["Active Recovery"] == pytest.approx(
        1 - (1.026432 / 1.1664 - 1) / (0.93312 / 1.1664 - 1)
    )
    # Two recovered drawdowns, one period down and one back, each recovering to above the peak:
    # 0.9 -> 1.08 and 0.972 -> 1.1664, a return of 20% from the trough.
    assert row["Average Recovery Ratio"] == 1
    assert row["Average Recovery Return"] == pytest.approx(0.2)

    # Always at a new high, with dates without a return: no drawdown, nothing open.
    never = statistics.loc["C"]
    assert never["Coverage"] == 0
    assert never["Active Drawdown"] == 0
    assert never["Active Duration"] == 0
    assert np.isnan(never["Active Recovery"])

    table = risk_model.get_drawdown_table(returns[["A"]], top=3)
    # The deepest drawdown (-20%) is the open one.
    assert np.isnan(table.loc[("A", 1), "Recovery Ratio"])
    assert table["Recovery Ratio"].dropna().tolist() == [1.0, 1.0]
    assert table["Recovery Return"].dropna().to_numpy() == pytest.approx([0.2, 0.2])
