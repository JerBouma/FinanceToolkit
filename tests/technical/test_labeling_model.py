"""Labeling Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.technicals.labeling import labeling_model
from tests.technical.sample_prices import sample_prices


def prices_from_returns(returns: list[float]) -> pd.Series:
    """Prices starting at 100 that move by the given simple returns."""
    values = 100 * np.cumprod(np.r_[1.0, 1 + np.array(returns)])
    return pd.Series(
        values, index=pd.period_range("2024-01-01", periods=len(values), freq="D")
    )


def test_get_daily_volatility_is_the_ewm_standard_deviation_of_returns():
    prices = sample_prices()["Close"]
    returns = prices / prices.shift(1) - 1

    result = labeling_model.get_daily_volatility(prices, span=10)

    # The third value by hand: two returns with weights 1 - alpha and 1, alpha = 2 / 11.
    weights = np.array([1 - 2 / 11, 1.0])
    values = returns.iloc[1:3].to_numpy()
    mean = weights @ values / weights.sum()
    variance = (
        weights
        @ (values - mean) ** 2
        / (weights.sum() - (weights @ weights) / weights.sum())
    )
    assert result.iloc[2] == pytest.approx(np.sqrt(variance), rel=1e-12)
    pd.testing.assert_series_equal(result, returns.ewm(span=10).std())


def test_get_cusum_events_by_hand():
    # Log returns of 0.01, 0.015 (sum 0.025 > 0.02: up event and reset), -0.01, -0.005,
    # -0.006 (sum -0.021 < -0.02: down event), 0.019, 0.0 and 0.002 (sum 0.021: up event).
    log_returns = [0.01, 0.015, -0.01, -0.005, -0.006, 0.019, 0.0, 0.002]
    prices = pd.Series(100 * np.exp(np.cumsum(np.r_[0.0, log_returns])))

    events = labeling_model.get_cusum_events(prices, 0.02)

    assert np.isnan(events.iloc[0])
    assert events.iloc[1:].tolist() == [0, 1, 0, 0, -1, 0, 0, 1]


def test_get_cusum_events_with_a_threshold_per_date():
    prices = pd.Series(100 * np.exp(np.cumsum([0.0, 0.01, 0.01, 0.01])))
    # No threshold on the second date, so the sums keep accumulating until the third.
    threshold = pd.Series([np.nan, np.nan, 0.015, 0.5])

    events = labeling_model.get_cusum_events(prices, threshold)

    assert events.iloc[1:].tolist() == [0, 1, 0]


def test_get_z_score_events_by_hand():
    prices = pd.Series([10.0, 10, 10, 10, 13, 10, 10, 7])

    events = labeling_model.get_z_score_events(prices, window=4, threshold=1.4)

    # The window of 10, 10, 10, 13 has a mean of 10.75 and a standard deviation of 1.5, so
    # z = 1.5 at the fifth price; the last window of 13, 10, 10, 7 gives z = -3 / 2.449 = -1.22.
    # The first window has equal prices: no deviation, no event.
    assert events.iloc[:3].isna().all()
    assert events.iloc[3:].tolist() == [0, 1, 0, 0, 0]
    assert labeling_model.get_z_score_events(prices, 4, 1.2).iloc[3:].tolist() == [
        0,
        1,
        0,
        0,
        -1,
    ]


def test_get_triple_barrier_labels_by_hand():
    # A target of 2%: the first date reaches the upper barrier on its second step (+1%, then
    # +2.5% in total), the second date the lower barrier, and the vertical barrier of three
    # periods ends the third date.
    prices = prices_from_returns([0.01, 0.015, -0.05, 0.005, 0.005, 0.001, 0.0])
    target = pd.Series(0.02, index=prices.index)

    labels = labeling_model.get_triple_barrier_labels(prices, target, horizon=3)

    assert labels["Label"].iloc[:4].tolist() == [1, -1, -1, 1]
    assert labels["Barrier"].iloc[:4].tolist() == [1, -1, -1, 0]
    assert labels["Holding Period"].iloc[:4].tolist() == [2, 2, 1, 3]
    assert labels["Return"].iloc[0] == pytest.approx(1.01 * 1.015 - 1)
    assert labels["Return"].iloc[3] == pytest.approx(
        prices.iloc[6] / prices.iloc[3] - 1
    )
    # The vertical barrier of the last three dates lies beyond the data and no barrier is
    # reached before it, so their outcome is not known yet.
    assert labels["Label"].iloc[-3:].isna().all()
    assert labels["Target"].iloc[0] == 0.02


def test_get_triple_barrier_labels_options():
    prices = prices_from_returns([0.01, 0.015, -0.05, 0.005, 0.005, 0.001, 0.0])
    target = pd.Series(0.02, index=prices.index)

    vertical_zero = labeling_model.get_triple_barrier_labels(
        prices, target, horizon=3, vertical_barrier_zero=True
    )
    assert vertical_zero["Label"].iloc[3] == 0

    # Without a stop-loss the second date runs to its vertical barrier.
    no_stop = labeling_model.get_triple_barrier_labels(
        prices, target, horizon=3, stop_loss=0
    )
    assert no_stop["Barrier"].iloc[1] == 0
    assert no_stop["Holding Period"].iloc[1] == 3

    events = pd.Series([0, 1, 0, 0, 0, 0, 0, 0], index=prices.index)
    only_events = labeling_model.get_triple_barrier_labels(
        prices, target, horizon=3, events=events
    )
    assert only_events["Label"].notna().tolist() == [False, True] + [False] * 6

    # A target at or below the minimum is not labeled.
    assert (
        labeling_model.get_triple_barrier_labels(
            prices, target, horizon=3, minimum_target=0.02
        )["Label"]
        .isna()
        .all()
    )


def test_get_triple_barrier_meta_labels():
    prices = prices_from_returns([0.01, 0.015, -0.05, 0.005, 0.005, 0.001, 0.0])
    target = pd.Series(0.02, index=prices.index)
    side = pd.Series([1, -1, 0, -1, np.nan, 1, 1, 1], index=prices.index)

    labels = labeling_model.get_triple_barrier_labels(
        prices, target, horizon=3, side=side
    )

    # The long bet makes 2.5%; the short bet of the second date gains on the fall (the
    # profit-taking barrier of a short), the fourth date's short loses on the rise.
    assert labels["Label"].iloc[[0, 1, 3]].tolist() == [1, 1, 0]
    assert labels["Barrier"].iloc[1] == 1
    assert labels["Return"].iloc[1] == pytest.approx(
        -(prices.iloc[3] / prices.iloc[1] - 1)
    )
    assert labels["Side"].iloc[[0, 1, 3]].tolist() == [1, -1, -1]
    # No bet, no label.
    assert labels["Label"].iloc[[2, 4]].isna().all()


def test_get_fixed_horizon_labels_by_hand():
    forward = pd.DataFrame(
        {"A": [0.02, -0.005, np.nan], "B": [0.0, -0.03, 0.01], "C": [0.01, 0.01, 0.0]}
    )

    absolute = labeling_model.get_fixed_horizon_labels(forward, threshold=0.01)
    assert absolute["Label"]["A"].iloc[:2].tolist() == [1, 0]
    assert absolute["Label"]["B"].tolist() == [0, -1, 0]
    assert np.isnan(absolute["Label"]["A"].iloc[2])

    mean = labeling_model.get_fixed_horizon_labels(forward, relative_to="mean")
    assert mean["Return"]["A"].iloc[0] == pytest.approx(0.02 - 0.01)
    # The third mean is over B and C only, the tickers with a return.
    assert mean["Return"]["B"].iloc[2] == pytest.approx(0.005)

    median = labeling_model.get_fixed_horizon_labels(forward, relative_to="median")
    assert median["Label"].iloc[0].tolist() == [1, -1, 0]

    benchmark = labeling_model.get_fixed_horizon_labels(
        forward, relative_to="benchmark", benchmark_returns=pd.Series([0.01, 0.0, 0.0])
    )
    assert benchmark["Return"]["B"].tolist() == pytest.approx([-0.01, -0.03, 0.01])

    scaled = labeling_model.get_fixed_horizon_labels(
        forward,
        threshold=pd.DataFrame(0.015, index=forward.index, columns=forward.columns),
    )
    assert scaled["Label"]["A"].iloc[0] == 1
    assert scaled["Label"]["C"].iloc[0] == 0


def test_get_fixed_horizon_labels_needs_benchmark_returns():
    with pytest.raises(ValueError, match="benchmark"):
        labeling_model.get_fixed_horizon_labels(
            pd.DataFrame({"A": [0.01]}), relative_to="benchmark"
        )


def test_get_forward_returns():
    prices = prices_from_returns([0.01, 0.02, 0.03])

    result = labeling_model.get_forward_returns(prices, 2)

    assert result.iloc[0] == pytest.approx(1.01 * 1.02 - 1)
    assert result.iloc[-2:].isna().all()


def test_get_trend_scanning_labels_match_least_squares():
    prices = sample_prices()["Close"]

    labels = labeling_model.get_trend_scanning_labels(prices, 5, 12)

    # The t-value of the slope of every window by least squares, for one date.
    position = 7
    best = None
    for window in range(5, 13):
        y = prices.iloc[position : position + window].to_numpy()
        x = np.column_stack([np.ones(window), np.arange(window)])
        beta, residuals, *_ = np.linalg.lstsq(x, y, rcond=None)
        covariance = residuals[0] / (window - 2) * np.linalg.inv(x.T @ x)
        t_value = beta[1] / np.sqrt(covariance[1, 1])
        if best is None or abs(t_value) > abs(best[1]):
            best = (window, t_value)
    assert labels["t-Value"].iloc[position] == pytest.approx(best[1], rel=1e-10)
    assert labels["Holding Period"].iloc[position] == best[0] - 1
    assert labels["Label"].iloc[position] == np.sign(best[1])
    assert labels["Return"].iloc[position] == pytest.approx(
        prices.iloc[position + best[0] - 1] / prices.iloc[position] - 1
    )
    # The last dates lack a complete longest window.
    assert labels["Label"].iloc[-11:].isna().all()
    assert labels["Label"].iloc[:-11].notna().all()


def test_get_trend_scanning_labels_rejects_short_windows():
    with pytest.raises(ValueError, match="minimum_window"):
        labeling_model.get_trend_scanning_labels(sample_prices()["Close"], 2, 10)


def test_get_tail_set_labels_by_hand():
    scores = pd.DataFrame(
        [[0.5, -1.0, 2.0, 0.1, np.nan], [1.0, np.nan, np.nan, np.nan, np.nan]],
        columns=list("ABCDE"),
    )

    labels = labeling_model.get_tail_set_labels(scores, quantile=0.25)

    # Four scores: percentile ranks 2/3, 0, 1 and 1/3, so only the extremes are in the tails.
    assert labels.iloc[0, :4].tolist() == [0, -1, 1, 0]
    assert np.isnan(labels.iloc[0, 4])
    # A single score has no ranking.
    assert labels.iloc[1].isna().all()

    # Eleven tickers with a tenth in each tail: the boundary ranks (0.1 and 0.9) are inside.
    eleven = pd.DataFrame([np.arange(11.0)])
    assert labeling_model.get_tail_set_labels(eleven, 0.1).iloc[0].tolist() == [
        -1,
        -1,
    ] + [0] * 7 + [1, 1]


def test_get_matrix_flag_labels_by_hand():
    # A window of ten prices puts one price in each part: the first part's price is the lowest
    # (bottom decile) and the rest rise, so each part fills one decile, from the bottom row on
    # the left to the top row on the right.
    prices = pd.Series(np.arange(1.0, 11.0))
    template = np.arange(100.0).reshape(10, 10)

    flags = labeling_model.get_matrix_flag_labels(
        prices, window=10, template=template, threshold=50
    )

    expected = sum(template[9 - part, part] for part in range(10))
    assert flags["Score"].iloc[-1] == pytest.approx(expected)
    assert flags["Score"].iloc[:-1].isna().all()
    assert flags["Label"].iloc[-1] == 1


def test_get_matrix_flag_labels_templates():
    prices = sample_prices()["Close"]

    bull = labeling_model.get_matrix_flag_labels(prices, 20, "leigh_bull")
    bear = labeling_model.get_matrix_flag_labels(prices, 20, "leigh_bear")

    assert "Label" not in bull
    assert bull["Score"].notna().sum() == len(prices) - 19
    assert not np.allclose(bull["Score"].dropna(), bear["Score"].dropna())
    with pytest.raises(ValueError, match="template"):
        labeling_model.get_matrix_flag_labels(prices, 20, "unknown")
    with pytest.raises(ValueError, match="window"):
        labeling_model.get_matrix_flag_labels(prices, 5)


def test_concurrency_uniqueness_and_weights_by_hand():
    # Labels at positions 0 (to 2), 1 (to 2) and 3 (to 4).
    holding_periods = pd.Series([2, 1, np.nan, 1, np.nan])
    prices = pd.Series(100 * np.exp(np.cumsum([0.0, 0.02, -0.01, 0.03, 0.01])))

    concurrency = labeling_model.get_label_concurrency(holding_periods)
    assert concurrency.tolist() == [1, 2, 2, 1, 1]

    uniqueness = labeling_model.get_average_uniqueness(holding_periods)
    assert uniqueness.iloc[0] == pytest.approx((1 + 0.5 + 0.5) / 3)
    assert uniqueness.iloc[1] == pytest.approx(0.5)
    assert uniqueness.iloc[3] == pytest.approx(1.0)
    assert np.isnan(uniqueness.iloc[2])

    weights = labeling_model.get_return_attribution_weights(prices, holding_periods)
    raw = np.array([abs(0.02 / 2 - 0.01 / 2), abs(-0.01 / 2), abs(0.01)])
    assert weights.iloc[[0, 1, 3]].to_numpy() == pytest.approx(raw * 3 / raw.sum())
    assert weights.sum() == pytest.approx(3)

    decayed = labeling_model.get_time_decay_weights(uniqueness, decay=0.5)
    cumulative = np.cumsum([2 / 3, 0.5, 1.0])
    slope = 0.5 / cumulative[-1]
    assert decayed.iloc[[0, 1, 3]].to_numpy() == pytest.approx(
        1 - slope * cumulative[-1] + slope * cumulative
    )
    assert decayed.iloc[3] == pytest.approx(1)

    # A negative decay gives the oldest share of the uniqueness no weight.
    negative = labeling_model.get_time_decay_weights(uniqueness, decay=-0.5)
    assert negative.iloc[0] == 0
    assert negative.iloc[3] == pytest.approx(1)
    assert (labeling_model.get_time_decay_weights(uniqueness, 1).dropna() == 1).all()
    with pytest.raises(ValueError, match="decay"):
        labeling_model.get_time_decay_weights(uniqueness, decay=-1)


CUT_POINTS = [25, 40, 55]


@pytest.mark.parametrize("cut", CUT_POINTS)
def test_backward_looking_measures_have_no_look_ahead(cut):
    prices = sample_prices()["Close"]
    truncated = prices.iloc[:cut]

    for function in (
        lambda values: labeling_model.get_daily_volatility(values, 10),
        lambda values: labeling_model.get_cusum_events(values, 0.01),
        lambda values: labeling_model.get_cusum_events(
            values, labeling_model.get_daily_volatility(values)
        ),
        lambda values: labeling_model.get_z_score_events(values, 10, 1.5),
        lambda values: labeling_model.get_matrix_flag_labels(values, 20, "leigh_bull")[
            "Score"
        ],
        # The target of the barriers is known at the date of the label.
        lambda values: labeling_model.get_triple_barrier_labels(
            values, labeling_model.get_daily_volatility(values), 3
        )["Target"].iloc[:-3],
    ):
        full = function(prices).iloc[: len(function(truncated))]
        pd.testing.assert_series_equal(full, function(truncated), check_names=False)


@pytest.mark.parametrize("cut", CUT_POINTS)
def test_known_labels_never_change_with_later_prices(cut):
    # A label is final once known: later prices only add labels.
    prices = sample_prices()["Close"]
    target = labeling_model.get_daily_volatility(prices)

    full = labeling_model.get_triple_barrier_labels(prices, target, 5)
    truncated = labeling_model.get_triple_barrier_labels(
        prices.iloc[:cut], target.iloc[:cut], 5
    )
    known = truncated["Label"].notna()
    pd.testing.assert_frame_equal(full.iloc[:cut][known], truncated[known])

    full = labeling_model.get_trend_scanning_labels(prices, 5, 10)
    truncated = labeling_model.get_trend_scanning_labels(prices.iloc[:cut], 5, 10)
    known = truncated["Label"].notna()
    pd.testing.assert_frame_equal(full.iloc[:cut][known], truncated[known])


def test_get_labels_recorded(recorder):
    prices = sample_prices()["Close"]
    target = labeling_model.get_daily_volatility(prices, 20)

    recorder.capture(
        labeling_model.get_triple_barrier_labels(prices, target, 5, 1.5, 1.0).round(6)
    )
    recorder.capture(labeling_model.get_trend_scanning_labels(prices, 5, 15).round(6))
    recorder.capture(
        labeling_model.get_matrix_flag_labels(
            prices, 23, "cervello_royo_bull", 0
        ).round(6)
    )
