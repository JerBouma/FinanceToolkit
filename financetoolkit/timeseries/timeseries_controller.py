"""Time Series Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.helpers import handle_portfolio
from financetoolkit.timeseries import (
    causality_model,
    cointegration_model,
    unitroot_model,
)
from financetoolkit.utilities.error_model import handle_errors

# pylint: disable=too-many-instance-attributes,too-few-public-methods


class TimeSeries:
    """
    The Time Series module is meant to apply statistical tests that operate on the
    time-series properties of price data itself — whether a series is stationary
    (Augmented Dickey-Fuller), whether two series share a long-run equilibrium
    (Engle-Granger cointegration), and whether one series has predictive power over
    another (Granger causality). These are foundational tools for pairs trading,
    spread modeling and lead-lag analysis, distinct from the accounting-based
    valuation models in the Models module or the risk/return metrics elsewhere.
    """

    def __init__(
        self,
        tickers: str | list[str],
        historical_data: dict[str, pd.DataFrame],
        quarterly: bool = False,
        rounding: int | None = 4,
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        """
        Initializes the Time Series Controller Class.

        Args:
            tickers (str | list[str]): The tickers to use for the calculations.
            historical_data (dict[str, pd.DataFrame]): The historical data containing all periods.
            quarterly (bool, optional): Whether to use quarterly data. Defaults to False.
            rounding (int, optional): The number of decimals to round the results to. Defaults to 4.
            start_date (str, optional): The start date to use for the calculations. Defaults to None.
            end_date (str, optional): The end date to use for the calculations. Defaults to None.
        """
        self._tickers = tickers
        self._historical_data = historical_data
        self._quarterly = quarterly
        self._rounding = rounding
        self._start_date = start_date
        self._end_date = end_date
        self._portfolio_weights: dict | None = None

    def _get_column(self, period: str, column: str) -> pd.DataFrame:
        if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            raise ValueError(
                "Period must be daily, weekly, monthly, quarterly, or yearly."
            )
        if period == "daily" and self._historical_data["intraday"].empty:
            raise ValueError("Intraday data is required for daily calculations.")

        return self._historical_data[period][column].dropna()

    @handle_portfolio
    @handle_errors
    def get_augmented_dickey_fuller(
        self,
        period: str | None = None,
        column: str = "Adj Close",
        max_lag: int | None = None,
        regression: str = "c",
        rounding: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate the Augmented Dickey-Fuller (ADF) test for a unit root, per asset.

        The test regresses the first difference of the series on its own lagged level and
        `p` lags of its own first difference. The null hypothesis is that the series has a
        unit root (is a random walk, not mean-reverting); the alternative is that it is
        stationary. This is a standard first step before modeling a price series or spread
        with mean-reverting methods, since those methods assume stationarity.

        For more information about the method, see the following paper:

        - Dickey, D.A. and Fuller, W.A. (1979). "Distribution of the Estimators for
        Autoregressive Time Series with a Unit Root." Journal of the American Statistical
        Association, 74(366a), 427-431.

        Also known as: ADF test, unit root test, stationarity test.

        Args:
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "quarterly".
            column (str, optional): The historical data column to test. Defaults to "Adj Close".
            max_lag (int, optional): The maximum number of lagged differences to consider. Defaults to
            the Schwert (1989) rule of thumb.
            regression (str, optional): Which deterministic terms to include, one of "n" (none), "c"
            (constant) or "ct" (constant and trend). Defaults to "c".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.DataFrame: The ADF statistic, lags used, observations used, 1%/5%/10% critical values, and
            whether the unit root is rejected at the 5% level, per asset.

        Notes:
        - The method retrieves historical price data based on the specified `period` and runs the ADF
        test for each asset in the Toolkit instance.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.timeseries.get_augmented_dickey_fuller(period="quarterly")
        ```

        Which returns:

        |                       |     AMZN |     TSLA |   Benchmark |
        |:----------------------|---------:|---------:|------------:|
        | ADF Statistic         |   0.2892 |   0.1023 |      2.3037 |
        | Lags Used             |  11      |  11      |     11      |
        | Observations          |  47      |  47      |     47      |
        | Critical Value 1%     |  -3.43   |  -3.43   |     -3.43   |
        | Critical Value 5%     |  -2.86   |  -2.86   |     -2.86   |
        | Critical Value 10%    |  -2.57   |  -2.57   |     -2.57   |
        | Reject Unit Root (5%) |   0      |   0      |      0      |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        prices = self._get_column(period, column)

        result = pd.DataFrame(
            {
                ticker: unitroot_model.get_augmented_dickey_fuller(
                    prices[ticker], max_lag=max_lag, regression=regression
                )
                for ticker in prices.columns
            }
        )

        return result.round(rounding if rounding is not None else self._rounding)

    @handle_errors
    def get_engle_granger_cointegration(
        self,
        ticker_a: str,
        ticker_b: str,
        period: str | None = None,
        column: str = "Adj Close",
        max_lag: int | None = None,
        rounding: int | None = None,
    ) -> pd.Series:
        """
        Calculate the Engle-Granger test for cointegration between two assets.

        Two individually non-stationary series (e.g. two stock prices, each following a
        random walk) are cointegrated if some linear combination of them is stationary,
        i.e. they share a long-run equilibrium relationship even though each wanders on
        its own in the short run. This is the classic statistical foundation for
        pairs-trading: if two assets are cointegrated, deviations of the spread from its
        equilibrium level tend to revert, making the spread itself tradeable.

        For more information about the method, see the following paper:

        - Engle, R.F. and Granger, C.W.J. (1987). "Co-integration and Error Correction:
        Representation, Estimation, and Testing." Econometrica, 55(2), 251-276.

        Also known as: EG test, residual-based cointegration test, pairs-trading test.

        Args:
            ticker_a (str): The dependent asset.
            ticker_b (str): The independent asset.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "quarterly".
            column (str, optional): The historical data column to test. Defaults to "Adj Close".
            max_lag (int, optional): The maximum number of lagged differences to consider in the
            underlying ADF test on the residuals. Defaults to the Schwert (1989) rule of thumb.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.Series: The Engle-Granger statistic, lags used, observations used, 1%/5%/10% critical
            values, and whether cointegration is found at the 5% level.

        Notes:
        - The method retrieves historical price data based on the specified `period` for the two given
        assets and runs the Engle-Granger test between them.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.timeseries.get_engle_granger_cointegration("AMZN", "TSLA", period="quarterly")
        ```

        Which returns:

        | Metric                  |    Value |
        |:-------------------------|---------:|
        | Engle-Granger Statistic |  -1.8094 |
        | Lags Used               |  11      |
        | Observations            |  47      |
        | Critical Value 1%       |  -3.9    |
        | Critical Value 5%       |  -3.34   |
        | Critical Value 10%      |  -3.04   |
        | Cointegrated (5%)       |   0      |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        prices = self._get_column(period, column)

        result = cointegration_model.get_engle_granger_cointegration(
            prices[ticker_a], prices[ticker_b], max_lag=max_lag
        )

        return result.round(rounding if rounding is not None else self._rounding)

    @handle_errors
    def get_granger_causality(
        self,
        ticker_a: str,
        ticker_b: str,
        period: str | None = None,
        column: str = "Return",
        max_lag: int = 5,
        rounding: int | None = None,
    ) -> pd.Series:
        """
        Calculate the Granger causality test for whether `ticker_b` helps predict `ticker_a`.

        "Granger causality" is a statement about predictive power, not true causation:
        `ticker_b` is said to Granger-cause `ticker_a` if past values of `ticker_b`,
        combined with past values of `ticker_a` itself, predict `ticker_a` significantly
        better than past values of `ticker_a` alone.

        For more information about the method, see the following paper:

        - Granger, C.W.J. (1969). "Investigating Causal Relations by Econometric Models
        and Cross-Spectral Methods." Econometrica, 37(3), 424-438.

        Also known as: Granger causality test, predictive causality, lead-lag test.

        Args:
            ticker_a (str): The asset being predicted (the dependent asset).
            ticker_b (str): The asset being tested for predictive power over `ticker_a`.
            period (str, optional): The data frequency (daily, weekly, monthly, quarterly, or yearly).
            Defaults to "quarterly".
            column (str, optional): The historical data column to test. Defaults to "Return", since
            Granger causality assumes a stationary series (unlike the ADF/Engle-Granger tests, which
            operate on price levels on purpose).
            max_lag (int, optional): The number of lags of both assets to include in the regressions.
            Defaults to 5.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to
            None.

        Returns:
            pd.Series: The F-statistic, its p-value, and whether `ticker_b` is found to Granger-cause
            `ticker_a` at the 5% level.

        Notes:
        - The method retrieves historical return data based on the specified `period` for the two given
        assets and runs the Granger causality test between them.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key="FINANCIAL_MODELING_PREP_KEY")

        toolkit.timeseries.get_granger_causality("AMZN", "TSLA", period="quarterly", max_lag=2)
        ```

        Which returns:

        | Metric               |   Value |
        |:----------------------|--------:|
        | F-Statistic           |  0.2156 |
        | P-Value               |  0.8067 |
        | Granger-Causes (5%)   |  0      |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = self._get_column(period, column)

        result = causality_model.get_granger_causality(
            returns[ticker_a], returns[ticker_b], max_lag=max_lag
        )

        return result.round(rounding if rounding is not None else self._rounding)
