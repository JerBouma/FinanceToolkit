"""Risk Module"""
__docformat__ = "google"

import pandas as pd

from financetoolkit.helpers import (
    calculate_growth,
    handle_errors,
)
from financetoolkit.risk import helpers, risk_model

# pylint: disable=too-many-instance-attributes,too-few-public-methods


class Risk:
    """
    The Risk module is meant to calculate important risk metrics such
    as Value at Risk (VaR), Conditional Value at Risk (cVaR), Maximum
    Drawdown, Correlations, GARCH, EWMA and more.
    """

    def __init__(
        self,
        tickers,
        daily_historical: pd.DataFrame = pd.DataFrame(),
        weekly_historical: pd.DataFrame = pd.DataFrame(),
        monthly_historical: pd.DataFrame = pd.DataFrame(),
        quarterly_historical: pd.DataFrame = pd.DataFrame(),
        yearly_historical: pd.DataFrame = pd.DataFrame(),
        quarterly: bool = False,
        rounding=4,
    ):
        """
        Initializes the Risk Controller Class.

        Args:
            tickers (str | list[str]): The tickers to use for the Toolkit instance.
            daily_historical (pd.DataFrame, optional): The daily historical data for the tickers.
            Defaults to pd.DataFrame().
            weekly_historical (pd.DataFrame, optional): The weekly historical data for the tickers.
            Defaults to pd.DataFrame().
            monthly_historical (pd.DataFrame, optional): The monthly historical data for the tickers.
            Defaults to pd.DataFrame().
            quarterly_historical (pd.DataFrame, optional): The quarterly historical data for the tickers.
            Defaults to pd.DataFrame().
            yearly_historical (pd.DataFrame, optional): The yearly historical data for the tickers.
            Defaults to pd.DataFrame().
            quarterly (bool, optional): Whether to use quarterly data. Defaults to False.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.

        As an example:

        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.risk.get_value_at_risk(period='yearly')
        ```

        Which returns:

        | Date   |    AAPL |    TSLA |
        |:-------|--------:|--------:|
        | 2012   |  0      |  0      |
        | 2013   |  0.1754 |  4.96   |
        | 2014   |  1.7515 |  0.9481 |
        | 2015   | -0.1958 |  0.1454 |
        | 2016   |  0.4177 | -0.3437 |
        | 2017   |  2.6368 |  1.2225 |
        | 2018   | -0.2786 |  0.0718 |
        | 2019   |  3.2243 |  0.4707 |
        | 2020   |  1.729  |  8.3319 |
        | 2021   |  1.3179 |  0.8797 |
        | 2022   | -0.8026 | -1.0046 |
        | 2023   |  1.8549 |  1.8238 |
        """
        if (
            daily_historical.empty
            and weekly_historical.empty
            and monthly_historical.empty
            and quarterly_historical.empty
            and yearly_historical.empty
        ):
            raise ValueError("At least one historical DataFrame is required.")

        self._tickers = tickers
        self._daily_historical = daily_historical
        self._weekly_historical = weekly_historical
        self._monthly_historical = monthly_historical
        self._quarterly_historical = quarterly_historical
        self._yearly_historical = yearly_historical
        self._quarterly = quarterly
        self._rounding = rounding

        # Return Calculations
        self._daily_returns = self._daily_historical["Return"]
        self._weekly_returns = (
            self._daily_historical["Return"]
            .groupby(pd.Grouper(freq="W"))
            .apply(lambda x: x)
        )
        self._monthly_returns = (
            self._daily_historical["Return"]
            .groupby(pd.Grouper(freq="M"))
            .apply(lambda x: x)
        )
        self._quarterly_returns = (
            self._daily_historical["Return"]
            .groupby(pd.Grouper(freq="Q"))
            .apply(lambda x: x)
        )
        self._yearly_returns = (
            self._daily_historical["Return"]
            .groupby(pd.Grouper(freq="Y"))
            .apply(lambda x: x)
        )

    @handle_errors
    def get_value_at_risk(
        self,
        period=None,
        alpha: float = 0.05,
        within_period: bool = True,
        rounding=None,
        growth: bool = False,
        lag=1,
        distribution="historic",
    ):
        """
        Calculate the Value at Risk (VaR) of an investment portfolio or asset's returns.

        Value at Risk (VaR) is a risk management metric that quantifies the maximum potential loss
        an investment portfolio or asset may experience over a specified time horizon and confidence level.
        It provides insights into the downside risk associated with an investment and helps investors make
        informed decisions about risk tolerance.

        The VaR is calculated as the quantile of the return distribution, representing the loss threshold
        that is not expected to be exceeded with a given confidence level (e.g., 5% for alpha=0.05).

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, quarterly, or yearly).
            Defaults to "yearly".
            alpha (float, optional): The confidence level for VaR calculation (e.g., 0.05 for 95% confidence).
            Defaults to 0.05.
            within_period (bool, optional): Whether to calculate VaR within the specified period or for the entire
            period. Thus whether to look at the VaR within a specific year (if period = 'yearly') or look at the entirety
            of all years. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the VaR values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            distribution (str): The distribution to use for the VaR calculations (historic, gaussian, cf
            or studentt). Defaults to "historic".

        Returns:
            pd.Series: VaR values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates VaR for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of VaR values using the specified `lag`.

        Example:
        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key=FMP_KEY)

        toolkit.risk.get_value_at_risk()
        ```

        Which returns:

        |      |    AMZN |    TSLA |
        |:-----|--------:|--------:|
        | 2012 | -0.0244 | -0.0343 |
        | 2013 | -0.0204 | -0.0537 |
        | 2014 | -0.0312 | -0.0423 |
        | 2015 | -0.0208 | -0.0422 |
        | 2016 | -0.0288 | -0.0394 |
        | 2017 | -0.0154 | -0.0345 |
        | 2018 | -0.0416 | -0.0503 |
        | 2019 | -0.0232 | -0.0492 |
        | 2020 | -0.0369 | -0.0741 |
        | 2021 | -0.0252 | -0.0499 |
        | 2022 | -0.0518 | -0.0713 |
        | 2023 | -0.0271 | -0.054  |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = helpers.handle_return_data_periods(self, period, within_period)

        if distribution == "historic":
            value_at_risk = risk_model.get_var_historic(returns, alpha)
        elif distribution == "gaussian":
            value_at_risk = risk_model.get_var_gaussian(returns, alpha)
        elif distribution == "cf":
            value_at_risk = risk_model.get_var_gaussian(returns, alpha, True)
        elif distribution == "studentt":
            value_at_risk = risk_model.get_var_studentt(returns, alpha)
        else:
            raise ValueError("Distribution must be historic, gaussian, cf or studentt.")

        if growth:
            return calculate_growth(
                value_at_risk,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return value_at_risk.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_conditional_value_at_risk(
        self,
        period=None,
        alpha: float = 0.05,
        within_period: bool = True,
        rounding=None,
        growth: bool = False,
        lag=1,
        distribution="historic",
    ):
        """
        Calculate the Conditional Value at Risk (CVaR) of an investment portfolio or asset's returns.

        Conditional Value at Risk (CVaR) is a risk management metric that quantifies the loss in the worst % of cases of
        an investment portfolio or asset may experience over a specified time horizon and confidence level.
        It provides insights into the downside risk associated with an investment and helps investors make
        informed decisions about risk tolerance.

        The CVaR is calculated as the expected loss given that the loss threshold (VaR) with a given confidence
        level (e.g., 5% for alpha=0.05) is excceeded.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, quarterly, or yearly).
            Defaults to "yearly".
            alpha (float, optional): The confidence level for CVaR calculation (e.g., 0.05 for 95% confidence).
            Defaults to 0.05.
            within_period (bool, optional): Whether to calculate CVaR within the specified period or for the entire
            period. Thus whether to look at the CVaR within a specific year (if period = 'yearly') or look at the entirety
            of all years. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the CVaR values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.
            distribution (str): The distribution to use for the CVaR calculations (historic, gaussian, studentt, laplace
            or logistic). Defaults to "historic".

        Returns:
            pd.Series: CVaR values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates CVaR for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of CVaR values using the specified `lag`.

        Example:
        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key=FMP_KEY)

        toolkit.risk.get_conditional_value_at_risk()
        ```

        Which returns:

        |      |    AMZN |    TSLA |
        |:-----|--------:|--------:|
        | 2012 | -0.0302 | -0.0622 |
        | 2013 | -0.0323 | -0.0807 |
        | 2014 | -0.0552 | -0.0607 |
        | 2015 | -0.0318 | -0.053  |
        | 2016 | -0.0456 | -0.0604 |
        | 2017 | -0.0236 | -0.0483 |
        | 2018 | -0.0540 | -0.0746 |
        | 2019 | -0.0327 | -0.0758 |
        | 2020 | -0.0510 | -0.1262 |
        | 2021 | -0.0327 | -0.0683 |
        | 2022 | -0.0685 | -0.0914 |
        | 2023 | -0.0397 | -0.0747 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = helpers.handle_return_data_periods(self, period, within_period)

        if distribution == "historic":
            conditional_value_at_risk = risk_model.get_cvar_historic(returns, alpha)
        elif distribution == "gaussian":
            conditional_value_at_risk = risk_model.get_cvar_gaussian(returns, alpha)
        elif distribution == "studentt":
            conditional_value_at_risk = risk_model.get_var_studentt(returns, alpha)
        elif distribution == "laplace":
            conditional_value_at_risk = risk_model.get_cvar_laplace(returns, alpha)
        elif distribution == "logistic":
            conditional_value_at_risk = risk_model.get_cvar_logistic(returns, alpha)
        else:
            raise ValueError(
                "Distribution must be historic, gaussian, studentt, laplace or logistic."
            )

        if growth:
            return calculate_growth(
                conditional_value_at_risk,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return conditional_value_at_risk.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_entropic_value_at_risk(
        self,
        period=None,
        alpha: float = 0.05,
        within_period: bool = True,
        rounding=4,
        growth: bool = False,
        lag=1,
    ):
        """
        Calculate the Entropic Value at Risk (EVaR) of an investment portfolio or asset's returns.

        Entropic Value at Risk (EVaR) is a risk management metric that quantifies upper bound for the value
        at risk (VaR) and the conditional value at risk (CVaR) over a specified time horizon and confidence
        level. EVaR is obtained from the Chernoff inequality. It provides insights into the downside risk
        associated with an investment and helps investors make informed decisions about risk tolerance.

        The EVaR is calculated as the upper bound of VaR and CVaR with a given confidence level (e.g., 5% for
        alpha=0.05).

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, quarterly, or yearly).
            Defaults to "yearly".
            alpha (float, optional): The confidence level for EVaR calculation (e.g., 0.05 for 95% confidence).
            Defaults to 0.05.
            within_period (bool, optional): Whether to calculate EVaR within the specified period or for the entire
            period. Thus whether to look at the CVaR within a specific year (if period = 'yearly') or look at the entirety
            of all years. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the CVaR values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.Series: EVaR values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates EVaR for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of EVaR values using the specified `lag`.

        Example:
        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key=FMP_KEY)

        toolkit.risk.get_entropic_value_at_risk()
        ```

        Which returns:

        |      |    AMZN |    TSLA |   SPY |
        |:-----|--------:|--------:|--------:|
        | 2012 | -0.0392 | -0.0604 | -0.0177 |
        | 2013 | -0.0377 | -0.0928 | -0.0152 |
        | 2014 | -0.0481 | -0.0689 | -0.0162 |
        | 2015 | -0.046  | -0.0564 | -0.0227 |
        | 2016 | -0.043  | -0.0571 | -0.0188 |
        | 2017 | -0.0289 | -0.0501 | -0.0091 |
        | 2018 | -0.0518 | -0.085  | -0.0252 |
        | 2019 | -0.0327 | -0.071  | -0.0173 |
        | 2020 | -0.054  | -0.1211 | -0.0497 |
        | 2021 | -0.0352 | -0.0782 | -0.0183 |
        | 2022 | -0.0758 | -0.1012 | -0.0362 |
        | 2023 | -0.0471 | -0.0793 | -0.0188 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = helpers.handle_return_data_periods(self, period, within_period)

        entropic_value_at_risk = risk_model.get_evar_gaussian(returns, alpha)

        if growth:
            return calculate_growth(
                entropic_value_at_risk,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return entropic_value_at_risk.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_maximum_drawdown(
        self,
        period=None,
        within_period: bool = True,
        rounding=None,
        growth: bool = False,
        lag=1,
    ):
        """
        Calculate the Maximum Drawdown (MDD) of an investment portfolio or asset's returns.

        Maximum Drawdown (MDD) is a risk management metric that quantifies the largest historical loss of
        n investment portfolio or asset experienced over a specified time horizon. It provides insights into
        the downside risk associated with an investment and helps investors make informed decisions about
        risk tolerance.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, quarterly, or yearly).
            Defaults to "yearly".
            alpha (float, optional): The confidence level for CVaR calculation (e.g., 0.05 for 95% confidence).
            Defaults to 0.05.
            within_period (bool, optional): Whether to calculate CVaR within the specified period or for the entire
            period. Thus whether to look at the CVaR within a specific year (if period = 'yearly') or look at the entirety
            of all years. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the CVaR values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.Series: Maximum Drawdown values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates MMD for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of MMD values using the specified `lag`.

        Example:
        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key=FMP_KEY)

        toolkit.risk.get_maximum_drawdown()
        ```

        Which returns:

        |      |    AMZN |    TSLA |
        |:-----|--------:|--------:|
        | 2012 | -0.1570 | -0.1601 |
        | 2013 | -0.1259 | -0.3768 |
        | 2014 | -0.2948 | -0.3085 |
        | 2015 | -0.1371 | -0.2669 |
        | 2016 | -0.2432 | -0.357  |
        | 2017 | -0.1085 | -0.2227 |
        | 2018 | -0.3410 | -0.3399 |
        | 2019 | -0.1561 | -0.4847 |
        | 2020 | -0.2274 | -0.6063 |
        | 2021 | -0.1457 | -0.3625 |
        | 2022 | -0.5198 | -0.7272 |
        | 2023 | -0.1964 | -0.2823 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = helpers.handle_return_data_periods(self, period, within_period)

        maximum_drawdown = risk_model.get_max_drawdown(returns)

        if growth:
            return calculate_growth(
                maximum_drawdown,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return maximum_drawdown.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_ulcer_index(
        self,
        period=None,
        rolling=14,
        rounding=4,
        growth: bool = False,
        lag=1,
    ):
        """
        The Ulcer Index is a financial metric used to assess the risk and volatility of an
        investment portfolio or asset. Developed by Peter Martin in the 1980s, the Ulcer Index
        is particularly useful for evaluating the downside risk and drawdowns associated with investments.

        The Ulcer Index differs from traditional volatility measures like standard deviation or variance
        because it focuses on the depth and duration of drawdowns rather than the dispersion of
        returns.

        The formula is a follows:

        Ulcer Index = SQRT(SUM[(Pn / Highest High)^2] / n)

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, quarterly, or yearly).
            Defaults to "yearly".
            rolling (int, optional): The rolling period to use for the calculation. Defaults to 14.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the UI values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.Series: UI values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates UI for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of VaR values using the specified `lag`.

        Example:
        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["AMZN", "TSLA"], api_key=FMP_KEY)

        toolkit.risk.get_ulcer_index()
        ```

        Which returns:

        |      |   AMZN |   TSLA |   Benchmark |
        |:-----|-------:|-------:|------------:|
        | 2012 | 0.0497 | 0.0454 |      0.0234 |
        | 2013 | 0.035  | 0.0829 |      0.0142 |
        | 2014 | 0.0659 | 0.0746 |      0.0174 |
        | 2015 | 0.0273 | 0.0624 |      0.0238 |
        | 2016 | 0.0519 | 0.0799 |      0.0151 |
        | 2017 | 0.0241 | 0.0616 |      0.0067 |
        | 2018 | 0.0619 | 0.0892 |      0.0356 |
        | 2019 | 0.0373 | 0.0839 |      0.016  |
        | 2020 | 0.0536 | 0.1205 |      0.0594 |
        | 2021 | 0.0427 | 0.085  |      0.0136 |
        | 2022 | 0.1081 | 0.1373 |      0.0492 |
        | 2023 | 0.0475 | 0.0815 |      0.0186 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = helpers.handle_return_data_periods(self, period, True)

        ulcer_index = risk_model.get_ui(returns, rolling)

        if growth:
            return calculate_growth(
                ulcer_index,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return ulcer_index.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_skewness(
        self,
        period=None,
        within_period: bool = True,
        rounding=None,
        growth: bool = False,
        lag=1,
    ):
        """
        Calculate the Skewness of an investment portfolio or asset's returns.

        Skewness is a statistical measure used in finance to assess the asymmetry in the distribution of
        returns for an investment portfolio or asset over a defined period. It offers valuable insights
        into the shape of the return distribution, indicating whether returns are skewed towards the
        positive or negative side of the mean. Skewness is a crucial tool for investors and analysts seeking
        to understand the potential risk and return characteristics of an investment, aiding in the assessment
        of the distribution's tails and potential outliers. It provides a means to gauge the level of
        skew in returns, enabling more informed investment decisions and risk management strategies.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, quarterly, or yearly).
            Defaults to "yearly".
            alpha (float, optional): The confidence level for CVaR calculation (e.g., 0.05 for 95% confidence).
            Defaults to 0.05.
            within_period (bool, optional): Whether to calculate CVaR within the specified period or for the entire
            period. Thus whether to look at the CVaR within a specific year (if period = 'yearly') or look at the entirety
            of all years. Defaults to True.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the CVaR values over time. Defaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.Series: CVaR values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates Skew for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of VaR values using the specified `lag`.

        Example:
        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["MSFT", "AAPL", "TSLA"], api_key=FMP_KEY)

        toolkit.risk.get_skewness()
        ```

        Which returns:

        |      |    MSFT |    AAPL |    TSLA |
        |:-----|--------:|--------:|--------:|
        | 2019 | -0.194  | -0.9216 | -0.0646 |
        | 2020 | -0.0747 | -0.0586 | -0.1824 |
        | 2021 | -0.0194 | -0.0716 |  0.6572 |
        | 2022 |  0.1478 |  0.3164 | -0.0263 |
        | 2023 |  0.5252 |  0.0318 | -0.0972 |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = helpers.handle_return_data_periods(self, period, within_period)

        skewness = risk_model.get_skewness(returns)

        if growth:
            return calculate_growth(
                skewness,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return skewness.round(rounding if rounding else self._rounding)

    @handle_errors
    def get_kurtosis(
        self,
        period=None,
        within_period: bool = True,
        fisher: bool = False,
        rounding=None,
        growth: bool = False,
        lag=1,
    ):
        """
        Calculate the Kurtosis of an investment portfolio or asset's returns.

        Kurtosis is a statistical measure used in finance to evaluate the shape of the probability
        distribution of returns for an investment portfolio or asset over a defined time period.
        It assesses the "tailedness" of the return distribution, indicating whether returns have
        fatter or thinner tails compared to a normal distribution. Kurtosis plays a critical role
        in risk assessment by revealing the potential presence of extreme outliers or the likelihood
        of heavy tails in the return data. This information aids investors and analysts in
        understanding the degree of risk associated with an investment and assists in making
        more informed decisions regarding risk tolerance. In essence, kurtosis serves as a valuable
        tool for comprehending the distribution characteristics of returns, offering insights
        into the potential for rare but significant events in the financial markets.

        Args:
            period (str, optional): The data frequency for returns (daily, weekly, quarterly, or yearly).
            Defaults to "yearly".
            within_period (bool, optional): Whether to calculate CVaR within the specified period or for the entire
            period. Thus whether to look at the CVaR within a specific year (if period = 'yearly') or look at
            the entirety of all years. Defaults to True.
            fisher (bool, optional): Whether to use Fisher's definition of kurtosis (kurtosis = 0.0
            for a normal distribution).
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to 4.
            growth (bool, optional): Whether to calculate the growth of the CVaR values over time.
            efaults to False.
            lag (int | list[int], optional): The lag to use for the growth calculation. Defaults to 1.

        Returns:
            pd.Series: CVaR values with time as the index.

        Notes:
        - The method retrieves historical return data based on the specified `period` and calculates VaR for each
        asset in the Toolkit instance.
        - If `growth` is set to True, the method calculates the growth of VaR values using the specified `lag`.

        Example:
        ```python
        from financetoolkit import Toolkit

        toolkit = Toolkit(["MSFT", "AAPL", "TSLA"]], api_key=FMP_KEY)

        toolkit.risk.get_kurtosis()
        ```

        Which returns:

        |      |   MSFT |    AAPL |   TSLA |
        |:-----|-------:|--------:|-------:|
        | 2019 | 4.0972 | 10.0741 | 9.128  |
        | 2020 | 9.2914 |  6.6307 | 5.2189 |
        | 2021 | 3.3152 |  3.3352 | 7.3197 |
        | 2022 | 3.852  |  4.0085 | 3.3553 |
        | 2023 | 4.2908 |  4.4568 | 4.07   |
        """
        period = period if period else "quarterly" if self._quarterly else "yearly"
        returns = helpers.handle_return_data_periods(self, period, within_period)

        kurtosis = risk_model.get_kurtosis(returns, fisher=fisher)

        if growth:
            return calculate_growth(
                kurtosis,
                lag=lag,
                rounding=rounding if rounding else self._rounding,
                axis="index",
            )

        return kurtosis.round(rounding if rounding else self._rounding)
