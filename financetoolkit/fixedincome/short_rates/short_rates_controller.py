"""Short Rate Models Module"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.economics.helpers import (
    validate_period,
)
from financetoolkit.fixedincome.curves import curve_model
from financetoolkit.fixedincome.curves.curves_controller import CurvesFixedIncome
from financetoolkit.fixedincome.short_rates import short_rate_model
from financetoolkit.utilities import logger_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import apply_rounding

# ruff: noqa: E501

logger = logger_model.get_logger()

# The observations per year of each period, for calibrating the short-rate models.
PERIODS_PER_YEAR = {"daily": 252, "weekly": 52, "monthly": 12}

SHORT_RATE_PARAMETERS = [
    "Speed of Mean Reversion",
    "Long-Run Mean",
    "Volatility",
    "Current Rate",
    "Half-Life",
]

SHORT_RATE_SIMULATION_MODELS = ("hull_white", "vasicek", "cir")

SIMULATION_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)

# The most rates a simulation may hold (paths times steps), which bounds its memory use.
MAX_SIMULATED_VALUES = 20_000_000


@filter_shown_columns
@track_sources
class ShortRatesFixedIncome(CurvesFixedIncome):
    """
    Short rate models: Vasicek, CIR and Hull-White fits and simulations.
    """

    @handle_errors
    def get_short_rate_model(
        self,
        country: str = "United States",
        short_rate: str = "3M",
        period: str = "daily",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Calibrates the Vasicek and Cox-Ingersoll-Ross (CIR) short-rate models to the history
        of a country's short-term government bond yield and compares the zero-coupon curve
        each model implies with today's. Short-rate models describe how the interest rate
        moves, pulled back to a long-run mean b at a speed a with a volatility σ:

            - Vasicek (1977): dr = a (b - r) dt + σ dW
            - CIR (1985): dr = a (b - r) dt + σ √r dW

        In CIR the volatility shrinks as the rate approaches zero, so the rate stays
        positive when 2ab ≥ σ² (the Feller condition) and can otherwise touch zero; in
        Vasicek it can turn negative. The models describe a continuously compounded short
        rate, so the yields are converted to continuous compounding first (US bill yields
        from their bond-equivalent quote). Both give bond prices, and so a full
        zero-coupon curve, in closed form. The parameters are estimated by regressing the
        change of the rate on its level over the period (CIR on the changes scaled by the
        square root of the rate, which needs a rate above zero throughout). The half-life is
        the time it takes to close half the gap to the long-run mean, ln(2) / a.

        A model calibrated to the history does not reproduce today's curve, which is what
        the comparison with the market shows; the Hull-White model of
        `get_short_rate_simulation` adds a time-varying mean that does.

        For more information, see the following papers:

        - Vasicek, O. (1977). "An Equilibrium Characterization of the Term Structure."
          Journal of Financial Economics, 5(2), 177-188.
        - Cox, J.C., Ingersoll, J.E. and Ross, S.A. (1985). "A Theory of the Term Structure
          of Interest Rates." Econometrica, 53(2), 385-407.

        Also known as: short rate model, mean reversion of interest rates, Vasicek model, CIR
        model, equilibrium term structure model.

        Args:
            country (str, optional): The country, see `get_government_bond_yield_curve`.
                Defaults to "United States".
            short_rate (str, optional): The maturity of the yield used as the short rate,
                such as "1M", "3M" or "6M". Defaults to "3M".
            period (str, optional): Whether to calibrate on daily, weekly or monthly rates.
                Defaults to "daily".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per model (columns Vasicek and CIR) the Speed of Mean Reversion, the
            Long-Run Mean, the Volatility, the Current Rate and the Half-Life in years, then
            the zero-coupon rate per maturity (continuously compounded), with the market's
            zero-coupon rates in the Market column.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2000-01-01', end_date='2026-10-23')

        fixedincome.get_short_rate_model(short_rate='6M')
        ```

        Which returns:

        | Parameter               |   Vasicek |    CIR |   Market |
        |:------------------------|----------:|-------:|---------:|
        | Speed of Mean Reversion |    0.0873 | 0.1481 | nan      |
        | Long-Run Mean           |    0.0144 | 0.017  | nan      |
        | Volatility              |    0.0059 | 0.0558 | nan      |
        | Current Rate            |    0.0425 | 0.0425 |   0.0425 |
        | Half-Life               |    7.9398 | 4.6816 | nan      |
        | Zero Rate 1M            |    0.0424 | 0.0424 |   0.041  |
        | Zero Rate 1.5M          |    0.0424 | 0.0423 |   0.041  |
        | Zero Rate 2M            |    0.0423 | 0.0422 |   0.0415 |
        | Zero Rate 3M            |    0.0422 | 0.0421 |   0.042  |
        | Zero Rate 4M            |    0.0421 | 0.0419 |   0.0427 |
        | Zero Rate 6M            |    0.0419 | 0.0416 |   0.0425 |
        | Zero Rate 1Y            |    0.0413 | 0.0407 |   0.0439 |
        | Zero Rate 2Y            |    0.0402 | 0.039  |   0.0464 |
        | Zero Rate 3Y            |    0.0391 | 0.0375 |   0.047  |
        | Zero Rate 5Y            |    0.0371 | 0.0347 |   0.0477 |
        | Zero Rate 7Y            |    0.0353 | 0.0324 |   0.0485 |
        | Zero Rate 10Y           |    0.0328 | 0.0297 |   0.0497 |
        | Zero Rate 20Y           |    0.0269 | 0.0242 |   0.0548 |
        | Zero Rate 30Y           |    0.0232 | 0.0216 |   0.0534 |
        """
        period = validate_period(
            period, ["daily", "weekly", "monthly"], "short rate model"
        )
        curve = self._yield_curves(country, period)[country]
        if short_rate not in curve.columns:
            raise ValueError(
                f"The {country} yield curve has no {short_rate} maturity. Choose from "
                f"{', '.join(curve.columns)}."
            )

        rates = self._continuous_short_rate(curve[short_rate], country, short_rate)
        latest = curve.dropna(how="all").iloc[[-1]]
        zero_rates = self._to_zero_coupon(latest, country).iloc[0].dropna()
        years = np.array(
            [curve_model.maturity_in_years(label) for label in zero_rates.index]
        )

        rows = [
            *SHORT_RATE_PARAMETERS,
            *(f"Zero Rate {label}" for label in zero_rates.index),
        ]
        result = pd.DataFrame(np.nan, index=rows, columns=["Vasicek", "CIR", "Market"])

        for model, column in (("vasicek", "Vasicek"), ("cir", "CIR")):
            try:
                parameters = short_rate_model.calibrate_short_rate_model(
                    rates, model, PERIODS_PER_YEAR[period]
                )
            except ValueError as error:
                logger.warning(
                    "The %s model could not be calibrated: %s", column, error
                )
                continue
            model_yields = short_rate_model.get_model_yields(
                years,
                model,
                parameters["Speed of Mean Reversion"],
                parameters["Long-Run Mean"],
                parameters["Volatility"],
                parameters["Current Rate"],
            )
            result[column] = [*parameters.values(), *model_yields]

        result.loc["Current Rate", "Market"] = rates.iloc[-1]
        result.loc[[f"Zero Rate {label}" for label in zero_rates.index], "Market"] = (
            np.log1p(zero_rates.to_numpy())
        )
        result.index.name = "Parameter"

        return apply_rounding(
            result, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_short_rate_simulation(
        self,
        model: str = "hull_white",
        country: str = "United States",
        short_rate: str = "3M",
        period: str = "daily",
        years: float = 10,
        steps_per_year: int = 12,
        paths: int = 10000,
        seed: int | None = 0,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Simulates the future path of a country's short-term interest rate with the
        Hull-White, Vasicek or Cox-Ingersoll-Ross model and summarizes the distribution of
        the paths per step: the mean and the 5th, 25th, 50th, 75th and 95th percentiles.
        Simulated rate paths value interest rate derivatives and path-dependent products
        (such as mortgages with prepayment), and stress the interest income of a balance
        sheet.

        The speed of mean reversion and the volatility are calibrated to the history of the
        rate, see `get_short_rate_model`. In the Vasicek and CIR models the rate is pulled
        to the long-run mean of that history. The Hull-White (1990) model replaces it with a
        mean that varies over time and follows from today's zero-coupon curve, so that the
        model reproduces the discount factors of that curve (its expected short rate lies
        above the forward rate by the convexity term σ² (1 - e^(-at))² / (2a²)):

            dr = [θ(t) - a r] dt + σ dW,  θ(t) = ∂f/∂t + a f(t) + σ² (1 - e^(-2at)) / (2a)

        with f the instantaneous forward rate. The paths are simulated with an Euler scheme
        starting at today's rate (CIR's are kept at or above zero).

        For more information, see the following paper:

        - Hull, J. and White, A. (1990). "Pricing Interest-Rate-Derivative Securities." The
          Review of Financial Studies, 3(4), 573-592.

        Also known as: interest rate simulation, Hull-White model, Monte Carlo short rate,
        interest rate scenarios.

        Args:
            model (str, optional): "hull_white", "vasicek" or "cir". Defaults to "hull_white".
            country (str, optional): The country, see `get_government_bond_yield_curve`.
                Defaults to "United States".
            short_rate (str, optional): The maturity of the yield used as the short rate,
                such as "1M", "3M" or "6M". Defaults to "3M".
            period (str, optional): Whether to calibrate on daily, weekly or monthly rates.
                Defaults to "daily".
            years (float, optional): The horizon in years. Defaults to 10.
            steps_per_year (int, optional): The steps per year. Defaults to 12.
            paths (int, optional): The number of simulated paths. Defaults to 10,000.
            seed (int | None, optional): The random seed, None for a different simulation
                every time. Defaults to 0.
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: The Mean and the percentiles of the simulated rate (as decimals),
            indexed by the time in years.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2000-01-01', end_date='2026-10-23')

        simulation = fixedincome.get_short_rate_simulation(
            short_rate='6M', years=5, steps_per_year=1
        )

        simulation
        ```

        Which returns:

        |   Years |   Mean |     5% |    25% |    50% |    75% |    95% |
        |--------:|-------:|-------:|-------:|-------:|-------:|-------:|
        |       0 | 0.0425 | 0.0425 | 0.0425 | 0.0425 | 0.0425 | 0.0425 |
        |       1 | 0.0489 | 0.0393 | 0.0449 | 0.0489 | 0.0529 | 0.0587 |
        |       2 | 0.0483 | 0.035  | 0.0429 | 0.0483 | 0.0537 | 0.0614 |
        |       3 | 0.0485 | 0.0332 | 0.0421 | 0.0484 | 0.0547 | 0.0643 |
        |       4 | 0.0493 | 0.0324 | 0.0423 | 0.0492 | 0.0563 | 0.0666 |
        |       5 | 0.0503 | 0.0322 | 0.0427 | 0.0503 | 0.0579 | 0.0689 |
        """
        period = validate_period(
            period, ["daily", "weekly", "monthly"], "short rate simulation"
        )
        model = model.lower().replace("-", "_").replace(" ", "_")
        if model not in SHORT_RATE_SIMULATION_MODELS:
            raise ValueError(
                f"The model must be one of {', '.join(SHORT_RATE_SIMULATION_MODELS)}, not {model!r}."
            )
        if paths * years * steps_per_year > MAX_SIMULATED_VALUES:
            raise ValueError(
                f"The simulation of {paths:,} paths over {years * steps_per_year:,.0f} steps is "
                f"too large; keep paths times steps under {MAX_SIMULATED_VALUES:,}."
            )

        curve = self._yield_curves(country, period)[country]
        if short_rate not in curve.columns:
            raise ValueError(
                f"The {country} yield curve has no {short_rate} maturity. Choose from "
                f"{', '.join(curve.columns)}."
            )

        parameters = short_rate_model.calibrate_short_rate_model(
            self._continuous_short_rate(curve[short_rate], country, short_rate),
            "cir" if model == "cir" else "vasicek",
            PERIODS_PER_YEAR[period],
        )
        speed = parameters["Speed of Mean Reversion"]
        if not speed > 0:
            raise ValueError(
                f"The {country} {short_rate} rate shows no mean reversion between "
                f"{self._start_date} and {self._end_date}; choose a longer period."
            )

        mean: float | np.ndarray = parameters["Long-Run Mean"]
        if model == "hull_white":
            latest = curve.dropna(how="all").iloc[[-1]]
            zero_rates = self._to_zero_coupon(latest, country).iloc[0].dropna()
            mean = short_rate_model.hull_white_mean(
                np.array(
                    [curve_model.maturity_in_years(label) for label in zero_rates.index]
                ),
                np.log1p(zero_rates.to_numpy()),
                speed,
                parameters["Volatility"],
                parameters["Current Rate"],
                years,
                steps_per_year,
            )

        simulated = short_rate_model.simulate_short_rate(
            model,
            speed,
            mean,
            parameters["Volatility"],
            parameters["Current Rate"],
            years=years,
            steps_per_year=steps_per_year,
            paths=paths,
            seed=seed,
        )

        summary = pd.DataFrame(
            {
                "Mean": simulated.mean(axis=0),
                **{
                    f"{quantile:.0%}": np.quantile(simulated, quantile, axis=0)
                    for quantile in SIMULATION_QUANTILES
                },
            },
            index=pd.Index(
                np.round(np.arange(simulated.shape[1]) / steps_per_year, 4),
                name="Years",
            ),
        )

        return apply_rounding(
            summary, rounding if rounding is not None else self._rounding
        )
