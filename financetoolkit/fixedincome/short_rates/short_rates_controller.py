"""Short Rate Models Module"""

__docformat__ = "google"

import numpy as np
import pandas as pd

from financetoolkit.economics.helpers import validate_period
from financetoolkit.fixedincome.curves import curve_model
from financetoolkit.fixedincome.curves.curves_controller import CurvesFixedIncome
from financetoolkit.fixedincome.helpers import parse_maturities
from financetoolkit.fixedincome.short_rates import (
    short_rate_model,
    trinomial_tree_model,
)
from financetoolkit.fixedincome.swaps import swap_model
from financetoolkit.utilities import logger_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import (
    check_option,
    handle_errors,
    validate_window,
)
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

# The swaption models that fit today's curve, so that at-the-money strikes are consistent.
SWAPTION_MODELS = ("hull_white", "cir++")
DEFAULT_SWAPTION_EXPIRIES = (1, 2, 5, 10)
DEFAULT_SWAPTION_TENORS = (1, 2, 5, 10, 30)

# The most rates a simulation may hold (paths times steps), which bounds its memory use.
MAX_SIMULATED_VALUES = 20_000_000


@filter_shown_columns
@track_sources
class ShortRatesFixedIncome(CurvesFixedIncome):
    """
    Short rate models: Vasicek, CIR and Hull-White fits and simulations, and swaption prices
    in the Hull-White and CIR++ models.
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
        of a country's short-term government bond yield: how fast the rate reverts to its
        long-run mean and how volatile it is, with the zero-coupon curve each model implies
        next to today's.

        Short-rate models describe how the interest rate moves, pulled back to a long-run
        mean b at a speed a with a volatility σ:

            - Vasicek (1977): dr = a (b - r) dt + σ dW
            - CIR (1985): dr = a (b - r) dt + σ √r dW

        In CIR the volatility shrinks as the rate approaches zero, so the rate stays
        positive when 2ab ≥ σ² (the Feller condition, Feller 1951, reported as the Feller
        Ratio 2ab / σ², which meets it at 1 or more) and can otherwise touch zero; in
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
            period (str, optional): The frequency of the rates. Options are: "daily", "weekly", "monthly". Defaults to
                "daily".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per model (columns Vasicek and CIR) the Speed of Mean Reversion, the
            Long-Run Mean, the Volatility, the Current Rate, the Half-Life in years and (CIR
            only) the Feller Ratio, then the zero-coupon rate per maturity (continuously compounded), with the market's
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
        | Feller Ratio            |  nan      | 1.6148 | nan      |
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
            "Feller Ratio",
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
            feller_ratio = (
                short_rate_model.get_feller_ratio(
                    parameters["Speed of Mean Reversion"],
                    parameters["Long-Run Mean"],
                    parameters["Volatility"],
                )
                if model == "cir"
                else np.nan
            )
            result[column] = [*parameters.values(), feller_ratio, *model_yields]

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
        simulations: int = 10000,
        seed: int | None = 0,
        scheme: str = "euler",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ):
        """
        Simulates the future path of a country's short-term interest rate with the
        Hull-White, Vasicek or Cox-Ingersoll-Ross model: the distribution of the rate per
        step, the mean and the 5th, 25th, 50th, 75th and 95th percentiles of the paths.

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

        with f the instantaneous forward rate. The paths start at today's rate and are
        simulated with an Euler scheme (CIR's are kept at or above zero) or, with the
        exact scheme, by drawing the rate at the end of every step from its exact
        distribution given the rate at the start: normal for Vasicek and Hull-White
        (r' = r e^(-aΔ) + m (1 - e^(-aΔ)) + σ √((1 - e^(-2aΔ)) / (2a)) Z, with m the mean of
        the step) and a scaled non-central chi-squared for CIR (Glasserman, 2003, Monte Carlo
        Methods in Financial Engineering, sections 3.3 and 3.4). The exact scheme has no
        discretization error at any step size, and its CIR rate stays at or above zero
        without being cut off, which matters most when the Feller ratio of
        `get_short_rate_model` is below 1.

        The simulated short rates are continuously compounded, the rate these models describe:
        the quoted yield of the short_rate maturity is converted from its quote convention
        (for US bills the bond-equivalent yield) and the zero-coupon curve of the Hull-White
        mean from annual compounding.

        For more information, see the following paper:

        - Hull, J. and White, A. (1990). "Pricing Interest-Rate-Derivative Securities." The
          Review of Financial Studies, 3(4), 573-592.

        Also known as: interest rate simulation, Hull-White model, Monte Carlo short rate,
        interest rate scenarios.

        Shares similar logic:
        - FinancePy, CIRMonteCarlo (cir): https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/models/cir_montecarlo.py#L10

        Args:
            model (str, optional): The model. Options are: "hull_white" (Hull-White, fitted to today's curve), "vasicek"
                (Vasicek), "cir" (Cox-Ingersoll-Ross). Defaults to "hull_white".
            country (str, optional): The country, see `get_government_bond_yield_curve`.
                Defaults to "United States".
            short_rate (str, optional): The maturity of the yield used as the short rate,
                such as "1M", "3M" or "6M". Defaults to "3M".
            period (str, optional): The frequency of the rates. Options are: "daily", "weekly", "monthly". Defaults to
                "daily".
            years (float, optional): The horizon in years. Defaults to 10.
            steps_per_year (int, optional): The steps per year, at least 1. Defaults to 12.
            simulations (int, optional): The number of simulated paths, at least 1. Defaults to 10,000.
            seed (int | None, optional): The random seed, None for a different simulation
                every time. Defaults to 0.
            scheme (str, optional): The simulation scheme. Options are: "euler" (an Euler scheme), "exact" (sampling the
                exact distribution of the rate). Defaults to "euler".
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
        check_option("model", model, SHORT_RATE_SIMULATION_MODELS)
        check_option("scheme", scheme, short_rate_model.SCHEMES)
        validate_window("simulations", simulations)
        validate_window("steps_per_year", steps_per_year)
        if simulations * years * steps_per_year > MAX_SIMULATED_VALUES:
            raise ValueError(
                f"The simulation of {simulations:,} paths over {years * steps_per_year:,.0f} steps is "
                f"too large; keep simulations times steps under {MAX_SIMULATED_VALUES:,}."
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
                scheme,
            )

        simulated = short_rate_model.simulate_short_rate(
            model,
            speed,
            mean,
            parameters["Volatility"],
            parameters["Current Rate"],
            years=years,
            steps_per_year=steps_per_year,
            simulations=simulations,
            seed=seed,
            scheme=scheme,
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

    @handle_errors
    def get_swaption_price(
        self,
        expiries: float | str | list | None = None,
        tenors: float | str | list | None = None,
        model: str = "hull_white",
        swaption_type: str = "payer",
        strike: float | None = None,
        fixed_frequency: int = 1,
        country: str = "United States",
        short_rate: str = "3M",
        period: str = "daily",
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
        exercise: str = "european",
    ):
        """
        Calculates the prices of European or Bermudan swaptions for a grid of expiries and
        swap tenors in a one-factor short-rate model fitted to today's government
        zero-coupon curve: what the right to enter an interest rate swap at the option's
        expiry is worth.

        A payer swaption is the right to pay the fixed rate (it gains when rates rise), a
        receiver swaption the right to receive it. Without a strike every swaption is at the
        money: struck at the forward swap rate of its expiry and tenor on the same curve
        (`get_swap_rate` on the government curve, starting at the expiry).

        Two models reproduce the curve and so price at-the-money swaptions consistently:

            - Hull-White (1990): dr = [θ(t) - a r] dt + σ dW, Vasicek with a time-varying
              mean, so rates are normal and can turn negative
            - CIR++ (Brigo and Mercurio, 2001): r = x + φ(t), with x a Cox-Ingersoll-Ross
              process dx = a (b - x) dt + σ √x dW and φ the deterministic shift that fits the
              curve, so the volatility grows with the level of rates

        The speed of mean reversion a and the volatility σ (and for CIR++ the long-run mean
        b) are calibrated to the history of the short_rate yield between the start and end
        date, see `get_short_rate_model`; CIR++ starts its process at today's short rate.
        The price follows from Jamshidian's (1989) decomposition: at expiry the payer
        swaption is a put on a coupon bond paying the fixed rate, which in a one-factor model
        splits into puts on its zero-coupon bonds, each priced in closed form (see
        `short_rate_model.get_swaption` and `get_zero_coupon_bond_option`).

        A Bermudan swaption (Hull-White only) can also be exercised on
        every fixed payment date after the expiry, into the rest of a swap that ends at the
        expiry plus the tenor (co-terminal, "10 non-call 1"). It has no closed form and is
        valued on a Hull-White trinomial tree fitted exactly to the zero curve with the
        two-stage procedure of Hull and White (1994), with at most 1/50 of a year between
        two dates of the tree, comparing at every exercise date the value of the swap with
        the value of waiting (see `trinomial_tree_model.get_bermudan_swaption`). It is worth
        at least the European swaption of the same expiry, tenor and strike.

        The formula is as follows:

        - c_i = K τ_i, plus 1 on the last date T_n
        - x* solves SUM(c_i P(T, T_i; x*)) = 1, X_i = P(T, T_i; x*)
        - Payer Swaption = SUM(c_i ZBP(T, T_i, X_i))
        - Receiver Swaption = SUM(c_i ZBC(T, T_i, X_i))
        - Bermudan: Option = max(E[Option next date] · e^(-r Δt), ±(1 - B(T_k))) at
          T_0 = T, T_1, ..., T_{n-1}, with B(T_k) the coupon bond's payments after T_k on
          the tree (+ for a payer, - for a receiver)

        with T the expiry, T_i the fixed payment dates every 1 / fixed_frequency years after
        it, τ_i = 1 / fixed_frequency, K the strike, P(T, T_i; x) the bond price at expiry
        in state x of the model and ZBP and ZBC the zero-coupon bond puts and calls of the
        model. The price is per 1 of notional (0.01 is 1% of the notional).

        The short rate and the model parameters are continuously compounded (the yield is
        converted from its quote convention, as in `get_short_rate_model`). The zero-coupon
        curve is that of the end date, bootstrapped from the par yields and annually
        compounded, and the strike is a nominal annual rate paid fixed_frequency times a year,
        with no day count. The calibration uses the history up to the end date only.

        For more information, see the following sources:

        - Jamshidian, F. (1989). "An Exact Bond Option Formula." Journal of Finance, 44(1),
          205-209.
        - Hull, J. and White, A. (1990). "Pricing Interest-Rate-Derivative Securities." The
          Review of Financial Studies, 3(4), 573-592.
        - Brigo, D. and Mercurio, F. (2001). "A Deterministic-Shift Extension of Analytically
          Tractable and Time-Homogeneous Short-Rate Models." Finance and Stochastics, 5(3),
          369-387.

        - Hull, J. and White, A. (1994). "Numerical Procedures for Implementing Term
          Structure Models I: Single-Factor Models." Journal of Derivatives, 2(1), 7-16.

        Also known as: swaption price, Bermudan swaption, Hull-White swaption, CIR++ swaption, Jamshidian
        decomposition, interest rate option.

        Referenced against:
        - QuantLib, JamshidianSwaptionEngine (european): https://www.quantlib.org/reference/class_quant_lib_1_1_jamshidian_swaption_engine.html
        - QuantLib, TreeSwaptionEngine (bermudan): https://www.quantlib.org/reference/class_quant_lib_1_1_tree_swaption_engine.html

        Args:
            expiries (float | str | list | None, optional): The expiries of the swaptions, in
                years or as labels such as '6M'. Defaults to None, 1, 2, 5 and 10 years.
            tenors (float | str | list | None, optional): The tenors of the underlying swaps,
                in years or as labels such as '10Y'. Defaults to None, 1, 2, 5, 10 and 30
                years.
            model (str, optional): The model. Options are: "hull_white" (Hull-White), "cir++" (CIR++ of Brigo and
                Mercurio). Defaults to "hull_white".
            swaption_type (str, optional): The swaption. Options are: "payer" (the right to pay the fixed rate),
                "receiver" (the right to receive it). Defaults to "payer".
            strike (float | None, optional): The fixed rate of the swaps (decimal). Defaults
                to None, the at-the-money forward swap rate of each expiry and tenor.
            fixed_frequency (int, optional): The fixed payments per year, at least 1. Defaults to 1.
            country (str, optional): The country, see `get_government_bond_yield_curve`.
                Defaults to "United States".
            short_rate (str, optional): The maturity of the yield used as the short rate,
                such as "1M", "3M" or "6M". Defaults to "3M".
            period (str, optional): The frequency of the rates. Options are: "daily", "weekly", "monthly". Defaults to
                "daily".
            rounding (int | None, optional): The number of decimals to round the results to. Defaults to None.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.
            exercise (str, optional): The exercise. Options are: "european" (at the expiry only), "bermudan" (on every
                fixed payment date from the expiry on, Hull-White only). Defaults to "european".

        Returns:
            pd.DataFrame: The swaption price per 1 of notional, by expiry in years (rows) and
            tenor in years (columns).

        Notes:
        - One factor moves the whole curve, so the model ties the volatilities of all
          expiries and tenors to two parameters; fitted to the history of a short rate it
          does not match the volatilities that swaption markets quote.
        - CIR++ needs a rate above zero throughout the calibration period (see
          `get_short_rate_model`), and the Feller Ratio there shows whether its process stays
          away from zero.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2000-01-01', end_date='2026-10-23')

        fixedincome.get_swaption_price(short_rate='6M')
        ```

        Which returns:

        |   Expiry |      1 |      2 |      5 |     10 |     30 |
        |---------:|-------:|-------:|-------:|-------:|-------:|
        |        1 | 0.0021 | 0.0039 | 0.008  | 0.012  | 0.0154 |
        |        2 | 0.0027 | 0.005  | 0.0103 | 0.0154 | 0.0198 |
        |        5 | 0.0032 | 0.0061 | 0.0125 | 0.0185 | 0.024  |
        |       10 | 0.003  | 0.0055 | 0.0114 | 0.0167 | 0.0221 |
        """
        period = validate_period(
            period, ["daily", "weekly", "monthly"], "swaption price"
        )
        model = model.lower().replace("-", "_").replace(" ", "_")
        check_option("model", model, SWAPTION_MODELS)
        check_option("exercise", exercise, ("european", "bermudan"))
        validate_window("fixed_frequency", fixed_frequency)
        if exercise == "bermudan" and model != "hull_white":
            raise ValueError(
                "Bermudan swaptions are valued on a Hull-White tree; use model='hull_white'."
            )

        curve = self._yield_curves(country, period)[country]
        if short_rate not in curve.columns:
            raise ValueError(
                f"The {country} yield curve has no {short_rate} maturity. Choose from "
                f"{', '.join(curve.columns)}."
            )
        parameters = short_rate_model.calibrate_short_rate_model(
            self._continuous_short_rate(curve[short_rate], country, short_rate),
            "cir" if model == "cir++" else "vasicek",
            PERIODS_PER_YEAR[period],
        )
        if not parameters["Speed of Mean Reversion"] > 0:
            raise ValueError(
                f"The {country} {short_rate} rate shows no mean reversion between "
                f"{self._start_date} and {self._end_date}; choose a longer period."
            )
        zero_curve = self._latest_zero_curve(country)

        expiry_years = parse_maturities(expiries, DEFAULT_SWAPTION_EXPIRIES)
        tenor_years = parse_maturities(tenors, DEFAULT_SWAPTION_TENORS)
        prices = pd.DataFrame(
            np.nan,
            index=pd.Index(expiry_years, name="Expiry"),
            columns=pd.Index(tenor_years, name="Tenor"),
        )
        if exercise == "bermudan":
            # One tree holds every expiry and payment date, and values all the swaptions.
            tree = trinomial_tree_model.build_short_rate_tree(
                zero_curve,
                [
                    time
                    for expiry in expiry_years
                    for tenor in tenor_years
                    for time in (
                        expiry,
                        *trinomial_tree_model.swap_payment_times(
                            expiry, tenor, fixed_frequency
                        ),
                    )
                ],
                parameters["Speed of Mean Reversion"],
                parameters["Volatility"],
                "hull_white",
            )
        for expiry in expiry_years:
            for tenor in tenor_years:
                fixed_rate = (
                    strike
                    if strike is not None
                    else swap_model.get_swap_rate(
                        zero_curve, tenor, fixed_frequency, forward_start=expiry
                    )[0]
                )
                if exercise == "bermudan":
                    prices.loc[expiry, tenor] = (
                        trinomial_tree_model.price_bermudan_swaption(
                            tree,
                            expiry,
                            tenor,
                            fixed_rate,
                            swaption_type,
                            fixed_frequency,
                        )
                    )
                    continue
                prices.loc[expiry, tenor] = short_rate_model.get_swaption(
                    expiry,
                    tenor,
                    fixed_rate,
                    parameters["Speed of Mean Reversion"],
                    parameters["Volatility"],
                    parameters["Long-Run Mean"],
                    parameters["Current Rate"],
                    zero_curve,
                    model,
                    swaption_type,
                    fixed_frequency,
                )

        return apply_rounding(
            prices, rounding if rounding is not None else self._rounding
        )
