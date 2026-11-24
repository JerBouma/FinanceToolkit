"""Credit Default Swap Model

Credit default swaps on the ISDA CDS Standard Model conventions (Markit, 2009): the protection
buyer pays a running coupon quarterly on an ACT/360 basis, the premium accrued since the last
payment date is paid on default, and the protection seller pays (1 - R) of the notional at the
time of default. The legs are valued on a piecewise flat hazard rate curve with the exact
integrals of the standard model (see `credit_curve_model.get_default_integrals`).

Every function takes the zero-coupon curve as the Toolkit's curves are returned: annually
compounded zero rates by maturity in years, interpolated linearly in between (see
`financetoolkit.fixedincome.helpers.discount_factors`). Times are year fractions from today on
ACT/365F, so a period of Δt years accrues Δt * 365 / 360 of the coupon (ACT/360). The premium
dates are counted back from the maturity in periods of 1 / frequency years, so a maturity
between payment dates has a first period that started before today (the full first coupon of
a standard CDS), whose accrued premium separates the clean value from the full value.
Protection starts today, without the standard model's one-day step-in or business-day
adjustments.
"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from financetoolkit.fixedincome.bonds import bond_model, derivative_model
from financetoolkit.fixedincome.credit.credit_curve_model import (
    DEFAULT_RECOVERY_RATE,
    MAXIMUM_HAZARD_RATE,
    get_default_integrals,
    get_survival_probability,
)
from financetoolkit.fixedincome.helpers import discount_factors
from financetoolkit.utilities.error_model import check_option

# The premium accrues on ACT/360 while the times are ACT/365F year fractions.
ACCRUAL_PER_YEAR = 365 / 360


def get_credit_default_swap(
    maturity: float,
    zero_curve: pd.Series,
    hazard_rates: np.ndarray | list | pd.Series,
    hazard_times: np.ndarray | list,
    coupon: float = 0.01,
    recovery_rate: float = DEFAULT_RECOVERY_RATE,
    notional: float = 1.0,
    frequency: int = 4,
) -> pd.Series:
    """
    Calculate the par spread, risky PV01, upfront and mark-to-market of a credit default
    swap with the ISDA CDS Standard Model (O'Kane, 2008, ch. 6; Markit, 2009): what
    protection against the default of a reference entity costs and is worth.

    The premium leg pays the coupon c on the dates t_k, each accruing τ_k = (t_k - t_(k-1))
    * 365 / 360, as long as the reference entity survives, and the premium accrued since
    t_(k-1) when it defaults in between. The protection leg pays (1 - R) at default.

    The formula is as follows:

    - Full RPV01 = SUM(τ_k P(t_k) Q(t_k)) + 365 / 360 * SUM(∫_t_(k-1)^t_k (t - t_(k-1)) P(t) (-dQ(t)))
    - RPV01 = Full RPV01 - Accrued, Accrued = 365 / 360 * (0 - t_0), the premium accrued
      since the start t_0 ≤ 0 of the current period
    - Protection Leg = (1 - R) * ∫_0^T P(t) (-dQ(t))
    - Par Spread = Protection Leg / RPV01
    - Upfront = Protection Leg - c * RPV01 = (Par Spread - c) * RPV01
    - Mark-to-Market = Notional * (Protection Leg - c * Full RPV01)

    where P(t) = (1 + r(t))^(-t) is the discount factor of the annually compounded zero rate
    r(t), Q(t) the survival probability of the piecewise flat hazard rate curve and R the
    recovery rate. The integrals are exact on the joint grid of the curve maturities, the
    hazard rate maturities and the premium dates, the standard model's piecewise exact
    integration with constant forward and hazard rates between grid points.

    The upfront is the clean price in points of the notional the protection buyer pays to
    enter the contract at the standard coupon c (0.01 or 0.05 since the 2009 "Big Bang"),
    positive when the par spread exceeds the coupon. The mark-to-market is the full value
    to the protection buyer of a contract with coupon c: the upfront less the accrued premium
    the buyer owes, times the notional. The par spread is the coupon at which the clean value
    is zero.

    For more information, see the following sources:

    - O'Kane, D. (2008). "Modelling Single-name and Multi-name Credit Derivatives." Wiley,
      ch. 6.
    - O'Kane, D. and Turnbull, S. (2003). "Valuation of Credit Default Swaps." Lehman
      Brothers Quantitative Credit Research Quarterly.
    - Markit (2009). "ISDA CDS Standard Model" and "The CDS Big Bang."
    - Hull, J.C. and White, A. (2000). "Valuing Credit Default Swaps I: No Counterparty
      Default Risk." The Journal of Derivatives, 8(1), 29-40.

    Referenced against:
    - QuantLib, CreditDefaultSwap: https://www.quantlib.org/reference/class_quant_lib_1_1_credit_default_swap.html
    - QuantLib, IsdaCdsEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_isda_cds_engine.html
    - QuantLib, MidPointCdsEngine: https://github.com/lballabio/QuantLib/blob/v1.44/ql/pricingengines/credit/midpointcdsengine.hpp#L34
    - QuantLib, IntegralCdsEngine: https://github.com/lballabio/QuantLib/blob/v1.44/ql/pricingengines/credit/integralcdsengine.hpp#L33

    Shares similar logic:
    - FinancePy, CDS: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/credit/cds.py#L262

    Args:
        maturity (float): The years until the protection ends.
        zero_curve (pd.Series): The annually compounded zero-coupon rates (decimals),
            indexed by maturity in years.
        hazard_rates (np.ndarray | list | pd.Series): The piecewise flat hazard rates per
            year.
        hazard_times (np.ndarray | list): The maturities up to which each holds.
        coupon (float, optional): The running coupon (contractual spread) per year, as a
            decimal. Defaults to 0.01, the standard coupon of investment grade names.
        recovery_rate (float, optional): The share of the notional recovered on default.
            Defaults to 0.4, the ISDA standard for senior unsecured debt.
        notional (float, optional): The notional, which scales the legs and the
            mark-to-market. Defaults to 1.
        frequency (int, optional): The number of premium payments per year. Defaults to 4.

    Returns:
        pd.Series: The Par Spread, RPV01 (clean, in years), Full RPV01, Protection Leg,
        Premium Leg (c * Full RPV01), Accrued Premium, Upfront (a decimal of the notional)
        and Mark-to-Market. The legs, the accrued premium and the mark-to-market are in
        units of the notional.

    Raises:
        ValueError: If the maturity is not positive or the recovery rate is not between 0
            and 1.
    """
    if not maturity > 0:
        raise ValueError(
            f"The maturity of a CDS must be positive, received {maturity!r}."
        )
    if not 0 <= recovery_rate < 1:
        raise ValueError("The recovery rate must lie between 0 and 1.")

    ends = bond_model.get_cash_flow_periods(maturity, frequency) / frequency
    starts = ends - 1 / frequency
    accrual_starts = np.maximum(starts, 0.0)

    risky_discount = discount_factors(zero_curve, ends) * get_survival_probability(
        hazard_rates, hazard_times, ends
    )
    # Integrated per premium period, so the premium dates are part of the grid.
    default_value, accrual_on_default = np.sum(
        [
            get_default_integrals(
                zero_curve, hazard_rates, hazard_times, begin, end, accrual_start=start
            )
            for begin, end, start in zip(accrual_starts, ends, starts)
        ],
        axis=0,
    )
    full_rpv01 = ACCRUAL_PER_YEAR * float(
        np.sum((ends - starts) * risky_discount) + accrual_on_default
    )
    accrued = ACCRUAL_PER_YEAR * float(-starts[0]) if starts[0] < 0 else 0.0
    rpv01 = full_rpv01 - accrued
    protection = (1 - recovery_rate) * default_value

    return pd.Series(
        {
            "Par Spread": protection / rpv01 if rpv01 > 0 else np.nan,
            "RPV01": rpv01,
            "Full RPV01": full_rpv01,
            "Protection Leg": notional * protection,
            "Premium Leg": notional * coupon * full_rpv01,
            "Accrued Premium": notional * coupon * accrued,
            "Upfront": protection - coupon * rpv01,
            "Mark-to-Market": notional * (protection - coupon * full_rpv01),
        }
    )


def get_cds_implied_hazard_rates(
    maturities: np.ndarray | list,
    spreads: np.ndarray | list | pd.Series,
    zero_curve: pd.Series,
    recovery_rate: float = DEFAULT_RECOVERY_RATE,
    frequency: int = 4,
) -> pd.Series:
    """
    Bootstrap a piecewise flat hazard rate curve from the par spreads of credit default
    swaps (O'Kane, 2008, ch. 7): the default intensity of the reference entity over time
    that the CDS market prices in.

    The hazard rate up to each CDS maturity T_i is solved in maturity order, with the
    earlier ones fixed, so that the par spread of that CDS (see `get_credit_default_swap`)
    equals its quote. The credit triangle λ ≈ s / (1 - R) is the first-order answer.

    The formula is as follows:

    - Solve λ_i: Protection Leg(T_i; λ_1, ..., λ_i) / RPV01(T_i; λ_1, ..., λ_i) = s_i
    - Q(t) = exp(-SUM(λ_j * (min(t, T_j) - T_(j-1))^+))

    For more information, see the following sources:

    - O'Kane, D. (2008). "Modelling Single-name and Multi-name Credit Derivatives." Wiley,
      ch. 7.
    - Markit (2009). "ISDA CDS Standard Model."

    Referenced against:
    - QuantLib, PiecewiseFlatHazardRate: https://www.quantlib.org/reference/class_quant_lib_1_1_piecewise_default_curve.html
    - QuantLib, SpreadCdsHelper: https://www.quantlib.org/reference/class_quant_lib_1_1_spread_cds_helper.html
    - FinancePy, CDSCurve: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/market/curves/cds_curve.py#L45

    Args:
        maturities (np.ndarray | list): The CDS maturities in years, increasing.
        spreads (np.ndarray | list | pd.Series): The par spread of each CDS, as decimals.
        zero_curve (pd.Series): The annually compounded zero-coupon rates (decimals),
            indexed by maturity in years.
        recovery_rate (float, optional): The share of the notional recovered on default.
            Defaults to 0.4.
        frequency (int, optional): The number of premium payments per year. Defaults to 4.

    Returns:
        pd.Series: The hazard rate (continuous, per year) up to each maturity.

    Raises:
        ValueError: If the maturities do not increase, a spread is negative or missing, or a
            spread is so far below the ones before it that it implies a negative hazard rate.
    """
    maturities = np.asarray(maturities, dtype=float)
    spreads = np.asarray(spreads, dtype=float)
    if np.any(np.diff(maturities) <= 0) or not np.all(np.isfinite(spreads)):
        raise ValueError(
            "The maturities must increase and every maturity needs a spread."
        )
    if np.any(spreads < 0):
        raise ValueError("A negative spread implies no default risk to bootstrap.")

    hazard_rates = np.zeros(len(maturities))
    for position, spread in enumerate(spreads):

        def spread_error(hazard_rate: float, position=position, spread=spread):
            hazard_rates[position] = hazard_rate
            return (
                get_credit_default_swap(
                    maturities[position],
                    zero_curve,
                    hazard_rates[: position + 1],
                    maturities[: position + 1],
                    recovery_rate=recovery_rate,
                    frequency=frequency,
                )["Par Spread"]
                - spread
            )

        if spread == 0:
            hazard_rates[position] = 0.0
            continue
        # The par spread rises with the hazard rate, so a spread below that of a hazard rate
        # of zero after the earlier ones needs a negative hazard rate (survival rising with
        # time), which a steeply inverted curve can ask for.
        if spread_error(1e-12) > 0:
            raise ValueError(
                f"The spread of maturity {maturities[position]:g} is too low for the spreads "
                "before it: it implies a negative hazard rate up to that maturity."
            )
        if spread_error(MAXIMUM_HAZARD_RATE) < 0:
            raise ValueError(
                f"The spread of maturity {maturities[position]:g} implies a hazard rate above "
                f"{MAXIMUM_HAZARD_RATE:g} a year."
            )
        hazard_rates[position] = brentq(
            spread_error, 1e-12, MAXIMUM_HAZARD_RATE, xtol=1e-14
        )

    return pd.Series(hazard_rates, index=maturities, name="Hazard Rate")


CDS_OPTION_TYPES = ("payer", "receiver")


def get_cds_option(
    option_expiry: float,
    maturity: float,
    zero_curve: pd.Series,
    hazard_rates: np.ndarray | list | pd.Series,
    hazard_times: np.ndarray | list,
    strike_spread: float,
    volatility: float,
    recovery_rate: float = DEFAULT_RECOVERY_RATE,
    notional: float = 1.0,
    frequency: int = 4,
    option_type: str = "payer",
    knockout: bool = True,
) -> pd.Series:
    """
    Calculate the value of an option on a credit default swap (a CDS swaption): what the
    right to buy or sell protection at the strike spread at expiry is worth, with Black's
    formula on the forward CDS spread, discounted with the forward risky annuity
    (Schönbucher, 2003; O'Kane, 2008, ch. 9).

    A payer option is the right to buy protection from the expiry to the maturity at the
    strike spread; a receiver option the right to sell it. The forward CDS starts at the
    expiry, so its risky annuity is zero when the name defaults before then: under the
    survival measure that uses this annuity as numeraire, the forward spread is lognormal and
    the option is priced with Black's formula. A knockout option is cancelled by a default
    before the expiry, as single-name CDS options trade. A payer option that does not knock
    out also receives the front-end protection: (1 - R) at the expiry when the name defaulted
    before it.

    The formula is as follows:

    - Forward Protection = (1 - R) * ∫_Te^T P(t) (-dQ(t))
    - Forward RPV01 = SUM(τ_k P(t_k) Q(t_k)) + 365 / 360 * SUM(∫ (t - t_(k-1)) P(t) (-dQ(t))),
      over the premium periods from the expiry Te to the maturity T
    - Forward Spread S_f = Forward Protection / Forward RPV01
    - Payer = N RPV01_f [S_f N(d_1) - K N(d_2)], Receiver = N RPV01_f [K N(-d_2) - S_f N(-d_1)]
    - d_1 = (ln(S_f / K) + σ² Te / 2) / (σ √Te), d_2 = d_1 - σ √Te
    - Front-End Protection = N (1 - R) P(Te) (1 - Q(Te)), added to a payer that does not
      knock out

    where P(t) = (1 + r(t))^(-t) is the discount factor of the annually compounded zero rate,
    Q(t) the survival probability of the piecewise flat hazard rate curve, R the recovery
    rate, K the strike spread and σ the lognormal volatility of the forward spread. The
    premium dates are counted back from the maturity in periods of 1 / frequency years; the
    first period accrues from the expiry. The premium accrues on ACT/360 (τ_k = (t_k -
    t_(k-1)) * 365 / 360) and the legs are the exact integrals of `get_credit_default_swap`.

    For more information, see the following sources:

    - Schönbucher, P.J. (2003). "A Note on Survival Measures and the Pricing of Options on
      Credit Default Swaps." ETH Zürich working paper.
    - O'Kane, D. (2008). "Modelling Single-name and Multi-name Credit Derivatives." Wiley,
      ch. 9.
    - Hull, J.C. and White, A. (2003). "The Valuation of Credit Default Swap Options."
      Journal of Derivatives, 10(3), 40-50.

    QuantLib takes the forward spread and annuity from the engine of the underlying CDS; with its
    IntegralCdsEngine the values agree to within a few hundredths of a percent, the
    difference of its daily integration steps.

    Referenced against:
    - QuantLib, CdsOption: https://www.quantlib.org/reference/class_quant_lib_1_1_cds_option.html
    - QuantLib, BlackCdsOptionEngine: https://www.quantlib.org/reference/class_quant_lib_1_1_black_cds_option_engine.html

    Shares similar logic:
    - FinancePy, CDSOption: https://github.com/domokane/FinancePy/blob/V1.1.2/financepy/products/credit/cds_option.py#L41

    Args:
        option_expiry (float): The years until the option expires and the forward CDS starts.
        maturity (float): The years until the protection of the CDS ends.
        zero_curve (pd.Series): The annually compounded zero-coupon rates (decimals),
            indexed by maturity in years.
        hazard_rates (np.ndarray | list | pd.Series): The piecewise flat hazard rates per
            year, such as those of `get_cds_implied_hazard_rates`.
        hazard_times (np.ndarray | list): The maturities up to which each holds.
        strike_spread (float): The strike spread K per year, as a decimal.
        volatility (float): The lognormal volatility of the forward spread per year.
        recovery_rate (float, optional): The share of the notional recovered on default.
            Defaults to 0.4.
        notional (float, optional): The notional. Defaults to 1.
        frequency (int, optional): The number of premium payments per year. Defaults to 4.
        option_type (str, optional): The option. Options are: "payer" (the right to buy
            protection), "receiver" (the right to sell it). Defaults to "payer".
        knockout (bool, optional): Whether a default before the expiry cancels the option.
            Defaults to True.

    Returns:
        pd.Series: The Value (in the currency of the notional), the Forward Spread
        (decimal), the Forward RPV01 (in years, per unit of notional) and the Front-End
        Protection included in the value.

    Raises:
        ValueError: If the option type is unknown, the expiry is not before the maturity,
            the strike spread is not positive or the recovery rate is not between 0 and 1.

    As an example:

    ```python
    import pandas as pd

    from financetoolkit.fixedincome.credit import cds_model

    zero_curve = pd.Series([0.035, 0.037, 0.040], index=[1, 2, 5])

    cds_model.get_cds_option(
        option_expiry=0.5, maturity=5.5, zero_curve=zero_curve, hazard_rates=[0.02],
        hazard_times=[10], strike_spread=0.012, volatility=0.4, notional=10_000_000,
    )
    ```

    Which returns:

    ```
    Value                   54968.924412
    Forward Spread              0.011894
    Forward RPV01               4.256598
    Front-End Protection        0.000000
    dtype: float64
    ```
    """
    check_option("option_type", option_type, CDS_OPTION_TYPES)
    if not 0 < option_expiry < maturity:
        raise ValueError(
            f"The option expires ({option_expiry}) after today and before the maturity "
            f"({maturity}) of the CDS."
        )
    if strike_spread <= 0:
        raise ValueError("The strike spread of a CDS option must be positive.")
    if not 0 <= recovery_rate < 1:
        raise ValueError("The recovery rate must lie between 0 and 1.")

    ends = (
        option_expiry
        + bond_model.get_cash_flow_periods(maturity - option_expiry, frequency)
        / frequency
    )
    starts = np.maximum(ends - 1 / frequency, option_expiry)

    risky_discount = discount_factors(zero_curve, ends) * get_survival_probability(
        hazard_rates, hazard_times, ends
    )
    default_value, accrual_on_default = np.sum(
        [
            get_default_integrals(zero_curve, hazard_rates, hazard_times, begin, end)
            for begin, end in zip(starts, ends)
        ],
        axis=0,
    )
    forward_rpv01 = ACCRUAL_PER_YEAR * float(
        np.sum((ends - starts) * risky_discount) + accrual_on_default
    )
    forward_spread = (1 - recovery_rate) * default_value / forward_rpv01

    value = forward_rpv01 * derivative_model._black_formula(
        forward_spread,
        strike_spread,
        volatility,
        option_expiry,
        option_type == "payer",
    )
    front_end_protection = 0.0
    if option_type == "payer" and not knockout:
        front_end_protection = (1 - recovery_rate) * float(
            discount_factors(zero_curve, option_expiry)
            * (1 - get_survival_probability(hazard_rates, hazard_times, option_expiry))
        )

    return pd.Series(
        {
            "Value": notional * (value + front_end_protection),
            "Forward Spread": forward_spread,
            "Forward RPV01": forward_rpv01,
            "Front-End Protection": notional * front_end_protection,
        }
    )
