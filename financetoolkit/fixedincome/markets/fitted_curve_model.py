"""Fitted Bond Curve Model"""

__docformat__ = "google"

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.optimize import brentq, least_squares

from financetoolkit.fixedincome.curves import curve_model
from financetoolkit.utilities.error_model import check_option

METHODS = ("nelson_siegel", "svensson", "exponential_spline", "cubic_spline")

# The exponential spline has nine exponentials e^(-k κ t), k = 1 to 9, as QuantLib's
# ExponentialSplinesFitting by default. Given κ the fit is linear, so κ is searched on a
# logarithmic grid and refined between the grid points around the best one, where the
# derivative of the cost to κ is zero.
EXPONENTIALS = 9
KAPPA_GRID = np.geomspace(0.01, 2.0, 120)

# Singular values below this share of the largest are left out of the linear least
# squares: the exponentials are nearly collinear, and the directions they leave
# undetermined would otherwise differ between platforms without changing the fit.
RELATIVE_CONDITION = 1e-12

# The bounds of the Nelson-Siegel and Svensson fits: betas within ±100%, the first decay
# from a third of the shortest maturity (as get_nelson_siegel_svensson keeps it) to 30
# years and the second at 1.5 to 200 times the first. The level is kept at or above the
# lower of zero and the yield of the longest bond, as get_nelson_siegel_svensson keeps it.
BETA_BOUND = 1.0
MAXIMUM_TAU = 30.0
TAU_RATIO = (1.5, 200.0)
# The decays every fit also starts from, besides the fit to the bonds' yields, so a local
# optimum near one start does not decide the result.
START_TAUS = (0.5, 1.0, 2.0, 5.0)
# The fit stops when the cost changes by less than this share, and counts as well defined
# when the smallest singular value of its Jacobian is at least this share of the largest.
FIT_TOLERANCE = 1e-12
IDENTIFIED = 1e-8
# Below this, t / τ counts as zero in the derivative of the slope loading.
NEAR_ZERO = 1e-8


def _parameter_names(method: str) -> list[str]:
    """The parameters of the Nelson-Siegel or Svensson model."""
    if method == "nelson_siegel":
        return ["Beta 0", "Beta 1", "Beta 2", "Tau 1"]
    return ["Beta 0", "Beta 1", "Beta 2", "Beta 3", "Tau 1", "Tau 2"]


def get_fitted_discount_factors(
    parameters: pd.Series, years: np.ndarray | list | float, method: str
) -> np.ndarray:
    """
    Calculate the discount factors of a fitted bond curve: what a payment at each time is
    worth today on the curve, see `fit_bond_curve`.

    Args:
        parameters (pd.Series): The parameters `fit_bond_curve` returns.
        years (np.ndarray | list | float): The times, in years.
        method (str): The method the parameters were fitted with.

    Returns:
        np.ndarray: The discount factor at each time.

    Raises:
        ValueError: If the method does not exist.
    """
    years = np.asarray(years, dtype=float)

    if method in ("nelson_siegel", "svensson"):
        zero_rates = curve_model.nelson_siegel_svensson_yield(
            years.ravel(),
            parameters["Beta 0"],
            parameters["Beta 1"],
            parameters["Beta 2"],
            parameters.get("Beta 3", 0.0),
            parameters["Tau 1"],
            parameters.get("Tau 2", 1.0),
        ).reshape(years.shape)
        return np.exp(-zero_rates * years)

    if method == "exponential_spline":
        kappa = parameters["Kappa"]
        coefficients = parameters[
            [f"Coefficient {k}" for k in range(1, EXPONENTIALS + 1)]
        ]
        powers = np.arange(1, EXPONENTIALS + 1)
        return np.exp(-kappa * years[..., None] * powers) @ coefficients.to_numpy(
            dtype=float
        )

    if method == "cubic_spline":
        knots = parameters.filter(like="Knot").to_numpy(dtype=float)
        coefficients = parameters.filter(like="Coefficient").to_numpy(dtype=float)
        spline = BSpline(np.r_[[knots[0]] * 3, knots, [knots[-1]] * 3], coefficients, 3)
        last = knots[-1]
        # Beyond the longest bond the zero rate is held at that of the longest bond.
        inside = spline(np.minimum(years, last))
        return np.where(years > last, spline(last) ** (years / last), inside)

    raise ValueError(
        f"The method {method!r} does not exist. Choose from {', '.join(METHODS)}."
    )


def get_fitted_curve(
    parameters: pd.Series, years: np.ndarray | list | float, method: str
) -> pd.DataFrame:
    """
    Calculate the zero rates, discount factors and instantaneous forward rates of a fitted
    bond curve: the term structure the bond prices imply, see `fit_bond_curve`.

    The formula is as follows:

    - Zero Rate = d(t)^(-1 / t) - 1
    - Instantaneous Forward Rate = -d ln d(t) / dt, by a central difference over 0.0001 years

    Args:
        parameters (pd.Series): The parameters `fit_bond_curve` returns.
        years (np.ndarray | list | float): The maturities, in years, above 0.
        method (str): The method the parameters were fitted with.

    Returns:
        pd.DataFrame: The Zero Rate (annually compounded), the Discount Factor and the
        Instantaneous Forward Rate (continuously compounded) by maturity in years.
    """
    years = np.atleast_1d(np.asarray(years, dtype=float))
    if np.any(years <= 0):
        raise ValueError("The maturities must be longer than zero.")
    step = np.minimum(1e-4, years / 2)
    discount = get_fitted_discount_factors(parameters, years, method)
    forwards = (
        np.log(get_fitted_discount_factors(parameters, years - step, method))
        - np.log(get_fitted_discount_factors(parameters, years + step, method))
    ) / (2 * step)

    return pd.DataFrame(
        {
            "Zero Rate": discount ** (-1 / years) - 1,
            "Discount Factor": discount,
            "Instantaneous Forward Rate": forwards,
        },
        index=pd.Index(years, name="Maturity"),
    )


def get_fitted_bond_prices(
    parameters: pd.Series,
    cash_flow_times: np.ndarray,
    cash_flows: np.ndarray,
    method: str,
) -> np.ndarray:
    """
    Calculate the prices (with accrued interest) of bonds on a fitted curve: what their cash
    flows are worth discounted with `get_fitted_discount_factors`.

    Args:
        parameters (pd.Series): The parameters `fit_bond_curve` returns.
        cash_flow_times (np.ndarray): The time of every cash flow in years, a row per bond
            (padded with zeros).
        cash_flows (np.ndarray): The cash flows, in the same layout (padded with zeros).
        method (str): The method the parameters were fitted with.

    Returns:
        np.ndarray: The price of every bond.
    """
    discount = get_fitted_discount_factors(parameters, cash_flow_times, method)
    return np.sum(np.asarray(cash_flows, dtype=float) * discount, axis=1)


def _linear_fit(
    fixed: np.ndarray,
    basis: np.ndarray,
    prices: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    The weighted least squares coefficients a of prices ≈ fixed + basis a, and the cost
    Σ w (fixed + basis a - prices)².
    """
    root = np.sqrt(weights)
    coefficients, *_ = np.linalg.lstsq(
        basis * root[:, None], (prices - fixed) * root, rcond=RELATIVE_CONDITION
    )
    errors = fixed + basis @ coefficients - prices
    return coefficients, float(np.sum(weights * errors**2))


def _bond_yields(
    cash_flow_times: np.ndarray, cash_flows: np.ndarray, prices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The continuously compounded yield and the Macaulay duration of every bond."""
    yields = np.full(len(prices), 0.04)
    for _ in range(50):
        discounted = cash_flows * np.exp(-yields[:, None] * cash_flow_times)
        value = discounted.sum(axis=1)
        slope = -(cash_flow_times * discounted).sum(axis=1)
        step = (value - prices) / slope
        yields = yields - step
        if np.max(np.abs(step)) < FIT_TOLERANCE:
            break
    discounted = cash_flows * np.exp(-yields[:, None] * cash_flow_times)
    durations = (cash_flow_times * discounted).sum(axis=1) / discounted.sum(axis=1)
    return yields, durations


def _fit_nelson_siegel(
    cash_flow_times: np.ndarray,
    cash_flows: np.ndarray,
    prices: np.ndarray,
    weights: np.ndarray,
    method: str,
) -> pd.Series:
    """
    The Nelson-Siegel or Svensson fit to bond prices: a bounded nonlinear least squares
    (trust region reflective) from several fixed starts, the best of which is kept.
    """
    svensson = method == "svensson"
    names = _parameter_names(method)
    yields, durations = _bond_yields(cash_flow_times, cash_flows, prices)
    shortest = float(cash_flow_times.max(axis=1).min())
    floor = min(0.0, float(yields[np.argmax(cash_flow_times.max(axis=1))]))
    lowest_tau = max(shortest * curve_model.SHORTEST_MATURITY_RATIO, 0.01)
    root = np.sqrt(weights)

    # x = betas, ln τ1 and, for Svensson, ln(τ2 / τ1).
    betas = 4 if svensson else 3
    lower = [floor] + [-BETA_BOUND] * (betas - 1) + [np.log(lowest_tau)]
    upper = [BETA_BOUND] * betas + [np.log(MAXIMUM_TAU)]
    if svensson:
        lower.append(np.log(TAU_RATIO[0]))
        upper.append(np.log(TAU_RATIO[1]))
    lower, upper = np.array(lower), np.array(upper)

    def unpack(x: np.ndarray) -> pd.Series:
        values = list(x[:betas]) + [np.exp(x[betas])]
        if svensson:
            values.append(np.exp(x[betas]) * np.exp(x[betas + 1]))
        return pd.Series(values, index=names, dtype=float)

    def residuals(x: np.ndarray) -> np.ndarray:
        fitted = get_fitted_bond_prices(unpack(x), cash_flow_times, cash_flows, method)
        return root * (fitted - prices)

    def slope_derivative(x: np.ndarray) -> np.ndarray:
        # The derivative of (1 - e^(-x)) / x, which is -1/2 at x = 0.
        safe = np.where(x > IDENTIFIED, x, 1.0)
        return np.where(
            x > IDENTIFIED, (np.exp(-safe) * (1 + safe) - 1) / safe**2, -0.5
        )

    def jacobian(x: np.ndarray) -> np.ndarray:
        # dP_i / dx = Σ_k CF_ik d(t_ik) (-t_ik) dy(t_ik) / dx, with the loadings of the
        # betas and the derivatives of the loadings to the log decays.
        parameters = unpack(x)
        tau_1 = parameters["Tau 1"]
        x1 = cash_flow_times / tau_1
        slope_1 = curve_model.nelson_siegel_svensson_yield(
            cash_flow_times.ravel(), 0.0, 1.0, 0.0, 0.0, tau_1, 1.0
        ).reshape(cash_flow_times.shape)
        derivatives = [np.ones_like(x1), slope_1, slope_1 - np.exp(-x1)]
        if svensson:
            derivatives.append(
                curve_model.nelson_siegel_svensson_yield(
                    cash_flow_times.ravel(),
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    tau_1,
                    parameters["Tau 2"],
                ).reshape(cash_flow_times.shape)
            )
        first = -x1 * (
            parameters["Beta 1"] * slope_derivative(x1)
            + parameters["Beta 2"] * (slope_derivative(x1) + np.exp(-x1))
        )
        if svensson:
            x2 = cash_flow_times / parameters["Tau 2"]
            second = -x2 * parameters["Beta 3"] * (slope_derivative(x2) + np.exp(-x2))
            # τ2 = τ1 ρ, so ln τ1 moves both decays.
            derivatives += [first + second, second]
        else:
            derivatives.append(first)
        weighted = (
            -cash_flows
            * cash_flow_times
            * get_fitted_discount_factors(parameters, cash_flow_times, method)
        )
        return root[:, None] * np.column_stack(
            [np.sum(weighted * derivative, axis=1) for derivative in derivatives]
        )

    # The starts: the model fitted to the bonds' yields at their durations, and the betas
    # of that fit with each of a few fixed decays.
    start = curve_model.fit_nelson_siegel_svensson(durations, yields, svensson)
    first = [start[name] for name in names if name.startswith("Beta")]
    decays = [start["Tau 1"], start.get("Tau 2", 1.0)]
    starts = [
        first
        + [np.log(decays[0])]
        + ([np.log(decays[1] / decays[0])] if svensson else [])
    ]
    for tau in START_TAUS:
        starts.append(first + [np.log(tau)] + ([np.log(3.0)] if svensson else []))

    best = None
    for values in starts:
        x0 = np.clip(np.array(values, dtype=float), lower + 1e-12, upper - 1e-12)
        result = least_squares(
            residuals,
            x0,
            jac=jacobian,
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            ftol=FIT_TOLERANCE,
            xtol=FIT_TOLERANCE,
            gtol=FIT_TOLERANCE,
            max_nfev=2000,
        )
        if best is None or result.cost < best.cost - 1e-15:
            best = result

    # A Jacobian of (nearly) deficient rank means a ridge of equally good parameters, on
    # which the optimizer would stop at a point that depends on rounding.
    singular = np.linalg.svd(best.jac, compute_uv=False)
    if singular[-1] < IDENTIFIED * singular[0]:
        raise ValueError(
            f"The {method} fit is not identified: its parameters can move along a ridge "
            "without changing the fit. Use the Nelson-Siegel model or a spline."
        )

    return unpack(best.x)


def fit_bond_curve(
    cash_flow_times: np.ndarray | list,
    cash_flows: np.ndarray | list,
    prices: np.ndarray | list,
    weights: np.ndarray | list | None = None,
    method: str = "svensson",
) -> pd.Series:
    """
    Fit a discount curve to the prices of coupon bonds: the smooth term structure closest to
    every bond's price, whose residuals show which bonds trade rich (above the curve) or
    cheap (below it).

    Where a bootstrap needs one instrument per maturity, a fitted curve takes every bond
    there is and smooths out the noise of the individual prices. The fit minimizes the
    weighted squared price errors, with weights inversely proportional to duration, as
    Gürkaynak, Sack and Wright (2007) fit the Federal Reserve's Treasury curve. A price
    error is about the duration times the price times the yield error, so the weights reduce
    the weight of the long bonds' larger price errors, while still giving a basis point of a
    long bond more weight than one of a short bond (equal weight in yield would need 1 /
    D²).

    The formula is as follows:

    - min Σ_i w_i (P_i^model - P_i)², with P_i^model = Σ_k CF_ik d(t_ik) and w_i = 1 / D_i

    where P_i is the price of bond i with accrued interest, CF_ik its cash flows at times
    t_ik, D_i its duration and d(t) the discount function of the method:

    - Nelson-Siegel: d(t) = e^(-y(t) t) with y(t) the Nelson-Siegel (1987) zero rate,
      β0 + β1 (1 - e^(-t/τ1)) / (t/τ1) + β2 [(1 - e^(-t/τ1)) / (t/τ1) - e^(-t/τ1)]
    - Svensson: the same with Svensson's (1994) second curvature,
      + β3 [(1 - e^(-t/τ2)) / (t/τ2) - e^(-t/τ2)]
    - Exponential spline: d(t) = Σ_{k=1}^{9} a_k e^(-k κ t) with Σ a_k = 1, the
      exponential basis of Vasicek and Fong (1982), whose discount function decays
      exponentially at long maturities
    - Cubic spline: d(t) = Σ_j c_j B_j(t), a cubic spline with d(0) = 1 (McCulloch, 1975),
      in the B-spline basis; the number of basis functions is about the square root of the
      number of bonds and the knots lie at quantiles of the maturities, so that as many
      bonds mature between every two knots, as McCulloch places them

    Given the decays the Nelson-Siegel and Svensson prices are not linear in the betas, so
    those fits are a bounded nonlinear least squares (trust region reflective) from fixed
    starts: the model fitted to the bonds' continuously compounded yields at their
    durations, and that fit's betas with first decays of 0.5, 1, 2 and 5 years. The bounds
    are those of `get_nelson_siegel_svensson`: the first decay from a third of the shortest
    maturity to 30 years, the second at least 1.5 times the first and the level at or
    above the lower of zero and the yield of the longest bond. A fit whose Jacobian is
    (nearly) rank deficient raises a ValueError, as its parameters are not identified. The
    exponential and cubic splines are linear in their coefficients, so given κ their fit
    is an exact weighted least squares; κ is searched on a grid and refined to where the
    derivative of the cost to κ is zero (Brent's method on the analytic derivative, as the
    cost is flat in κ and its minimum alone would pin κ down only to about 1e-8). The zero
    rates of the curve are continuously compounded, as the models define them.

    For more information, see the following papers:

    - McCulloch, J.H. (1975). "The Tax-Adjusted Yield Curve." The Journal of Finance,
      30(3), 811-830.
    - Vasicek, O.A. and Fong, H.G. (1982). "Term Structure Modeling Using Exponential
      Splines." The Journal of Finance, 37(2), 339-348.
    - Nelson, C.R. and Siegel, A.F. (1987). "Parsimonious Modeling of Yield Curves." The
      Journal of Business, 60(4), 473-489.
    - Svensson, L.E.O. (1994). "Estimating and Interpreting Forward Interest Rates: Sweden
      1992-1994." NBER Working Paper 4871.
    - Gürkaynak, R.S., Sack, B. and Wright, J.H. (2007). "The U.S. Treasury Yield Curve:
      1961 to the Present." Journal of Monetary Economics, 54(8), 2291-2304.

    Shares similar logic:
    - QuantLib, FittedBondDiscountCurve: https://www.quantlib.org/reference/class_quant_lib_1_1_fitted_bond_discount_curve.html
    - QuantLib, NelsonSiegelFitting (nelson_siegel): https://www.quantlib.org/reference/class_quant_lib_1_1_nelson_siegel_fitting.html
    - QuantLib, SvenssonFitting (svensson): https://www.quantlib.org/reference/class_quant_lib_1_1_svensson_fitting.html
    - QuantLib, ExponentialSplinesFitting (exponential_spline): https://www.quantlib.org/reference/class_quant_lib_1_1_exponential_splines_fitting.html
    - QuantLib, CubicBSplinesFitting (cubic_spline): https://www.quantlib.org/reference/class_quant_lib_1_1_cubic_b_splines_fitting.html

    Args:
        cash_flow_times (np.ndarray | list): The time of every cash flow in years, a row per
            bond (padded with zeros where a bond has fewer cash flows).
        cash_flows (np.ndarray | list): The cash flows per 100 of par, in the same layout
            (padded with zeros).
        prices (np.ndarray | list): The price of every bond with accrued interest, per 100 of
            par.
        weights (np.ndarray | list | None, optional): The weight of every bond's squared
            price error. Defaults to None, one over the Macaulay duration (continuously
            compounded).
        method (str, optional): The discount function. Options are: "nelson_siegel" (the
            Nelson-Siegel zero rate), "svensson" (the Nelson-Siegel-Svensson zero rate),
            "exponential_spline" (the exponential spline of Vasicek and Fong),
            "cubic_spline" (McCulloch's cubic spline). Defaults to "svensson".

    Returns:
        pd.Series: The parameters: Beta 0 to Beta 3 (continuously compounded) and Tau 1 and
        Tau 2 (years) for the Nelson-Siegel models, Kappa and Coefficient 1 to 9 for the
        exponential spline, and the Knots (years) and Coefficients of the cubic spline.

    Raises:
        ValueError: If the method does not exist, there are fewer bonds than parameters or
            the Nelson-Siegel or Svensson fit is not identified.
    """
    check_option("method", method, METHODS)
    cash_flow_times = np.atleast_2d(np.asarray(cash_flow_times, dtype=float))
    cash_flows = np.atleast_2d(np.asarray(cash_flows, dtype=float))
    prices = np.asarray(prices, dtype=float)
    if weights is None:
        _, durations = _bond_yields(cash_flow_times, cash_flows, prices)
        weights = 1 / durations
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    maturities = cash_flow_times.max(axis=1)

    if method in ("nelson_siegel", "svensson"):
        if len(prices) < len(_parameter_names(method)):
            raise ValueError(
                f"The {method} fit needs at least {len(_parameter_names(method))} bonds."
            )
        return _fit_nelson_siegel(cash_flow_times, cash_flows, prices, weights, method)

    if method == "exponential_spline":
        if len(prices) < EXPONENTIALS:
            raise ValueError(
                f"The exponential spline needs at least {EXPONENTIALS} bonds."
            )
        powers = np.arange(1, EXPONENTIALS + 1)

        def fit(kappa: float) -> tuple[np.ndarray, float]:
            # Σ a_k = 1 holds with a_9 = 1 - Σ_{k<9} a_k, which leaves e^(-9 κ t) as the
            # fixed part and e^(-k κ t) - e^(-9 κ t) as the basis.
            terms = np.exp(-kappa * cash_flow_times[..., None] * powers)
            values = np.einsum("bk,bkn->bn", cash_flows, terms)
            return _linear_fit(
                values[:, -1], values[:, :-1] - values[:, -1:], prices, weights
            )

        def slope(kappa: float) -> float:
            # The derivative of the fit's cost to κ with the coefficients held at their
            # optimum (the envelope theorem): 2 Σ w_i e_i dP_i/dκ, with
            # dP_i/dκ = Σ_k CF_ik Σ_j a_j (-j t_ik) e^(-j κ t_ik). Its root locates κ to
            # near machine precision, where the minimum of the cost itself, which is flat
            # in κ, is only found to about the square root of it.
            coefficients, _ = fit(kappa)
            weights_k = np.array([*coefficients, 1 - coefficients.sum()]) * -powers
            terms = cash_flow_times[..., None] * np.exp(
                -kappa * cash_flow_times[..., None] * powers
            )
            derivatives = np.einsum("bk,bkn,n->b", cash_flows, terms, weights_k)
            values = np.einsum(
                "bk,bkn,n->b",
                cash_flows,
                np.exp(-kappa * cash_flow_times[..., None] * powers),
                np.array([*coefficients, 1 - coefficients.sum()]),
            )
            return float(2 * np.sum(weights * (values - prices) * derivatives))

        costs = np.array([fit(kappa)[1] for kappa in KAPPA_GRID])
        index = int(np.argmin(costs))
        kappa = float(KAPPA_GRID[index])
        low = float(KAPPA_GRID[max(index - 1, 0)])
        high = float(KAPPA_GRID[min(index + 1, len(KAPPA_GRID) - 1)])
        if slope(low) < 0 < slope(high):
            kappa = brentq(slope, low, high, xtol=1e-15, rtol=4 * np.finfo(float).eps)
        coefficients, _ = fit(kappa)
        values = [kappa, *coefficients, 1 - coefficients.sum()]
        return pd.Series(
            values,
            index=["Kappa"] + [f"Coefficient {k}" for k in range(1, EXPONENTIALS + 1)],
            dtype=float,
        )

    # The cubic spline: about √n basis functions, so √n - 4 knots inside, at quantiles of
    # the maturities.
    functions = max(int(round(np.sqrt(len(prices)))), 5)
    if len(prices) < functions:
        raise ValueError(f"The cubic spline needs at least {functions} bonds.")
    inner = np.quantile(maturities, np.linspace(0, 1, functions - 2)[1:-1])
    knots = np.r_[0.0, inner, maturities.max()]
    padded = np.r_[[0.0] * 3, knots, [knots[-1]] * 3]
    # Every basis function at every cash flow time; the first is the only one that is not
    # zero at t = 0, where it is one, so d(0) = 1 fixes its coefficient at one.
    basis = BSpline.design_matrix(
        np.clip(cash_flow_times.ravel(), 0.0, knots[-1]), padded, 3
    ).toarray()
    basis = basis.reshape(*cash_flow_times.shape, -1)
    values = np.einsum("bk,bkn->bn", cash_flows, basis)
    coefficients, _ = _linear_fit(values[:, 0], values[:, 1:], prices, weights)

    return pd.Series(
        [*knots, 1.0, *coefficients],
        index=[f"Knot {k}" for k in range(1, len(knots) + 1)]
        + [f"Coefficient {k}" for k in range(1, len(coefficients) + 2)],
        dtype=float,
    )
