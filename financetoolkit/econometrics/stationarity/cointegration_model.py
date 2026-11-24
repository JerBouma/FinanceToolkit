"""Cointegration Model"""

__docformat__ = "google"

import warnings

import numpy as np
import pandas as pd

from financetoolkit.econometrics.helpers import import_optional
from financetoolkit.econometrics.stationarity.unitroot_model import (
    MINIMUM_TEST_OBSERVATIONS,
    default_max_lag,
    get_adf_regression,
)
from financetoolkit.utilities.error_model import InvalidArgumentError

JOHANSEN_MINIMUM_NUMBER_OF_SERIES = 2
# The observations the Johansen test needs beyond the parameters of its VECM.
MINIMUM_JOHANSEN_FREEDOM = 3

# The conventional 5% significance level used for the "Cointegrated (5%)" flag.
SIGNIFICANCE_LEVEL = 0.05

# The R-squared of the cointegrating regression from which statsmodels' coint treats the two
# series as (almost) perfectly collinear: 1 - 100 x the square root of machine precision.
COLLINEAR_R_SQUARED = 1 - 100 * np.sqrt(np.finfo(float).eps)


def get_engle_granger_cointegration(
    series_a: pd.Series, series_b: pd.Series, max_lag: int | None = None
) -> pd.Series:
    """
    Calculate the Engle-Granger test: whether two series are cointegrated, held together by a
    long-run equilibrium so that the spread between them reverts.

    The test gives the results of statsmodels' coint, with the lag search of the ADF test on
    the residuals from one decomposition (see `unitroot_model.get_adf_regression`).

    Two individually non-stationary series (e.g. two stock prices, each following a
    random walk) are cointegrated if some linear combination of them is stationary,
    i.e. they share a long-run equilibrium relationship even though each wanders on
    its own in the short run. This is the classic statistical foundation for
    pairs-trading: if two series are cointegrated, deviations of the spread from its
    equilibrium level tend to revert, making the spread itself tradeable.

    The test proceeds in two steps:

    1. Regress `series_a` on `series_b` (plus a constant): `series_a = alpha + beta * series_b + residuals`.
    2. Run an Augmented Dickey-Fuller test (with no constant, since OLS residuals
       already have zero mean) on the regression residuals.

    The null hypothesis is that the two series are NOT cointegrated (the residuals
    have a unit root). Because the residuals come from an estimated regression rather
    than being observed directly, the test statistic must be compared against
    Engle-Granger specific critical values, which are more negative than plain ADF
    critical values.

    For more information about the method, see the following paper:

    - Engle, R.F. and Granger, C.W.J. (1987). "Co-integration and Error Correction:
    Representation, Estimation, and Testing." Econometrica, 55(2), 251-276.

    Also known as: EG test, residual-based cointegration test.

    Referenced against:
    - statsmodels, coint: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html

    Args:
        series_a (pd.Series): The dependent series.
        series_b (pd.Series): The independent series.
        max_lag (int, optional): The maximum number of lagged differences to consider in the
        underlying ADF test on the residuals. Defaults to the Schwert (1989) rule of thumb,
        capped at (n - 5) // 3 for short samples.

    Returns:
        pd.Series: The Engle-Granger statistic, its p-value, the 1%/5%/10% critical values,
        and whether cointegration is found at the 5% level.
    """
    aligned = pd.concat([series_a, series_b], axis=1, join="inner").dropna()
    dependent = aligned.iloc[:, 0].to_numpy()
    independent = aligned.iloc[:, 1].to_numpy()
    stattools = import_optional(
        "statsmodels.tsa.stattools", "The Engle-Granger cointegration test"
    )
    adfvalues = import_optional(
        "statsmodels.tsa.adfvalues", "The Engle-Granger cointegration test"
    )

    try:
        if len(dependent) < MINIMUM_TEST_OBSERVATIONS:
            raise ValueError("Too few observations for the test.")

        maximum = max_lag if max_lag is not None else default_max_lag(len(dependent))

        # The steps of statsmodels' coint, with the lag search of the ADF test on the
        # residuals from one decomposition (see get_adf_regression), which gives the same
        # lag and statistic. Collinear series and the cases get_adf_regression leaves to
        # statsmodels go through coint itself.
        design = np.column_stack([independent, np.ones(len(independent))])
        coefficients = np.linalg.lstsq(design, dependent, rcond=None)[0]
        residuals = dependent - design @ coefficients
        deviations = dependent - dependent.mean()
        r_squared = 1 - (residuals @ residuals) / (deviations @ deviations)
        fast = (
            get_adf_regression(residuals, maximum, "n")
            if r_squared < COLLINEAR_R_SQUARED
            else None
        )

        if fast is not None:
            eg_statistic = fast[0]
            p_value = adfvalues.mackinnonp(eg_statistic, regression="c", N=2)
            # One observation fewer, as statsmodels (and Stata's egranger) does.
            critical_values = adfvalues.mackinnoncrit(
                N=2, regression="c", nobs=len(dependent) - 1
            )
        else:
            eg_statistic, p_value, critical_values = stattools.coint(
                dependent, independent, trend="c", maxlag=maximum, autolag="aic"
            )
    except (ValueError, np.linalg.LinAlgError):
        return pd.Series(
            {
                "Engle-Granger Statistic": np.nan,
                "P-Value": np.nan,
                "Critical Value 1%": np.nan,
                "Critical Value 5%": np.nan,
                "Critical Value 10%": np.nan,
                "Cointegrated (5%)": False,
            }
        )

    return pd.Series(
        {
            "Engle-Granger Statistic": eg_statistic,
            "P-Value": p_value,
            "Critical Value 1%": critical_values[0],
            "Critical Value 5%": critical_values[1],
            "Critical Value 10%": critical_values[2],
            "Cointegrated (5%)": bool(p_value < SIGNIFICANCE_LEVEL),
        }
    )


def get_johansen_cointegration(
    data: pd.DataFrame, det_order: int = 0, lags: int = 1
) -> pd.DataFrame:
    """
    Calculate the Johansen test: how many long-run equilibrium relationships (cointegrating
    relations) hold two or more series together.

    The test runs statsmodels' coint_johansen. The Engle-Granger test (see
    `cointegration_model.get_engle_granger_cointegration`) only handles two series and
    imposes an arbitrary normalization (which series is "dependent"). Johansen's test
    generalizes this to `N >= 2` series at once by testing the rank of the long-run
    coefficient matrix `Pi` in the Vector Error Correction Model (VECM) representation of a
    VAR(p):

    - Delta_Y_t = Pi * Y_(t-1) + SUM(Gamma_i * Delta_Y_(t-i)) + mu + e_t

    `Pi`'s rank `r` equals the number of independent cointegrating relationships among
    the `N` series: `r = 0` means no cointegration (each series wanders independently,
    or their differences do), `r = N` means the whole system is already stationary in
    levels, and `0 < r < N` means `r` stationary linear combinations exist among `N`
    individually non-stationary series.

    Two statistics are computed for every candidate rank `r = 0, ..., N-1`, each
    compared against its own Johansen-specific asymptotic critical values (a standard
    t/F table does not apply):

    - Trace statistic: tests H0: rank <= r against a general alternative.
    - Maximum-eigenvalue statistic: tests H0: rank = r against H1: rank = r + 1.

    Reading down either column and finding the first row that is NOT rejected gives
    the estimated cointegration rank.

    For more information about the method, see the following papers:

    - Johansen, S. (1988). "Statistical Analysis of Cointegration Vectors." Journal of
    Economic Dynamics and Control, 12(2-3), 231-254.
    - Johansen, S. (1991). "Estimation and Hypothesis Testing of Cointegration Vectors
    in Gaussian Vector Autoregressive Models." Econometrica, 59(6), 1551-1580.
    - MacKinnon, J.G., Haug, A.A., & Michelis, L. (1999). "Numerical Distribution
    Functions of Likelihood Ratio Tests for Cointegration." Journal of Applied
    Econometrics, 14(5), 563-577.

    Also known as: Johansen test, Johansen procedure, VECM rank test.

    Args:
        data (pd.DataFrame): A DataFrame with one column per series (e.g. price levels of
        several assets), at least 2 columns.
        det_order (int, optional): Which deterministic term to include: -1 (none,
        Johansen's case 1), 0 (an unrestricted constant, case 3: the levels may drift
        and the cointegrating relations have a nonzero mean) or 1 (a linear trend: the
        levels are linearly detrended before the test, evaluated against the case 5
        critical values of an unrestricted trend). The critical values are those of
        MacKinnon, Haug & Michelis (1999) that `statsmodels` returns. Defaults to 0.
        lags (int, optional): The number of lagged first differences to include as
        short-run dynamics (equivalent to fitting a levels-VAR of order `lags + 1`).
        Defaults to 1.

    Returns:
        pd.DataFrame: One row per candidate rank `r = 0, ..., N-1` (indexed `"r <= 0"`,
        `"r <= 1"`, ...), with the corresponding eigenvalue, trace statistic, max-eigenvalue
        statistic, their 90%/95%/99% critical values, and whether each is rejected at the 5%
        level.

    Raises:
        ValueError: If `data` has fewer than 2 columns, or if `lags` is negative.
    """
    if det_order not in (-1, 0, 1):
        raise InvalidArgumentError(
            "det_order must be one of -1 (none), 0 (constant) or 1 (constant and trend)."
        )

    if lags < 0:
        raise ValueError("lags must be a non-negative integer.")

    aligned = data.dropna()
    number_of_series = aligned.shape[1]

    if number_of_series < JOHANSEN_MINIMUM_NUMBER_OF_SERIES:
        raise ValueError(
            "data must have at least 2 columns (series) -- Johansen cointegration "
            "requires at least 2 series."
        )

    nan_row = {
        "Eigenvalue": np.nan,
        "Trace Statistic": np.nan,
        "Trace Critical Value 10%": np.nan,
        "Trace Critical Value 5%": np.nan,
        "Trace Critical Value 1%": np.nan,
        "Reject (Trace, 5%)": False,
        "Max-Eigenvalue Statistic": np.nan,
        "Max-Eigenvalue Critical Value 10%": np.nan,
        "Max-Eigenvalue Critical Value 5%": np.nan,
        "Max-Eigenvalue Critical Value 1%": np.nan,
        "Reject (Max-Eigenvalue, 5%)": False,
    }
    index = [f"r <= {rank}" for rank in range(number_of_series)]

    # The VECM estimates `lags` lagged differences of every series plus the deterministic
    # terms; with too few observations left over the eigenvalues reach 1 and every rank is
    # rejected, so shorter samples return NaN.
    minimum_observations = (
        number_of_series * (lags + 1) + number_of_series + MINIMUM_JOHANSEN_FREEDOM
    )
    vecm = import_optional(
        "statsmodels.tsa.vector_ar.vecm", "The Johansen cointegration test"
    )

    try:
        if len(aligned) < minimum_observations:
            raise ValueError("Too few observations for the test.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = vecm.coint_johansen(aligned.to_numpy(), det_order, lags)
    except (ValueError, np.linalg.LinAlgError):
        return pd.DataFrame([nan_row] * number_of_series, index=index)

    eigenvalues = np.real(result.eig)
    trace_statistics = np.real(result.lr1)
    max_eigenvalue_statistics = np.real(result.lr2)

    rows = []
    for rank in range(number_of_series):
        rows.append(
            {
                "Eigenvalue": eigenvalues[rank],
                "Trace Statistic": trace_statistics[rank],
                "Trace Critical Value 10%": result.cvt[rank, 0],
                "Trace Critical Value 5%": result.cvt[rank, 1],
                "Trace Critical Value 1%": result.cvt[rank, 2],
                "Reject (Trace, 5%)": bool(
                    trace_statistics[rank] > result.cvt[rank, 1]
                ),
                "Max-Eigenvalue Statistic": max_eigenvalue_statistics[rank],
                "Max-Eigenvalue Critical Value 10%": result.cvm[rank, 0],
                "Max-Eigenvalue Critical Value 5%": result.cvm[rank, 1],
                "Max-Eigenvalue Critical Value 1%": result.cvm[rank, 2],
                "Reject (Max-Eigenvalue, 5%)": bool(
                    max_eigenvalue_statistics[rank] > result.cvm[rank, 1]
                ),
            }
        )

    return pd.DataFrame(rows, index=index)
