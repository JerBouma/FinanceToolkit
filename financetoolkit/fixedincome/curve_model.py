"""Yield Curve Fitting, Bootstrapping and Principal Components Model"""

__docformat__ = "google"

import re

import numpy as np
import pandas as pd

# pylint: disable=too-many-locals

# The decay parameters (in years) the fits search over, on logarithmic grids. For every grid
# point the betas follow exactly from linear least squares; around the best point the search
# then zooms in, on a 7 by 7 grid that shrinks threefold each round, until the decays are
# within about 0.01% of the optimum. The second decay of the Svensson model is kept at least
# 1.5 times the first, as the two curvature factors are indistinguishable otherwise, and the
# first decay at least a third of the shortest maturity: a shorter decay turns the curvature
# into a dummy for the shortest yield, which fits it perfectly and sends the extrapolated
# short end far off. The Nelson-Siegel grid reaches 30 years: a flat or inverted curve with a
# hump at the long end can be fitted best by a decay of 20 years or more, which a grid ending
# at 10 years misses by close to a basis point of fit error.
NELSON_SIEGEL_TAUS = np.geomspace(0.1, 30.0, 240)
SVENSSON_TAUS_1 = np.geomspace(0.1, 5.0, 50)
SVENSSON_TAUS_2 = np.geomspace(0.5, 30.0, 50)
TAU_RATIO = 1.5
SHORTEST_MATURITY_RATIO = 1 / 3
ZOOM_POINTS = 7
ZOOM_ROUNDS = 6
# The length of a maturity unit in years; a number without a unit is a number of years.
MATURITY_UNITS = {"": 1.0, "D": 1 / 365, "W": 7 / 365, "M": 1 / 12, "Y": 1.0}
# The number of curves whose grid errors are computed at once, which bounds the memory used.
CHUNK_SIZE = 250


def maturity_in_years(label: str | float) -> float:
    """
    The maturity of a label such as "3M", "1.5M", "10Y", "52W", "3month" or "0.5" (years),
    in years.

    Args:
        label (str | float): The label, or a number of years.

    Returns:
        float: The maturity in years.
    """
    if isinstance(label, int | float | np.number):
        return float(label)
    match = re.fullmatch(r"\s*([\d.]+)\s*([A-Za-z]*)\s*", str(label))
    unit = match.group(2).upper()[:1] if match else ""
    if not match or unit not in MATURITY_UNITS:
        raise ValueError(
            f"The maturity {label!r} is not a number of days, weeks, months or years, such "
            "as '3M' or '10Y'."
        )
    return float(match.group(1)) * MATURITY_UNITS[unit]


def _design(
    maturities: np.ndarray, tau_1: np.ndarray, tau_2: np.ndarray | None = None
) -> np.ndarray:
    """
    The factor loadings for every pair of decays (first axis), maturity (second axis) and
    factor (third axis): level, slope and curvature, plus the second curvature of Svensson.
    """
    x1 = maturities / np.asarray(tau_1, dtype=float)[..., None]
    slope = _slope(x1)
    columns = [np.ones_like(slope), slope, slope - np.exp(-x1)]
    if tau_2 is not None:
        x2 = maturities / np.asarray(tau_2, dtype=float)[..., None]
        columns.append(_slope(x2) - np.exp(-x2))
    return np.stack(columns, axis=-1)


def _slope(x: np.ndarray) -> np.ndarray:
    """(1 - e^(-x)) / x, which is 1 at a maturity of zero (the instantaneous short rate)."""
    safe = np.where(x > 0, x, 1.0)
    return np.where(x > 0, -np.expm1(-safe) / safe, 1.0)


def nelson_siegel_svensson_yield(
    maturities: np.ndarray | float,
    beta_0: float,
    beta_1: float,
    beta_2: float,
    beta_3: float,
    tau_1: float,
    tau_2: float,
) -> np.ndarray:
    """
    The yield of the Nelson-Siegel-Svensson model at the given maturities:

        y(m) = β0 + β1 (1 - e^(-m/τ1)) / (m/τ1)
               + β2 [(1 - e^(-m/τ1)) / (m/τ1) - e^(-m/τ1)]
               + β3 [(1 - e^(-m/τ2)) / (m/τ2) - e^(-m/τ2)]

    Args:
        maturities (np.ndarray | float): The maturities in years.
        beta_0 (float): The level (the yield at an infinite maturity).
        beta_1 (float): The slope (the short end minus the long end).
        beta_2 (float): The first curvature, whose loading peaks at about 1.8 τ1.
        beta_3 (float): The second curvature, whose loading peaks at about 1.8 τ2; zero for
            the Nelson-Siegel model.
        tau_1 (float): The first decay, in years.
        tau_2 (float): The second decay, in years (irrelevant when beta_3 is zero).

    Returns:
        np.ndarray: The yields.
    """
    maturities = np.atleast_1d(np.asarray(maturities, dtype=float))
    design = _design(maturities, np.array([tau_1]), np.array([tau_2]))[0]
    return design @ np.array([beta_0, beta_1, beta_2, beta_3])


def _level_floor(maturities: np.ndarray, yields: np.ndarray) -> float:
    """
    The lowest level β0 a fit may have. The level is the yield at an infinite maturity, so it
    is kept at or above the lower of zero and the longest yield (as Gürkaynak, Sack and
    Wright keep it at or above zero); unbounded, the level and the second curvature offset
    each other into a level far below any yield.
    """
    return min(0.0, float(yields[np.argmax(maturities)]))


def _solve(design: np.ndarray, yields: np.ndarray) -> np.ndarray:
    """
    The least squares betas for a stack of designs (fits by maturities by factors) and the
    yields of each fit (fits by maturities).
    """
    q, r = np.linalg.qr(design)
    return np.linalg.solve(r, np.einsum("gmk,gm->gk", q, yields)[..., None])[..., 0]


def _zoom(
    maturities: np.ndarray,
    yields: np.ndarray,
    taus: np.ndarray,
    steps: tuple[float, float],
    shortest: float,
    single: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The best fits around the grid points of several curves at once: per curve a grid of
    decays spanning one and a half grid steps on either side (in logarithms), which zooms in
    on its best point threefold each round. The level is kept at or above its floor (see
    _level_floor): with one bound on a linear least squares problem the best fit either
    meets the bound or lies on it.

    Args:
        maturities (np.ndarray): The maturities in years.
        yields (np.ndarray): The yields, curves by maturities.
        taus (np.ndarray): The decays of the best grid point per curve, curves by two (the
            second is ignored for Nelson-Siegel).
        steps (tuple[float, float]): The steps of the decay grids, in logarithms.
        shortest (float): The lowest first decay.
        single (bool): Whether the model has one decay (Nelson-Siegel).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The squared error, the betas and the
        decays of the best fit per curve.
    """
    curves = len(yields)
    offsets = np.linspace(-1, 1, ZOOM_POINTS)
    if single:
        grid = offsets[:, None] * np.array([1.0, 0.0])
    else:
        first, second = np.meshgrid(offsets, offsets, indexing="ij")
        grid = np.column_stack([first.ravel(), second.ravel()])
    centers = np.log(taus)
    widths = 1.5 * np.asarray(steps)
    floors = np.minimum(0.0, yields[:, np.argmax(maturities)])
    repeated = np.repeat(yields, len(grid), axis=0)
    best_errors = np.full(curves, np.inf)
    best_betas = np.full((curves, 3 if single else 4), np.nan)
    best_taus = taus.copy()

    for _ in range(ZOOM_ROUNDS):
        points = np.exp(centers[:, None, :] + grid[None, :, :] * widths)
        tau_1, tau_2 = points[..., 0].ravel(), points[..., 1].ravel()
        design = _design(maturities, tau_1, None if single else tau_2)
        betas = _solve(design, repeated)
        below = betas[:, 0] < np.repeat(floors, len(grid))
        if below.any():
            level = np.repeat(floors, len(grid))[below]
            rest = _solve(design[below][:, :, 1:], repeated[below] - level[:, None])
            betas[below] = np.column_stack([level, rest])
        errors = np.sum(
            (np.einsum("gmk,gk->gm", design, betas) - repeated) ** 2, axis=1
        )
        allowed = tau_1 >= shortest
        if not single:
            allowed &= tau_2 >= TAU_RATIO * tau_1
        errors = np.where(allowed, errors, np.inf).reshape(curves, len(grid))
        index = np.argmin(errors, axis=1)
        rows = np.arange(curves)
        chosen = errors[rows, index]
        better = chosen < best_errors
        best_errors[better] = chosen[better]
        best_betas[better] = betas.reshape(curves, len(grid), -1)[rows, index][better]
        best_taus[better] = points[rows, index][better]
        centers = np.where(
            np.isfinite(chosen)[:, None], np.log(points[rows, index]), centers
        )
        widths = widths / 3

    return best_errors, best_betas, best_taus


def fit_nelson_siegel_curves(
    curves: pd.DataFrame, svensson: bool = True
) -> pd.DataFrame:
    """
    Fit the Nelson-Siegel or Nelson-Siegel-Svensson model to every curve (row) of a yield
    curve history. Given the decays the betas follow exactly from linear least squares, so
    the decays are searched on a grid, for all curves with the same maturities at once,
    after which the search zooms in on the best grid point of each curve. The level is
    kept at or above the lower of zero and the longest yield, and the first decay at least a
    third of the shortest maturity.

    The Nelson-Siegel model describes a curve with a level, a slope and a curvature:

        y(m) = β0 + β1 (1 - e^(-m/τ1)) / (m/τ1) + β2 [(1 - e^(-m/τ1)) / (m/τ1) - e^(-m/τ1)]

    and Svensson adds a second curvature, β3 with decay τ2, which fits the long end better.
    With a dozen maturities the six Svensson parameters are only weakly identified (the
    level and the second curvature can offset each other), so the Nelson-Siegel parameters
    are the ones to read as the level, slope and curvature of the curve.

    For more information, see the following papers:

    - Nelson, C.R. and Siegel, A.F. (1987). "Parsimonious Modeling of Yield Curves." The
      Journal of Business, 60(4), 473-489.
    - Svensson, L.E.O. (1994). "Estimating and Interpreting Forward Interest Rates: Sweden
      1992-1994." NBER Working Paper 4871.
    - Diebold, F.X. and Li, C. (2006). "Forecasting the Term Structure of Government Bond
      Yields." Journal of Econometrics, 130(2), 337-364.
    - Gürkaynak, R.S., Sack, B. and Wright, J.H. (2007). "The U.S. Treasury Yield Curve: 1961
      to the Present." Journal of Monetary Economics, 54(8), 2291-2304.

    Args:
        curves (pd.DataFrame): The yields as decimals (continuously compounded, as the
            models define them), by date (rows) and maturity in years (columns). Missing
            yields are left out of the fit of that curve.
        svensson (bool, optional): Whether to fit the Svensson model (True) or the
            Nelson-Siegel model (False). Defaults to True.

    Returns:
        pd.DataFrame: Beta 0 to Beta 2 (and Beta 3), Tau 1 (and Tau 2) and the Root Mean
        Squared Error per curve. A curve with fewer maturities than parameters is NaN.
    """
    maturities = np.asarray(curves.columns, dtype=float)
    values = curves.to_numpy(dtype=float)

    if svensson:
        grid_1, grid_2 = np.meshgrid(SVENSSON_TAUS_1, SVENSSON_TAUS_2, indexing="ij")
        valid = grid_2 >= TAU_RATIO * grid_1
        names = ["Beta 0", "Beta 1", "Beta 2", "Beta 3", "Tau 1", "Tau 2"]
        steps = (
            float(np.diff(np.log(SVENSSON_TAUS_1))[0]),
            float(np.diff(np.log(SVENSSON_TAUS_2))[0]),
        )
    else:
        grid_1 = NELSON_SIEGEL_TAUS[:, None]
        grid_2 = None
        valid = np.ones_like(grid_1, dtype=bool)
        names = ["Beta 0", "Beta 1", "Beta 2", "Tau 1"]
        steps = (float(np.diff(np.log(NELSON_SIEGEL_TAUS))[0]), 0.0)
    parameters = len(names)

    result = np.full((len(curves), len(names) + 1), np.nan)
    available = np.isfinite(values) & (maturities > 0)
    patterns, groups = np.unique(available, axis=0, return_inverse=True)

    for number, pattern in enumerate(patterns):
        if pattern.sum() < parameters:
            continue
        rows = np.flatnonzero(groups.ravel() == number)
        group_maturities = maturities[pattern]
        shortest = min(
            group_maturities.min() * SHORTEST_MATURITY_RATIO, float(grid_1.max())
        )
        group_valid = valid & (grid_1 >= shortest)
        design = _design(
            group_maturities,
            grid_1[group_valid],
            None if grid_2 is None else grid_2[group_valid],
        )
        basis, triangular = np.linalg.qr(design)
        # The bases side by side, maturities by grid points and factors, for one product;
        # the same without the level, for the fits that hold the level at its floor.
        stacked = basis.transpose(1, 0, 2).reshape(len(group_maturities), -1)
        level_basis, _ = np.linalg.qr(design[:, :, 1:])
        level_stacked = level_basis.transpose(1, 0, 2).reshape(
            len(group_maturities), -1
        )
        last = np.argmax(group_maturities)
        # The level of every fit is the first row of R⁻¹ times the projection.
        first_rows = np.linalg.inv(triangular)[:, 0, :]

        for start in range(0, len(rows), CHUNK_SIZE):
            chunk = rows[start : start + CHUNK_SIZE]
            yields = values[np.ix_(chunk, pattern)]
            # The squared error of every grid point: what the projection on the factors
            # leaves of each curve.
            projection = (yields @ stacked).reshape(len(chunk), *basis.shape[::2])
            unbounded = np.sum(yields**2, axis=1)[:, None] - np.sum(
                projection**2, axis=2
            )
            # Where the unbounded level falls below its floor, the error of the fit that
            # holds the level at the floor: what the other factors leave of the curve
            # minus the floor.
            levels = np.einsum("gk,cgk->cg", first_rows, projection)
            floors = np.minimum(0.0, yields[:, last])
            shifted = yields - floors[:, None]
            level_projection = (shifted @ level_stacked).reshape(
                len(chunk), level_basis.shape[0], -1
            )
            bounded = np.sum(shifted**2, axis=1)[:, None] - np.sum(
                level_projection**2, axis=2
            )
            errors = np.full((len(chunk), *valid.shape), np.inf)
            errors[:, group_valid] = np.where(
                levels >= floors[:, None], unbounded, bounded
            )

            best = np.argmin(errors.reshape(len(chunk), -1), axis=1)
            i, j = np.unravel_index(best, valid.shape)
            starts = np.column_stack(
                [grid_1[i, j], np.ones(len(chunk)) if grid_2 is None else grid_2[i, j]]
            )
            fit_errors, betas, taus = _zoom(
                group_maturities, yields, starts, steps, shortest, grid_2 is None
            )
            decays = taus[:, :1] if grid_2 is None else taus
            result[chunk] = np.column_stack(
                [betas, decays, np.sqrt(fit_errors / pattern.sum())]
            )

    return pd.DataFrame(
        result, index=curves.index, columns=[*names, "Root Mean Squared Error"]
    )


def fit_nelson_siegel_svensson(
    maturities: np.ndarray, yields: np.ndarray, svensson: bool = True
) -> dict[str, float]:
    """
    Fit the Nelson-Siegel-Svensson (or Nelson-Siegel) model to one yield curve, see
    fit_nelson_siegel_curves.

    Args:
        maturities (np.ndarray): The maturities in years.
        yields (np.ndarray): The yields, as decimals.
        svensson (bool, optional): Whether to fit the Svensson model (True) or the
            Nelson-Siegel model (False). Defaults to True.

    Returns:
        dict[str, float]: The parameters and the Root Mean Squared Error of the fit.
    """
    curve = pd.DataFrame(
        [np.asarray(yields, dtype=float)], columns=np.asarray(maturities, dtype=float)
    )
    fit = fit_nelson_siegel_curves(curve, svensson).iloc[0]
    if fit.isna().all():
        raise ValueError(
            f"At least {len(fit) - 1} maturities are needed to fit the "
            f"{'Svensson' if svensson else 'Nelson-Siegel'} model."
        )
    return {name: float(value) for name, value in fit.items()}


BILL_CONVENTIONS = ("bond-equivalent", "act/365", "act/360")


def bill_yields_to_annual(
    maturities: np.ndarray,
    yields: np.ndarray,
    frequency: int = 2,
    convention: str | None = None,
) -> np.ndarray:
    """
    Convert the yields of zero-coupon bills (up to a year) to annually compounded rates. By
    default the yields are compounded `frequency` times a year; a convention gives the way a
    country quotes its bills instead:

        - "bond-equivalent": the US Treasury's bond-equivalent yield, simple interest up to
          half a year and for a longer bill a half-year compounding step followed by simple
          interest for the rest, 1 + annual rate = [(1 + y min(m, ½)) (1 + y max(m - ½, 0))]^(1/m)
        - "act/365": simple interest over the actual days on a 365-day year, as the Government
          of Canada quotes its treasury bills, 1 + annual rate = (1 + y m)^(1/m)
        - "act/360": simple interest over the actual days on a 360-day year, as the Swedish
          National Debt Office quotes its treasury bills (statsskuldväxlar),
          1 + annual rate = (1 + y m 365 / 360)^(1/m)

    with m the maturity in years. Longer maturities are compounded `frequency` times a year
    whatever the convention.

    For more information, see the following sources:

    - Department of Finance Canada. "Determining Bond and Treasury Bill Prices and Yields."
      Government of Canada Securities Technical Guide.
    - Riksgälden (2023). "Räkneexempel statsskuldväxlar." Based on Svenska
      Fondhandlareföreningen (2001), "Beräkningsprinciper för den svenska penningmarknaden."

    Args:
        maturities (np.ndarray): The maturities in years.
        yields (np.ndarray): The yields, as decimals.
        frequency (int, optional): The compounding periods per year. Defaults to 2.
        convention (str | None, optional): "bond-equivalent", "act/365" or "act/360", or None
            for yields compounded `frequency` times a year. Defaults to None.

    Returns:
        np.ndarray: The annually compounded rates.
    """
    maturities = np.asarray(maturities, dtype=float)
    yields = np.asarray(yields, dtype=float)
    compounded = (1 + yields / frequency) ** frequency - 1
    if convention is None:
        return compounded
    if convention == "bond-equivalent":
        growth = (1 + yields * np.minimum(maturities, 0.5)) * (
            1 + yields * np.maximum(maturities - 0.5, 0)
        )
    elif convention in ("act/365", "act/360"):
        # A maturity in years counts days of a 365-day year, so on a 360-day basis the same
        # days earn 365/360 times the quoted yield.
        basis = 365 / 360 if convention == "act/360" else 1.0
        growth = 1 + yields * basis * maturities
    else:
        raise ValueError(
            f"The convention must be one of {', '.join(BILL_CONVENTIONS)} or None."
        )
    years = np.where(maturities > 0, maturities, 1.0)
    return np.where(
        (maturities > 0) & (maturities <= 1 + 1e-9),
        growth ** (1 / years) - 1,
        compounded,
    )


def bootstrap_zero_rates(
    maturities: np.ndarray,
    par_yields: np.ndarray,
    frequency: int = 2,
    bill_convention: str | None = None,
) -> np.ndarray:
    """
    Bootstrap zero-coupon rates from par yields, quoted with `frequency` compounding periods a
    year (bond-equivalent yields). Maturities up to one year are treated as zero-coupon
    (bills), so only their compounding changes (see bill_yields_to_annual); longer ones as
    bonds paying `frequency` coupons a year at their par yield. The par yields are
    interpolated linearly to every coupon date, and each discount factor follows from the
    earlier ones:

        P(t_n) = (1 - c/f Σ_{i<n} P(t_i)) / (1 + c/f)

    Args:
        maturities (np.ndarray): The maturities in years, ascending.
        par_yields (np.ndarray): The par yields, as decimals.
        frequency (int, optional): The coupons per year. Defaults to 2.
        bill_convention (str | None, optional): How the bill yields (up to a year) are
            quoted: "bond-equivalent" (the US Treasury), "act/365" (Canada), "act/360"
            (Sweden) or None for yields compounded `frequency` times a year, see
            bill_yields_to_annual. Defaults to None.

    Returns:
        np.ndarray: The zero rates (annually compounded) at the given maturities.
    """
    maturities = np.asarray(maturities, dtype=float)
    par_yields = np.asarray(par_yields, dtype=float)
    order = np.argsort(maturities)
    maturities, par_yields = maturities[order], par_yields[order]

    step = 1 / frequency
    grid = np.arange(step, maturities.max() + 1e-9, step)
    grid_yields = np.interp(grid, maturities, par_yields)

    discount = np.empty_like(grid)
    for n, (time, coupon) in enumerate(zip(grid, grid_yields, strict=True)):
        if time <= 1 + 1e-9:
            # The coupon dates up to a year discount at the bill yield, in its convention.
            discount[n] = (
                1
                + bill_yields_to_annual(
                    np.array([time]), np.array([coupon]), frequency, bill_convention
                )[0]
            ) ** (-time)
        else:
            discount[n] = (1 - coupon * step * discount[:n].sum()) / (1 + coupon * step)

    zero_grid = discount ** (-1 / grid) - 1
    zero_rates = np.interp(maturities, grid, zero_grid)
    # Bills (and maturities before the first coupon date) are zero-coupon already, so their
    # yield only changes compounding.
    zero_rates = np.where(
        maturities <= 1 + 1e-9,
        bill_yields_to_annual(maturities, par_yields, frequency, bill_convention),
        zero_rates,
    )

    result = np.empty_like(zero_rates)
    result[order] = zero_rates
    return result


def get_principal_components(
    curves: pd.DataFrame, components: int = 3, changes: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    The principal components of a yield curve history (Litterman and Scheinkman, 1991):
    the few independent movements that explain most of how the curve changes, typically a
    parallel shift (level), a steepening (slope) and a bending (curvature). The sign of a
    component is arbitrary; PC1 is oriented so that its loadings sum to a positive number (a
    rise in all yields), PC2 so that the longest maturity loads above the shortest (a
    steepening) and the others so that their loadings sum to a positive number.

    Args:
        curves (pd.DataFrame): The yields by date (rows) and maturity (columns).
        components (int, optional): The number of components. Defaults to 3.
        changes (bool, optional): Whether to analyze the changes of the yields (the usual
            choice, as yields themselves are not stationary) or their levels. Defaults to True.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series]: The loadings (maturities by component),
        the scores (dates by component) and the share of the variance each component explains.
    """
    data = curves.dropna(how="any")
    if changes:
        data = data.diff().dropna(how="any")
    if len(data) < components + 2:
        raise ValueError("Too few complete curves for a principal component analysis.")

    centered = data - data.mean()
    covariance = np.cov(centered.to_numpy(), rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:components]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

    # The sign of a component is arbitrary: the first is a rise in all yields, the second a
    # steepening (the longest maturity rising against the shortest).
    orientation = eigenvectors.sum(axis=0)
    if components > 1:
        years = [maturity_in_years(label) for label in data.columns]
        orientation[1] = (
            eigenvectors[np.argmax(years), 1] - eigenvectors[np.argmin(years), 1]
        )
    signs = np.sign(orientation)
    signs[signs == 0] = 1
    eigenvectors = eigenvectors * signs

    names = [f"PC{number}" for number in range(1, components + 1)]
    loadings = pd.DataFrame(eigenvectors, index=data.columns, columns=names)
    scores = pd.DataFrame(
        centered.to_numpy() @ eigenvectors, index=data.index, columns=names
    )
    explained = pd.Series(
        eigenvalues / np.trace(covariance), index=names, name="Explained Variance"
    )
    return loadings, scores, explained
