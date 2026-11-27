"""Helpers Module"""

import re
from collections.abc import Callable

import numpy as np
import pandas as pd
import yaml
from scipy import sparse, stats
from scipy.optimize import linprog

from financetoolkit.helpers import get_non_trading_days
from financetoolkit.utilities.excel_model import read_excel as read_workbook
from financetoolkit.utilities.statistics_model import drop_non_finite


def read_excel(location: str):
    """
    Read an Excel (.xlsx or .xls) or CSV (.csv) file into a Pandas DataFrame.

    This function reads and loads data from an Excel or CSV file located at the specified 'location'
    into a Pandas DataFrame.

    Args:
        location (str): The file path of the Excel or CSV file to read.

    Returns:
        pandas.DataFrame: A DataFrame containing the data from the file.

    Raises:
        ValueError: If the specified file does not have a '.xlsx' or '.csv' extension.
    """
    if location.endswith(".xlsx") or location.endswith(".xls"):
        return read_workbook(location)
    if location.endswith(".csv"):
        csv_dataset = pd.read_csv(location, delimiter=",")

        if len(csv_dataset.columns) == 1:
            csv_dataset = pd.read_csv(location, delimiter=";")

        return csv_dataset

    raise ValueError("File type not supported. Please use .xlsx or .csv")


def read_yaml_file(location: str):
    """
    Read and parse a YAML file.

    This function reads a YAML file located at the specified 'location' and parses its contents into
    a Python dictionary. It handles exceptions for file not found, YAML parsing errors, and other
    general exceptions.

    Args:
        location (str): The file path of the YAML file to read and parse.

    Returns:
        dict: A dictionary containing the parsed YAML data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        yaml.YAMLError: If there is an error in parsing the YAML content.
        Exception: For any other general exceptions that may occur during file reading or parsing.
    """
    try:
        with open(location) as yaml_file:
            data = yaml.safe_load(yaml_file)
        return data
    except FileNotFoundError as exc:
        raise ValueError(f"The file '{location}' does not exist.") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Error parsing '{location}': {exc}") from exc
    except KeyError as exc:
        raise ValueError(f"An error occurred: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"An error occurred: {exc}") from exc


def convert_to_float(value: str, decimal_comma: bool = True):
    """
    Convert a numeric string with locale-specific formatting to a float using a dot as the decimal separator.

    This function handles input strings formatted in one of two ways:
    - Using a dot as the thousands separator and a comma as the decimal separator (e.g., "-4.587,24").
    - Using a comma as the thousands separator and a dot as the decimal separator (e.g., "-4,587.24").

    Whole numbers with more than one thousands separator ("1,234,567" or "1.234.567") lose their
    separators, and spaces or apostrophes between the thousands are removed. A single dot is a
    decimal point ("1.234" is 1.234). A single comma is a decimal comma ("2,500" is 2.5, as in a
    semicolon-separated European file) unless decimal_comma is False, in which case a comma
    followed by three digits separates thousands ("2,500" is 2500). convert_column_to_float
    decides this per column.

    The function works as follows:
    1. If the input is a string, it checks its format using regular expressions.
    2. Depending on the detected format, it replaces the appropriate characters:
        - For numbers like "-4.587,24": removes dots (thousands separator) and replaces the comma with a dot.
        - For numbers like "-4,587.24": removes commas (thousands separator).
    3. Converts the resulting string to a float.

    If the input is not a string, it will attempt to convert it directly to float.

    Args:
        value (str): The numeric string to convert to a float.
        decimal_comma (bool, optional): Whether a single comma followed by three digits is a decimal
            comma. Defaults to True.

    Raises:
         ValueError: If the conversion fails or the input does not conform to the expected numeric formats.

    Returns:
         float: The converted floating-point number.
    """
    try:
        if isinstance(value, str):
            # Spaces, non-breaking spaces and apostrophes only separate thousands ("1 234,56", "1'234.56").  # noqa: E501
            value = re.sub(r"[\s\u00a0\u202f']", "", value)
            # Match numbers like "2,500" and "-1,234,567" in a column that separates thousands with commas.  # noqa: E501
            if not decimal_comma and re.match(r"^-?\d{1,3}(,\d{3})+$", value):
                value = value.replace(",", "")
            # Match numbers like "-4.587,24", "1234,56" and "2,500": a single comma is a decimal comma.
            elif re.match(r"^-?(\d{1,3}(\.\d{3})*|\d+),\d+$", value):
                value = value.replace(".", "").replace(",", ".")
            # Match numbers like "-4,587.24" and "1,234,567"
            elif re.match(r"^-?\d{1,3}((,\d{3})*\.\d+|(,\d{3}){2,})$", value):
                value = value.replace(",", "")
            # Match numbers like "1.234.567"
            elif re.match(r"^-?\d{1,3}(\.\d{3}){2,}$", value):
                value = value.replace(".", "")
        return float(value)
    except Exception as e:
        raise ValueError(f"Cannot convert {value} to float: {e}") from e


def convert_column_to_float(column: pd.Series) -> pd.Series:
    """
    Convert a column of numeric strings to floats, reading its separators once for the whole column.

    "2,500" can be 2.5 (a decimal comma) or 2500 (a thousands comma). A single value can't tell, but
    its column can: a value with a comma before a decimal point ("1,234.56") or more than one comma
    ("1,234,567") means the commas separate thousands, so "2,500" in that column is 2500. Otherwise a
    single comma is a decimal comma, as in a semicolon-separated European file.

    Args:
        column (pd.Series): The values to convert (strings, numbers or NaN).

    Raises:
        ValueError: If a value does not conform to the expected numeric formats.

    Returns:
        pd.Series: The converted floating-point numbers.
    """
    thousands_comma = any(
        re.match(
            r"^-?\d{1,3}((,\d{3})+\.\d+|(,\d{3}){2,})$",
            re.sub(r"[\s\u00a0\u202f']", "", value),
        )
        for value in column
        if isinstance(value, str)
    )

    return column.apply(convert_to_float, decimal_comma=not thousands_comma)


BENCHMARK_TICKER = "Benchmark"


PORTFOLIO_TICKER = "Portfolio"


def divide_by_capital(
    numerator: pd.Series | pd.DataFrame, denominator: pd.Series | pd.DataFrame
) -> pd.Series | pd.DataFrame:
    """
    Divide two objects, returning NaN wherever the denominator is zero or negative.

    A return is only defined against capital that was actually deployed. Dividing by a
    zero or negative denominator produces an infinite or sign-flipped number that reads
    as a real result, so those cases are reported as missing instead.

    Args:
        numerator (pd.Series | pd.DataFrame): The numerator of the division.
        denominator (pd.Series | pd.DataFrame): The denominator of the division.

    Returns:
        pd.Series | pd.DataFrame: The element-wise division with non-positive
            denominators replaced by NaN.
    """
    result = numerator / denominator.where(denominator > 0)

    return drop_non_finite(result)


def get_trading_day_locations(
    historical_data: pd.DataFrame, dates: pd.Index, tickers: pd.Index
) -> np.ndarray:
    """
    The position in the dates of the historical data of the first day on or after each date
    that its ticker traded on, the day a transaction on that date is priced and booked on.

    With tickers on different calendars, such as a stock next to a cryptocurrency, the
    historical data holds the dates of all of them and carries a ticker's last close forward
    over the dates only another ticker traded (see `get_non_trading_days`). Taking the first
    date with a price would book a Saturday purchase of a stock at Friday's close, in the
    week or month before its position can start; the stock's own next trading day, Monday,
    is the day the order fills. A ticker without historical data, or a transaction before the
    ticker's first price, takes the first date on or after the transaction, and a transaction
    after the last trading day of its ticker takes the last date of the data, the latest close
    it is priced at.

    Args:
        historical_data (pd.DataFrame): The daily historical data with (column, ticker)
            columns, including "Close" and "Return", on sorted dates.
        dates (pd.Index): The date of every transaction, of the same type as the dates of the
            historical data.
        tickers (pd.Index): The ticker of every transaction.

    Returns:
        np.ndarray: The position of the trading day of every transaction.
    """
    all_dates = historical_data.index
    locations = all_dates.get_indexer(dates, method="backfill")

    if (
        isinstance(historical_data.columns, pd.MultiIndex)
        and "Close" in historical_data
    ):
        # Only the carried-forward dates are skipped: a date before a ticker's first price
        # keeps the transaction on it, as a position held without a price yet.
        trading_days = ~get_non_trading_days(historical_data, "Close")
        for ticker in pd.unique(np.asarray(tickers)):
            if ticker not in trading_days.columns:
                continue
            selected = np.asarray(tickers == ticker)
            own_positions = np.flatnonzero(trading_days[ticker].to_numpy())
            own_locations = all_dates[own_positions].get_indexer(
                dates[selected], method="backfill"
            )
            locations[selected] = np.where(
                own_locations == -1, -1, own_positions[own_locations]
            )

    locations[locations == -1] = len(all_dates) - 1

    return locations


def get_view_vector(view: str, names: pd.Index) -> np.ndarray:
    """
    The pick vector of a view on one name ("AAPL": +1 for AAPL) or on the difference between
    two names.

    The view "AAPL" is +1 for AAPL and "AAPL-MSFT" +1 for AAPL and -1 for MSFT, as the views
    of Black-Litterman and Entropy Pooling are written. Names can hold a hyphen themselves
    (BRK-B, Mkt-RF), so a name that matches in full is read as one name and otherwise every
    hyphen is tried as the split between the two legs.

    Args:
        view (str): The view, a name or two names joined by a hyphen.
        names (pd.Index): The names (assets or factors) the view can refer to.

    Returns:
        np.ndarray: The pick vector, one entry per name.
    """
    row = np.zeros(len(names))
    if view in names:
        row[names.get_loc(view)] = 1
        return row
    legs = [
        (view[:position], view[position + 1 :])
        for position, character in enumerate(view)
        if character == "-"
        and view[:position] in names
        and view[position + 1 :] in names
    ]
    if len(legs) > 1:
        raise ValueError(f"The view {view!r} can be read as more than one pair.")
    if not legs:
        raise ValueError(
            f"The view {view!r} names no asset or factor; use 'NAME' or 'NAME-NAME'."
        )
    row[names.get_loc(legs[0][0])] = 1
    row[names.get_loc(legs[0][1])] = -1
    return row


def get_risk_budget(risk_budget: pd.Series | None, assets: pd.Index) -> np.ndarray:
    """
    The share of the risk every asset should contribute, scaled to sum to one, equal shares
    without a risk budget.

    A budget that names an asset outside the portfolio, or leaves an asset without a
    positive budget, is refused with the assets named, rather than silently dropping or
    optimizing without them.

    Args:
        risk_budget (pd.Series | None): The risk budget per asset.
        assets (pd.Index): The assets of the portfolio.

    Returns:
        np.ndarray: The risk budget per asset, in the order of assets, summing to one.
    """
    if risk_budget is None:
        return np.full(len(assets), 1 / len(assets))
    unknown = risk_budget.index.difference(assets)
    budget = risk_budget.reindex(assets).astype(float)
    problems = [
        f"{description}: {', '.join(map(str, names))}"
        for description, names in (
            ("not in the portfolio", unknown),
            ("without a risk budget", budget.index[budget.isna()]),
            ("with a risk budget that is not positive", budget.index[budget <= 0]),
        )
        if len(names)
    ]
    if problems:
        raise ValueError(
            "The risk budget needs a positive share for every asset of the portfolio and "
            f"no other; assets {'; assets '.join(problems)}."
        )
    return (budget / budget.sum()).to_numpy()


def get_regularized_covariance(
    covariance: np.ndarray, l2_regularization: float
) -> np.ndarray:
    """
    The covariance matrix with the L2 penalty γ w'w on the weights folded in, Σ + γI, so
    that w'(Σ + γI)w = w'Σw + γ w'w.

    With weights that sum to one, w'w is the squared distance from the equal weights plus
    1/N, so the penalty pulls a portfolio towards equal weights.

    Args:
        covariance (np.ndarray): The covariance matrix Σ.
        l2_regularization (float): The weight γ of the penalty, in the units of Σ.

    Returns:
        np.ndarray: The matrix Σ + γI.
    """
    if l2_regularization < 0:
        raise ValueError("The L2 regularization cannot be negative.")
    return covariance + l2_regularization * np.eye(len(covariance))


def get_linear_regression(target: np.ndarray, regressors: np.ndarray) -> dict:
    """
    The ordinary least squares regression of a target on regressors and a constant, with the
    statistics the stepwise selection of factors compares.

    It gives the two-sided p-values of the t-statistics, the Akaike and Schwarz (Bayesian)
    information criteria and the (adjusted) R squared, defined as statsmodels' OLS defines
    them: the log-likelihood ln L = -n / 2 (ln(2π) + ln(SSR / n) + 1), AIC = -2 ln L + 2k
    and BIC = -2 ln L + k ln n with k the number of coefficients including the constant.

    Args:
        target (np.ndarray): The n observations of the dependent variable.
        regressors (np.ndarray): The n × m regressors, without the constant (m may be 0).

    Returns:
        dict: "params" (the constant first), "pvalues" (of the m regressors), "residuals",
        "aic", "bic", "r_squared" and "adjusted_r_squared".
    """
    observations = len(target)
    design = np.column_stack([np.ones(observations), regressors])
    coefficients = design.shape[1]
    if observations <= coefficients:
        raise ValueError(
            "The regression needs more observations than coefficients to estimate."
        )
    params, *_ = np.linalg.lstsq(design, target, rcond=None)
    residuals = target - design @ params
    sum_of_squares = float(residuals @ residuals)
    degrees_of_freedom = observations - coefficients
    variance = sum_of_squares / degrees_of_freedom
    standard_errors = np.sqrt(np.diag(variance * np.linalg.pinv(design.T @ design)))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_statistics = params / standard_errors
    pvalues = 2 * stats.t.sf(np.abs(t_statistics), degrees_of_freedom)
    log_likelihood = (
        -observations
        / 2
        * (np.log(2 * np.pi) + np.log(sum_of_squares / observations) + 1)
    )
    centred = target - target.mean()
    r_squared = 1 - sum_of_squares / float(centred @ centred)
    return {
        "params": params,
        "pvalues": pvalues[1:],
        "residuals": residuals,
        "aic": -2 * log_likelihood + 2 * coefficients,
        "bic": -2 * log_likelihood + np.log(observations) * coefficients,
        "r_squared": r_squared,
        "adjusted_r_squared": 1
        - (observations - 1) / degrees_of_freedom * (1 - r_squared),
    }


# HiGHS's feasibility tolerances for cutting planes: its default of 1e-7 lets a solution
# violate its newest cut by that much, so the cut never closes the gap to its tolerance.
CUTTING_PLANE_OPTIONS = {
    "primal_feasibility_tolerance": 1e-10,
    "dual_feasibility_tolerance": 1e-10,
}

# How far a solution may violate a cut and still count as meeting it, and how far the
# cost of the best feasible point found may exceed the lower bound of the programme (per
# unit of cost) for it to count as optimal.
CUTTING_PLANE_TOLERANCE = 1e-10
CUTTING_PLANE_GAP = 1e-12

# Where cuts are placed between the best feasible point and the solution of the linear
# programme: 0.2 of the way to the solution.
CUTTING_PLANE_STEP = 0.2


def get_cutting_plane_solution(
    cost: np.ndarray,
    inequality: sparse.csr_matrix,
    upper: np.ndarray,
    equality: sparse.csr_matrix | None,
    target: np.ndarray | None,
    bounds: list[tuple[float | None, float | None]],
    cuts: Callable[[np.ndarray], list[tuple[np.ndarray, float]]],
    feasible: Callable[[np.ndarray], np.ndarray | None] | None = None,
    iterations: int = 2000,
    unbounded: str = "",
) -> np.ndarray:
    """
    Minimize cost'x subject to inequality x ≤ upper, equality x = target and the bounds with
    HiGHS.

    Part of the feasible set is convex but not polyhedral, or polyhedral with too many faces
    to write down. That part enters as cutting planes (Kelley, 1960): the linear programme
    is solved, cuts(x) gives the supporting hyperplanes a'x ≤ b of the convex set at a point
    (none at a point inside it), those that the solution violates are added and the
    programme is solved again. No cut removes a feasible point, so a solution that violates
    none is the exact optimum.

    With feasible, which turns a solution into a feasible point of the same or a higher
    cost (or None), the cuts are placed between the cheapest feasible point found and the
    solution, the in-out stabilization of Ben-Ameur and Neto (2007), which keeps Kelley's
    method from jumping between far-apart solutions. The cut at the solution itself is
    used when those do not remove it, and the search stops when the cost of the cheapest
    feasible point is within the tolerance of the programme's, which bounds the optimum
    from below.

    See Kelley, J. E. (1960), "The Cutting-Plane Method for Solving Convex Programs",
    Journal of the Society for Industrial and Applied Mathematics, 8(4), 703-712, and
    Ben-Ameur, W. and Neto, J. (2007), "Acceleration of Cutting-Plane and Column
    Generation Algorithms: Applications to Network Design", Networks, 49(1), 3-17.

    Args:
        cost (np.ndarray): The cost of each variable.
        inequality (sparse.csr_matrix): The fixed inequality rows.
        upper (np.ndarray): Their right-hand side.
        equality (sparse.csr_matrix | None): The equality rows, None without any.
        target (np.ndarray | None): Their right-hand side.
        bounds (list[tuple[float | None, float | None]]): The bounds of each variable.
        cuts (Callable): The supporting rows (a, b) of the convex set at a point.
        feasible (Callable | None, optional): A feasible point from a solution, or None
            where there is none. Defaults to None, cuts at the solutions only.
        iterations (int, optional): The largest number of linear programmes. Defaults
            to 2000.
        unbounded (str, optional): What to add to the error of an unbounded programme.
            Defaults to "".

    Returns:
        np.ndarray: The optimal variables.
    """
    rows: list[np.ndarray] = []
    limits: list[float] = []
    best, incumbent = np.inf, None
    for _ in range(iterations):
        result = linprog(
            cost,
            A_ub=(
                sparse.vstack([inequality, sparse.csr_matrix(np.array(rows))])
                if rows
                else inequality
            ),
            b_ub=np.concatenate([upper, limits]),
            A_eq=equality,
            b_eq=target,
            bounds=bounds,
            method="highs",
            options=CUTTING_PLANE_OPTIONS,
        )
        if not result.success:
            raise ValueError(
                f"The optimization did not converge: {result.message} {unbounded}"
                if result.status == 3 and unbounded  # noqa: PLR2004
                else f"The optimization did not converge: {result.message}"
            )
        solution = result.x
        point = solution
        if feasible is not None:
            candidate = feasible(solution)
            if candidate is not None and cost @ candidate < best:
                best, incumbent = cost @ candidate, candidate
            if incumbent is not None:
                if best - result.fun <= CUTTING_PLANE_GAP * max(1, abs(best)):
                    return incumbent
                point = incumbent + CUTTING_PLANE_STEP * (solution - incumbent)

        def violated(at):
            return [
                (row, limit)
                for row, limit in cuts(at)
                if row @ solution > limit + CUTTING_PLANE_TOLERANCE
            ]

        found = violated(point) or (violated(solution) if point is not solution else [])
        if not found:
            return solution
        if feasible is not None and point is not solution:
            # The point between them is often cheaper than both once made feasible.
            candidate = feasible(point)
            if candidate is not None and cost @ candidate < best:
                best, incumbent = cost @ candidate, candidate
        for row, limit in found:
            rows.append(row)
            limits.append(limit)
    raise ValueError(
        f"The optimization did not converge within {iterations} cutting planes."
    )


def get_transaction_order(
    dates: pd.Index, volumes: pd.Series | np.ndarray, purchases_first: bool = True
) -> np.ndarray:
    """
    The order in which transactions are matched against each other.

    By date and, within a day, the purchases before the sales, so a sale is matched against
    the units bought that day whatever the order of the rows in the file. With
    `purchases_first=False`, for transactions in the order they were made (dated with a time
    of day), the rows of a day keep their order.

    Args:
        dates (pd.Index): The date of every transaction.
        volumes (pd.Series | np.ndarray): The volume of every transaction, negative for a sale.
        purchases_first (bool, optional): Whether the purchases of a day come before its
            sales. Defaults to True.

    Returns:
        np.ndarray: The positions of the transactions in the order they are matched in.
    """
    return (
        pd.DataFrame(
            {
                "date": dates,
                "sell": (np.asarray(volumes, dtype=float) < 0) & purchases_first,
            }
        )
        .sort_values(["date", "sell"], kind="stable")
        .index.to_numpy()
    )


def to_timestamps(dates: pd.Index | pd.Series) -> pd.DatetimeIndex:
    """
    The dates as timestamps, the start of each period for periods, so that dates of either
    type give durations in calendar days.

    Args:
        dates (pd.Index | pd.Series): Dates as periods or timestamps.

    Returns:
        pd.DatetimeIndex: The dates as timestamps.
    """
    dates = pd.Index(dates)
    if isinstance(dates, pd.PeriodIndex):
        return dates.to_timestamp()
    return pd.DatetimeIndex(dates)
