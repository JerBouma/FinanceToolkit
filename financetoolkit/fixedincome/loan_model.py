"""Loan Model Module"""

__docformat__ = "google"

import numpy as np
import pandas as pd


def get_loan_payment(
    principal: float,
    rate: float | pd.Series,
    years: float,
    payments_per_year: int = 12,
) -> float | pd.Series:
    """
    Calculates the fixed payment that repays a loan in equal instalments (an annuity, the
    payment of a fixed-rate mortgage).

    The formula is as follows:

    - Payment = L * r / (1 - (1 + r)^-n), or L / n when r is zero

    Where L is the principal, r = annual rate / payments per year the rate per payment and
    n = years * payments per year the number of payments. The annual rate is a nominal rate
    compounded at the payment frequency, the way US mortgages are quoted (a 7.18% rate is
    0.5983% a month).

    Args:
        principal (float): The amount borrowed.
        rate (float | pd.Series): The nominal annual rate as a decimal (0.0718 for 7.18%), or a
            series of rates, which gives the payment a new loan would have at each of them.
        years (float): The term of the loan in years.
        payments_per_year (int, optional): The number of payments per year. Defaults to 12.

    Returns:
        float | pd.Series: The payment per period, a series when rate is one.
    """
    periodic_rate = np.asarray(rate, dtype=float) / payments_per_year
    number_of_payments = years * payments_per_year

    # With a zero rate the annuity factor is 0 / 0; the loan is then repaid in equal parts.
    with np.errstate(divide="ignore", invalid="ignore"):
        payment = np.where(
            periodic_rate == 0,
            principal / number_of_payments,
            principal
            * periodic_rate
            / (1 - (1 + periodic_rate) ** -float(number_of_payments)),
        )

    if isinstance(rate, pd.Series):
        return pd.Series(payment, index=rate.index, name=rate.name)

    return float(payment)


def get_affordable_loan(
    payment: float,
    rate: float | pd.Series,
    years: float,
    payments_per_year: int = 12,
) -> float | pd.Series:
    """
    Calculates the loan a fixed payment carries: the principal that the payment repays in
    equal instalments, the inverse of `get_loan_payment`.

    The formula is as follows:

    - Loan = P * (1 - (1 + r)^-n) / r, or P * n when r is zero

    Where P is the payment, r = annual rate / payments per year the rate per payment and
    n = years * payments per year the number of payments, with the annual rate a nominal rate
    compounded at the payment frequency.

    Args:
        payment (float): The payment per period.
        rate (float | pd.Series): The nominal annual rate as a decimal, or a series of rates,
            which gives the loan the payment carries at each of them.
        years (float): The term of the loan in years.
        payments_per_year (int, optional): The number of payments per year. Defaults to 12.

    Returns:
        float | pd.Series: The principal, a series when rate is one.
    """
    periodic_rate = np.asarray(rate, dtype=float) / payments_per_year
    number_of_payments = years * payments_per_year

    with np.errstate(divide="ignore", invalid="ignore"):
        loan = np.where(
            periodic_rate == 0,
            payment * number_of_payments,
            payment
            * (1 - (1 + periodic_rate) ** -float(number_of_payments))
            / periodic_rate,
        )

    if isinstance(rate, pd.Series):
        return pd.Series(loan, index=rate.index, name=rate.name)

    return float(loan)


def get_amortization_schedule(
    principal: float,
    rate: float,
    years: float,
    payments_per_year: int = 12,
) -> pd.DataFrame:
    """
    Calculates the amortization schedule of a fixed-rate loan repaid in equal payments: per
    payment the part that is interest, the part that repays principal and the balance left.

    The formula is as follows:

    - Interest(k) = Balance(k - 1) * r
    - Principal(k) = Payment - Interest(k)
    - Balance(k) = Balance(k - 1) - Principal(k), with Balance(0) = L

    Where L is the principal, r = annual rate / payments per year and Payment the annuity
    payment of `get_loan_payment`. The balance after the last payment is zero.

    Args:
        principal (float): The amount borrowed.
        rate (float): The nominal annual rate as a decimal (0.0718 for 7.18%).
        years (float): The term of the loan in years.
        payments_per_year (int, optional): The number of payments per year. Defaults to 12.

    Returns:
        pd.DataFrame: Per payment (1 to n) the Payment, Interest, Principal and Balance.
    """
    number_of_payments = int(round(years * payments_per_year))
    periodic_rate = rate / payments_per_year
    payment = get_loan_payment(principal, rate, years, payments_per_year)

    # The balance after k payments, in closed form so no rounding accumulates.
    periods = np.arange(number_of_payments + 1)
    growth = (1 + periodic_rate) ** periods
    balance = principal * growth - (
        payment * (growth - 1) / periodic_rate if periodic_rate else payment * periods
    )
    balance[-1] = 0.0

    interest = balance[:-1] * periodic_rate

    return pd.DataFrame(
        {
            "Payment": payment,
            "Interest": interest,
            "Principal": payment - interest,
            "Balance": balance[1:],
        },
        index=pd.Index(periods[1:], name="Payment Number"),
    )
