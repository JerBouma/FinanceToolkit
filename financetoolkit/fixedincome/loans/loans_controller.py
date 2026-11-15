"""Loans Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit.economics.sources import (
    fred_model as economics_fred_model,
    freddie_mac_model,
)
from financetoolkit.fixedincome.base.base_controller import FixedIncomeBase
from financetoolkit.fixedincome.loans import loan_model
from financetoolkit.utilities.dataframe_model import filter_shown_columns
from financetoolkit.utilities.error_model import handle_errors
from financetoolkit.utilities.sources_model import track_sources
from financetoolkit.utilities.statistics_model import apply_rounding

# ruff: noqa: E501


@filter_shown_columns
@track_sources
class LoansFixedIncome(FixedIncomeBase):
    """
    Loans: the loan payment, the affordable loan and the amortization schedule.
    """

    def _get_mortgage_rates(self) -> pd.Series:
        """
        The weekly 30-year fixed mortgage rate of Freddie Mac's Primary Mortgage Market Survey
        between the start and end date, as a decimal, from FRED when a FRED API key is set and
        from Freddie Mac otherwise (the same figures, see `Economics.get_mortgage_rate_30_year`)
        -- shared by the loan calculations, which take it as the rate of a new loan.
        """
        mortgage_rates = (
            economics_fred_model.get_mortgage_rate_30_year(
                self._start_date, self._end_date, self._fred_api_key
            )
            if self._fred_api_key
            else freddie_mac_model.get_mortgage_rate_30_year()
        )
        rates = (
            mortgage_rates["United States"]
            .loc[self._start_date : self._end_date]
            .dropna()
            .rename("Rate")
        )

        if rates.empty:
            raise ValueError(
                f"There is no 30-year mortgage rate between {self._start_date} and "
                f"{self._end_date}."
            )

        return rates

    @handle_errors
    def get_loan_payment(
        self,
        principal: float = 400_000,
        years: float = 30,
        payments_per_year: int = 12,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculates the monthly payment of a new 30-year fixed-rate mortgage at every weekly US
        mortgage rate: what the same loan costs a buyer at each date, with the total interest
        over its life and the interest in its first payment. By default the loan is 400,000
        US dollars, so the payment of the same house at the same price can be followed through
        the rate cycle (1,611.86 a month at the 2.65% of January 2021, 2,709.74 at 7.18%).

        The loan is an annuity, repaid in equal payments. The formula is as follows:

        - Payment = L * r / (1 - (1 + r)^-n), or L / n when r is zero
        - Total Interest = Payment * n - L
        - First Payment Interest = L * r

        Where:

        - L = Principal, the amount borrowed
        - r = Rate / payments per year, the rate per payment
        - n = years * payments per year, the number of payments

        US mortgage rates are quoted as a nominal annual rate compounded monthly, so the rate per
        payment is the annual rate divided by twelve (7.18% is 0.5983% a month). The rate is the
        weekly average 30-year fixed rate of Freddie Mac's Primary Mortgage Market Survey (the
        rate of `Economics.get_mortgage_rate_30_year`), from FRED when a FRED API key is set and
        from Freddie Mac otherwise. For a rate of your own, use
        `loan_model.get_loan_payment`, which takes a number or a series of rates.

        Also known as: mortgage payment, monthly payment, annuity payment, mortgage calculator,
        loan calculator, cost of a mortgage.

        Args:
            principal (float, optional): The amount borrowed. Defaults to 400,000.
            years (float, optional): The term of the loan in years. The rate stays the 30-year
                fixed mortgage rate for another term. Defaults to 30.
            payments_per_year (int, optional): The number of payments per year. Defaults to 12.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per survey date the Rate (a decimal), the Payment, the Total Interest
            over the life of the loan and the First Payment Interest, in the currency of the
            principal.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2021-01-01', end_date='2026-11-06')

        fixedincome.get_loan_payment().iloc[[0, -1]]
        ```

        Which returns:

        | Date       |   Rate |   Payment |   Total Interest |   First Payment Interest |
        |:-----------|-------:|----------:|-----------------:|-------------------------:|
        | 2021-01-07 | 0.0265 | 1611.8552 |      180267.8595 |                 883.3333 |
        | 2026-11-05 | 0.0718 | 2709.7393 |      575506.1579 |                2393.3333 |
        """
        rates = self._get_mortgage_rates()
        payment = loan_model.get_loan_payment(
            principal, rates, years, payments_per_year
        )

        loan_payment = pd.DataFrame(
            {
                "Rate": rates,
                "Payment": payment,
                "Total Interest": payment * years * payments_per_year - principal,
                "First Payment Interest": principal * rates / payments_per_year,
            }
        )

        return apply_rounding(
            loan_payment, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_affordable_loan(
        self,
        payment: float | None = None,
        principal: float = 400_000,
        years: float = 30,
        payments_per_year: int = 12,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculates the loan a fixed monthly payment carries at every weekly US mortgage rate:
        the 30-year fixed-rate mortgage the payment repays, and how that loan compares with the
        one it carried at the start date. By default the payment is the one a 400,000 US dollar
        loan had at the first rate of the range, so the answer is how much house the same
        monthly budget buys as rates move (the 1,611.86 a month of a 400,000 loan at 2.65%
        carries 237,935 at 7.18%).

        The loan is an annuity, repaid in equal payments. The formula is as follows:

        - Affordable Loan = P * (1 - (1 + r)^-n) / r, or P * n when r is zero
        - Change = Affordable Loan / Affordable Loan(first date) - 1

        Where:

        - P = Payment per period
        - r = Rate / payments per year, the rate per payment
        - n = years * payments per year, the number of payments

        US mortgage rates are quoted as a nominal annual rate compounded monthly, so the rate per
        payment is the annual rate divided by twelve. The rate is the weekly average 30-year fixed
        rate of Freddie Mac's Primary Mortgage Market Survey, from FRED when a FRED API key is set
        and from Freddie Mac otherwise. For a rate of your own, use
        `loan_model.get_affordable_loan`.

        Also known as: affordability, how much can I borrow, mortgage affordability, borrowing
        capacity, loan amount from payment.

        Args:
            payment (float | None, optional): The payment per period. Defaults to None, the
                payment of a loan of principal at the first rate between the start and end date.
            principal (float, optional): The loan whose payment at the first rate is used when
                no payment is given. Defaults to 400,000.
            years (float, optional): The term of the loan in years. The rate stays the 30-year
                fixed mortgage rate for another term. Defaults to 30.
            payments_per_year (int, optional): The number of payments per year. Defaults to 12.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per survey date the Rate (a decimal), the Payment, the Affordable Loan
            and its Change since the first date (a decimal).

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2021-01-01', end_date='2026-11-06')

        fixedincome.get_affordable_loan().iloc[[0, -1]]
        ```

        Which returns:

        | Date       |   Rate |   Payment |   Affordable Loan |   Change |
        |:-----------|-------:|----------:|------------------:|---------:|
        | 2021-01-07 | 0.0265 | 1611.8552 |       400000.0000 |   0.0000 |
        | 2026-11-05 | 0.0718 | 1611.8552 |       237935.0883 |  -0.4052 |
        """
        rates = self._get_mortgage_rates()

        if payment is None:
            payment = loan_model.get_loan_payment(
                principal, float(rates.iloc[0]), years, payments_per_year
            )

        affordable_loan = loan_model.get_affordable_loan(
            payment, rates, years, payments_per_year
        )

        affordable = pd.DataFrame(
            {
                "Rate": rates,
                "Payment": payment,
                "Affordable Loan": affordable_loan,
                "Change": affordable_loan / affordable_loan.iloc[0] - 1,
            }
        )

        return apply_rounding(
            affordable, rounding if rounding is not None else self._rounding
        )

    @handle_errors
    def get_amortization_schedule(
        self,
        principal: float = 400_000,
        years: float = 30,
        payments_per_year: int = 12,
        date: str | None = None,
        rounding: int | None = None,
        show_columns: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Calculates the amortization schedule of a new 30-year fixed-rate mortgage at the US
        mortgage rate of a date: per monthly payment the part that is interest, the part that
        repays the loan and the balance left. Early payments are mostly interest, the more so the
        higher the rate: the first payment of a 400,000 US dollar loan at 7.18% is 2,393.33 of
        interest and 316.41 of repayment.

        The formula is as follows:

        - Interest(k) = Balance(k - 1) * r
        - Principal(k) = Payment - Interest(k)
        - Balance(k) = Balance(k - 1) - Principal(k), with Balance(0) = L

        Where L is the principal, r = Rate / payments per year the rate per payment and Payment
        the annuity payment of `get_loan_payment`. US mortgage rates are quoted as a nominal
        annual rate compounded monthly. The rate is the weekly average 30-year fixed rate of
        Freddie Mac's Primary Mortgage Market Survey on the date (the last survey on or before
        it), from FRED when a FRED API key is set and from Freddie Mac otherwise. For a rate
        of your own, use `loan_model.get_amortization_schedule`.

        Also known as: amortization table, mortgage schedule, repayment schedule, interest and
        principal per payment, remaining balance.

        Args:
            principal (float, optional): The amount borrowed. Defaults to 400,000.
            years (float, optional): The term of the loan in years. The rate stays the 30-year
                fixed mortgage rate for another term. Defaults to 30.
            payments_per_year (int, optional): The number of payments per year. Defaults to 12.
            date (str | None, optional): The date of the rate (YYYY-MM-DD). Defaults to None, the
                latest rate up to the end date.
            rounding (int | None, optional): The number of decimals to round the results to.
                Defaults to None, which uses the rounding of the class.
            show_columns (str | list[str] | None): The columns to keep, a name or a list of names, matched
                on every column level, or the rows when no column has those names (such as the line
                items of a financial statement). Names that do not exist are logged, and when none
                exist the result is returned in full. Defaults to None, which keeps every column.

        Returns:
            pd.DataFrame: Per payment number (1 to years * payments per year) the Rate (a
            decimal), the Payment, the Interest, the Principal repaid and the Balance after it.

        As an example:

        ```python
        from financetoolkit import FixedIncome

        fixedincome = FixedIncome(start_date='2026-01-01', end_date='2026-11-06')

        fixedincome.get_amortization_schedule().iloc[[0, 1, 2, 119, 239, 359]]
        ```

        Which returns:

        |   Payment Number |   Rate |   Payment |   Interest |   Principal |     Balance |
        |-----------------:|-------:|----------:|-----------:|------------:|------------:|
        |                1 | 0.0718 | 2709.7393 |  2393.3333 |    316.4060 | 399683.5940 |
        |                2 | 0.0718 | 2709.7393 |  2391.4402 |    318.2992 | 399365.2948 |
        |                3 | 0.0718 | 2709.7393 |  2389.5357 |    320.2036 | 399045.0912 |
        |              120 | 0.0718 | 2709.7393 |  2066.2399 |    643.4995 | 344689.0688 |
        |              240 | 0.0718 | 2709.7393 |  1393.1739 |   1316.5655 | 231525.8632 |
        |              360 | 0.0718 | 2709.7393 |    16.1168 |   2693.6225 |      0.0000 |
        """
        rates = self._get_mortgage_rates()
        if date is not None:
            rates = rates.loc[:date]
            if rates.empty:
                raise ValueError(
                    f"There is no 30-year mortgage rate on or before {date} after the "
                    f"start date {self._start_date}."
                )
        rate = float(rates.iloc[-1])

        schedule = loan_model.get_amortization_schedule(
            principal, rate, years, payments_per_year
        )
        schedule.insert(0, "Rate", rate)

        return apply_rounding(
            schedule, rounding if rounding is not None else self._rounding
        )
