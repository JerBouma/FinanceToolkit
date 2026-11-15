"""Loan Model Tests"""

import numpy as np
import pandas as pd
import pytest

from financetoolkit.fixedincome.loans import loan_model


def test_get_loan_payment():
    # A 400,000 loan over 30 years at a nominal 2.65% and 7.18%, compounded monthly.
    assert loan_model.get_loan_payment(400_000, 0.0265, 30) == pytest.approx(
        1611.8552, abs=1e-4
    )
    assert loan_model.get_loan_payment(400_000, 0.0718, 30) == pytest.approx(
        2709.7393, abs=1e-4
    )

    # A series of rates gives the payment at each of them; a zero rate repays in equal parts.
    rates = pd.Series([0.0265, 0.0, 0.0718], index=["a", "b", "c"], name="Rate")
    payments = loan_model.get_loan_payment(400_000, rates, 30)
    assert list(payments.index) == ["a", "b", "c"]
    assert payments["b"] == pytest.approx(400_000 / 360)
    assert payments["c"] == pytest.approx(2709.7393, abs=1e-4)

    # Quarterly payments use the quarterly rate.
    assert loan_model.get_loan_payment(100, 0.04, 1, 4) == pytest.approx(
        100 * 0.01 / (1 - 1.01**-4)
    )


def test_get_affordable_loan_inverts_the_payment():
    # The payment of a 400,000 loan at 2.65% carries 237,935 at 7.18%.
    payment = loan_model.get_loan_payment(400_000, 0.0265, 30)
    assert loan_model.get_affordable_loan(payment, 0.0718, 30) == pytest.approx(
        237_935.09, abs=0.01
    )

    rates = pd.Series([0.0265, 0.0, 0.0718])
    loans = loan_model.get_affordable_loan(payment, rates, 30)
    assert loans.iloc[0] == pytest.approx(400_000)
    assert loans.iloc[1] == pytest.approx(payment * 360)


def test_get_amortization_schedule():
    schedule = loan_model.get_amortization_schedule(400_000, 0.0718, 30)

    assert len(schedule) == 360  # noqa: PLR2004
    assert schedule.index[0] == 1
    assert schedule["Interest"].iloc[0] == pytest.approx(2393.3333, abs=1e-4)
    assert schedule["Principal"].iloc[0] == pytest.approx(316.4060, abs=1e-4)
    assert schedule["Balance"].iloc[-1] == 0

    # The closed-form balance equals the balance carried forward payment by payment.
    balance, rate = 400_000.0, 0.0718 / 12
    payment = schedule["Payment"].iloc[0]
    for interest, balance_after in zip(
        schedule["Interest"], schedule["Balance"], strict=True
    ):
        assert interest == pytest.approx(balance * rate)
        balance -= payment - balance * rate
        assert balance_after == pytest.approx(balance, abs=1e-6)

    # The interest over the life of the loan is the payments minus the principal.
    assert schedule["Interest"].sum() == pytest.approx(payment * 360 - 400_000)

    # Without interest the loan is repaid in equal parts.
    flat = loan_model.get_amortization_schedule(1200, 0.0, 1)
    assert np.allclose(flat["Principal"], 100)
    assert flat["Balance"].iloc[0] == pytest.approx(1100)
