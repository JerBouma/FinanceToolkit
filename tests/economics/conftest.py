"""Economics test configuration — all tests require --live flag."""

import os

import pytest

# The Economics methods backed by FRED rather than OECD; these additionally need a
# FRED_API_KEY in live mode. Kept as test-name substrings so a keyless contributor
# running --live sees these skip rather than error.
FRED_BACKED_TESTS = [
    "nonfarm_payrolls",
    "initial_jobless_claims",
    "retail_sales",
    "industrial_production_index",
    "housing_starts",
    "real_personal_income",
    "mortgage_rate_30_year",
    "recession_indicator",
    "commercial_real_estate_prices",
    "real_yield_curve",
    "breakeven_inflation_expectations",
]


@pytest.fixture(autouse=True)
def skip_if_not_live(request, live_mode):
    """Skip economics tests in pickle mode — all methods fetch live API data.

    This runs before the test itself, but session-scoped fixtures the test requests
    (like economics_module) are set up before autouse function-scoped fixtures, so
    the same guard also lives inside economics_module in the root conftest.
    """
    if not live_mode:
        pytest.skip("Economics tests require --live flag")

    if not os.environ.get("FRED_API_KEY") and any(
        name in request.node.name for name in FRED_BACKED_TESTS
    ):
        pytest.skip("This economics test requires a FRED_API_KEY")
