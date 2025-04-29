"""Initialization Model"""

from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from financetoolkit import Toolkit
from pages import helpers

# pylint: disable=protected-access


def initalize_session_state(session_state: st.session_state):
    """
    Initializes the session state with default values for various variables.

    Parameters:
        session_state (SessionState): The session state object.

    Returns:
        SessionState: The updated session state object.
    """
    if "welcome_msg" not in session_state or session_state["invalid_api_key"]:
        session_state["welcome_msg"] = st.empty()

    if "intro_video" not in session_state or session_state["invalid_api_key"]:
        session_state["intro_video"] = st.empty()

    if "mobile_user" not in session_state or session_state["invalid_api_key"]:
        session_state["mobile_user"] = st.empty()

    if "sidebar_image" not in session_state or session_state["invalid_api_key"]:
        session_state["sidebar_image"] = st.sidebar.image(
            "https://github.com/JerBouma/FinanceToolkit/assets/46355364/198d47bd-e1b3-492d-acc4-5d9f02d1d009"
        )
    if "sidebar_title" not in session_state:
        session_state["sidebar_title"] = st.sidebar.title("🛠️ Finance Toolkit")

    if "api_key_explanation" not in session_state:
        session_state["api_key_explanation"] = st.sidebar.empty()

    if "api_key_container" not in session_state:
        session_state["api_key_container"] = st.sidebar.empty()

    if "api_key_checker" not in session_state:
        session_state["api_key_checker"] = st.sidebar.empty()

    if "api_key" not in session_state:
        session_state["api_key"] = st.sidebar.empty()
        session_state["api_key_value"] = None
        session_state["invalid_api_key"] = True

    if "portfolio_button" not in session_state:
        session_state["portfolio_button"] = st.sidebar.empty()

    if "tickers" not in session_state:
        session_state["tickers"] = []

    if "api_status_container" not in session_state:
        session_state["api_status_container"] = st.empty()

    if "foreign_exchanges_toggle" not in session_state:
        session_state["foreign_exchanges_toggle"] = st.empty()

    if "collect_data" not in session_state:
        session_state["collect_data"] = False

    if "historical_data" not in session_state:
        session_state["historical_data"] = st.empty()
    if "income_variables" not in session_state:
        session_state["income_variables"] = []

    if "foreign_exchanges_toggle" not in session_state:
        session_state["foreign_exchanges_toggle"] = st.empty()

    return session_state


def create_api_key_sidebar(session_state: st.session_state):
    """
    Creates the API key sidebar for entering and validating an API key.

    Args:
        session_state (SessionState): The session state object.

    Returns:
        SessionState: The updated session state object.
    """
    api_key_value = st.session_state["api_key_container"].text_input(
        "Enter your API key", type="password"
    )

    st.sidebar.markdown(
        """
        ----
        If you are interested in using your own portfolio with the Finance Toolkit, you can find
        the Portfolio Dashboard below.
        """
    )

    if st.sidebar.button("Go to the Portfolio Dashboard"):
        st.switch_page("./pages/portfolio.py")

    session_state["api_key_explanation"].markdown(
        """
        Access all **200+ metrics** from the Finance Toolkit right in your browser! Once you've entered your API key
        and pressed ENTER, you will automatically be able to access a rich history of financial data from any stock
        you choose.\n\n**To obtain an API key (with a 15% discount), please visit
        [this page](https://www.jeroenbouma.com/fmp).**
        """
    )

    if session_state["api_key_value"] != api_key_value and len(api_key_value) > 0:
        session_state["api_key_value"] = api_key_value

        (
            session_state["invalid_api_key"],
            session_state["premium_plan"],
            reason,
        ) = helpers.check_api_key_status(session_state["api_key_value"])

        if session_state["invalid_api_key"]:
            session_state["api_key_checker"].error(reason)

    return session_state


def create_configurations_section(session_state: st.session_state):
    """
    Creates the configurations sidebar for the Finance Toolkit app.

    Args:
        session_state (SessionState): The session state object.

    Returns:
        SessionState: The updated session state object.
    """
    st.markdown(
        """
        <style>
            .stMultiSelect [data-baseweb=select] span{
                max-width: 250px;
                font-size: 0.6rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "discovery" not in session_state:
        stock_list = pd.read_parquet("pages/stock_list.parquet")

        session_state["stocks_list"] = {
            f"{name} ({symbol})": symbol for symbol, name in stock_list["Name"].items()
        }
        session_state["stocks_list_excluding_foreign"] = {
            f"{name} ({symbol})": symbol
            for symbol, name in stock_list["Name"].items()
            if "." not in symbol
        }

    session_state["foreign_exchanges_toggle"] = st.toggle(
        "Include Foreign Exchanges",
        False,
        disabled=not session_state["premium_plan"],
        help="Whether to include foreign exchanges in the stock ticker selection.\n\nThis "
        "functionality is not available if you are using the Free plan.",
    )

    if session_state["foreign_exchanges_toggle"]:
        tickers = st.multiselect(
            "Enter the stock ticker", options=list(session_state["stocks_list"])
        )
    else:
        tickers = st.multiselect(
            "Enter the stock ticker",
            options=list(session_state["stocks_list_excluding_foreign"]),
        )

    session_state["tickers"] = [
        session_state["stocks_list"][ticker] for ticker in tickers
    ]

    col1, col2 = st.columns(2)

    with col1:
        session_state["start_date"] = st.date_input(
            "Start Date",
            value=(
                datetime.now() - timedelta(days=365 * 10)
                if session_state["premium_plan"]
                else datetime.now() - timedelta(days=365 * 5)
            ),
            min_value=(
                datetime(1980, 1, 1)
                if session_state["premium_plan"]
                else datetime.now() - timedelta(days=365 * 5)
            ),
            max_value=datetime.now(),
        )
    with col2:
        session_state["end_date"] = st.date_input(
            "End Date",
            value=datetime.now(),
            min_value=session_state["start_date"],
            max_value=datetime.now(),
        )

    metrics_list = list(pd.read_parquet("pages/metrics_list.parquet")[0].sort_values())

    session_state["metrics_list"] = st.multiselect("Metrics", options=metrics_list)

    st.divider()

    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])

    st.markdown(
        """
    <style>
    button {
        height: auto;
        padding-left: 50px !important;
        padding-right: 50px !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    session_state["collect_data"] = col1.button("Collect Data", type="primary")

    session_state["quarterly_data"] = col2.toggle(
        "Quarterly",
        False,
        disabled=not session_state["premium_plan"],
        help="Whether to collect quarterly financial statements."
        "\n\nThis functionality is not available if you are using the Free plan.",
    )

    if session_state["quarterly_data"]:
        session_state["ttm_rates"] = col3.toggle(
            "TTM",
            False,
            help="Whether to calculate the Twelve Trailing Months (TTM) values, available when using quarterly data."
            "\n\nThis functionality is not available if you are using the Free plan.",
        )
    else:
        session_state["ttm_rates"] = col3.toggle(
            "TTM",
            False,
            disabled=True,
            help="Whether to calculate the Twelve Trailing Months (TTM) values, available when using quarterly data."
            "\n\nThis functionality is not available if you are using the Free plan.",
        )

    session_state["growth_rates"] = col4.toggle(
        "Growth Rates",
        False,
        help="Whether to show the growth rates (relative change to the previous value).",
    )

    session_state["include_table"] = col5.toggle(
        "Include Table",
        False,
        help="Whether to show a table with the data plotted. This also allows you to download the data.",
    )

    return session_state


def initalize_financetoolkit(session_state: st.session_state):
    """
    Initializes the FinanceToolkit instance and collects data if necessary.

    Args:
        session_state (dict): The session state containing the required parameters.

    Returns:
        dict: The updated session state.

    """
    if (
        "financetoolkit" not in session_state
        or session_state["tickers"] != session_state["financetoolkit"]._tickers
        or session_state["quarterly_data"] != session_state["financetoolkit"]._quarterly
        or session_state["start_date"].strftime("%Y-%m-%d")
        != session_state["financetoolkit"]._start_date
        or session_state["end_date"].strftime("%Y-%m-%d")
        != session_state["financetoolkit"]._end_date
    ):
        instance = Toolkit(
            tickers=session_state["tickers"],
            api_key=session_state["api_key_value"],
            quarterly=session_state["quarterly_data"],
            start_date=(
                session_state["start_date"].strftime("%Y-%m-%d")
                if session_state["start_date"]
                else None
            ),
            end_date=(
                session_state["end_date"].strftime("%Y-%m-%d")
                if session_state["end_date"]
                else None
            ),
            progress_bar=False,
            enforce_source="FinancialModelingPrep",
        )

        session_state["financetoolkit"] = instance

    return session_state
