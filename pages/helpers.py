"""Helpers Module"""

import streamlit as st

from financetoolkit import helpers


def check_api_key_status(api_key: str):
    """
    Check the status of an API key.

    Args:
        api_key (str): The API key to check.

    Returns:
        Tuple[bool, bool, str]: A tuple containing the following values:
            - invalid_api_key (bool): True if the API key is invalid, False otherwise.
            - premium_plan (bool): True if the API key is associated with a premium plan, False otherwise.
            - reason (str): The reason for the status of the API key. None if the API key is valid and associated
            with a premium plan.

    """
    determine_plan = helpers.get_financial_data(
        url=f"https://financialmodelingprep.com/api/v3/income-statement/AAPL?period=quarter&apikey={api_key}",
        sleep_timer=False,
    )

    premium_plan = True
    invalid_api_key = False
    reason = None

    if "INVALID API KEY" in determine_plan:
        invalid_api_key = True
        reason = "Invalid API Key."

    if "NOT AVAILABLE" in determine_plan:
        invalid_api_key = False
        premium_plan = False

    if "LIMIT REACH" in determine_plan:
        invalid_api_key = True
        premium_plan = False
        reason = "API Key Limit Reached."

    return invalid_api_key, premium_plan, reason


def load_css(file_name):
    """
    Loads and applies custom CSS styles from a file.

    Args:
        file_name (str): The name of the CSS file to load.

    Returns:
        Markdown object from Streamlit.
    """
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
