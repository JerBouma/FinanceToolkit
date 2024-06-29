from financetoolkit import helpers


def check_api_key_status(api_key: str):
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
