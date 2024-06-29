import streamlit as st

from app import initalization_model, metrics_view, socials_model

st.set_page_config(
    page_title="Finance Toolkit in Streamlit", page_icon="📈", layout="wide"
)


st.session_state = initalization_model.initalize_session_state(st.session_state)

if "is_expanded" not in st.session_state:
    st.session_state["is_expanded"] = True

if st.session_state["invalid_api_key"]:
    st.session_state["welcome_msg"].markdown(
        """
        ## Welcome to the Finance Toolkit dashboard! 👋

        While browsing a variety of websites, I repeatedly observed significant fluctuations in the same financial metric
        among different sources.
        Similarly, the reported financial statements often didn't line up, and there was limited information on the
        methodology used to calculate each metric.

        For example, Microsoft's Price-to-Earnings (PE) ratio on the 6th of May, 2023 is reported to
        be 28.93 (Stockopedia), 32.05 (Morningstar),
        32.66 (Macrotrends), 33.09 (Finance Charts), 33.66 (Y Charts), 33.67 (Wall Street Journal), 33.80
        (Yahoo Finance) and 34.4 (Companies Market Cap). All of these calculations are correct, however the
        method of calculation varies leading to different results. Therefore, collecting data from multiple
        sources can lead to wrong interpretation of the results given that one source could apply a different
        definition than another. And that is, if that definition is even available as often the underlying
        methods are hidden behind a paid subscription.

        **This is why I designed the Finance Toolkit**, this is an open-source toolkit in which all relevant financial
        ratios 200+ indicators
        and performance measurements are written down in the most simplistic way allowing for complete transparency
        of the method of calculation.

        This dashboard is powered entirely by the Finance Toolkit and gves access to all relevant financial ratios,
        indicators and performance measurements. Key features of the dashboard are:

        - Compare any amount of tickers of any range of period (10+ years for Premium users, 5 years for Free users).
        - Plot any item from the Financial Statements (Balance Sheet, Income Statement, Cash Flow Statement).
        - Compare a large selection of financial ratios, technical indicators, performance metrics and risk metrics.
        - Show quarterly results, growth rates, trailing twelve months (TTM) as well as tables (with download
        to CSV option)

        Once you've entered you're API key you are ready to go!
        """
    )

    st.session_state = initalization_model.create_api_key_sidebar(st.session_state)


if "perform_initalization" not in st.session_state:
    st.session_state["perform_initalization"] = False

if not st.session_state["invalid_api_key"]:
    st.session_state["welcome_msg"].empty()

    st.title("🛠️ Finance Toolkit")
    st.markdown(
        """
        The Finance Toolkit is an open-source toolkit in which 200+ relevant financial ratios, indicators and
        performance measurements are written down
        in the most simplistic way allowing for complete transparency of the method of calculation. This enables
        you to avoid dependence on metrics
        from other providers that do not provide their methods. With a large selection of financial statements
        in hand, it facilitates
        streamlined calculations, promoting the adoption of a consistent and universally understood methods and formulas.

        **See the explanation section for more information how to use this dashboard and look at the configurations
        section to get started!**
    """
    )

    with st.expander("Explanation", expanded=False):
        st.markdown(
            unsafe_allow_html=True,
            body="""
        Within the configurations sections you have the following options:
        - **Stock Ticker**: you can select out over 30.000 companies. You can select any amount
        of tickers here to collect data from. These tickers will be plotted together in a graph once you press the button.
        If you are looking for foreign exchanges (outside of the US) you can enable the toggle to show these. Note
        that this only applies for Premium users.
        - **Start Date & End Date**: you can select the start and end date for the data you want to collect, when you have
        a Premium plan this defaults to 10 years, for the Free plan this defaults to 5 years (maximum period).
        - **Metrics:** this includes a wide variety of metrics from the Finance Toolkit. You can select any amount
        of metrics here which will be plotted in individual graphs.
        - **Quarterly Toggle**: this toggle allows you to switch between quarterly and annual data.
        - **Twelve Trailing Months (TTM)**: this toggle allows you to calculate the Twelve Trailing Months values,
        available when using quarterly data.
        - **Growth Rates**: this toggle allows you to calculate the growth rates (percentage change) of the data.
        - **Include Table**: this toggle allows you to include a table with the plotted data, this also allows
        you to download the data to a CSV file.

        The list below features all available metrics from the Finance Toolkit. In brackets, you will find the
        abbreviation to identify the respective type of metric.
        - **BS** (Balance Sheet Statement) includes all assets, liabilities, and equity items of a company.
        - **IS** (Income Statement) includes all revenue, expenses, and profit items of a company.
        - **CF** (Cash Flow Statement) includes all cash inflows and outflows of a company.
        - **FR** (Financial Ratios) includes all financial ratios that can be calculated from the financial statements.
        - **MD** (Market Data) includes all market data such as adjusted close, cumulative return, and more.
        - **PF** (Performance Metrics) includes all performance metrics such as Sharpe ratio, Sortino ratio, and more.
        - **RI** (Risk Metrics) includes all risk metrics such as Value at Risk, Conditional Value at Risk, and more.

        Once you have selected all the configurations you can press the **Collect Data** button to collect data and plot
        (and show tables) according to your selections. You can always change the configurations and press the button
        again to update the plots.

        *It's good to note that data is saved and won't be collected again when this is not required.* You can change
        up the metrics or enable the growth, TTM or table toggle without needing to recollect data. However, if you
        change the tickers, date range or quarterly toggle the dashboard will collect data again. This is however
        relatively quick!

        <span style="font-size:0.7em;">
        <b>DISCLAIMER</b><br>
        Please note that I am not a part of FinancialModelingPrep, the data provider used. I've merely picked them as the
        primary data provider given that they have a generous free tier and fair pricing compared to other providers.
        Therefore, any questions related to the data should go through
        <a href="https://site.financialmodelingprep.com/contact">their contact form</a>.
        When it comes to any type of ratios, performance metrics, risk metrics, technical indicators or other indicators,
        feel free to reach out to me as this is my own work.
        </span>
        """,
        )
    with st.expander("Configurations", expanded=True):
        # Initialize the sidebar which includes the ticker selection, start and end date selection and the
        # financial metric selection
        st.session_state = initalization_model.create_configurations_section(
            st.session_state
        )

    if st.session_state["collect_data"]:
        # Collect all historical data required to perform all calculations within
        # the Finance Toolkit based on the input from the sidebar
        st.session_state = initalization_model.initalize_financetoolkit(
            st.session_state
        )

        st.header("Results", divider="green")

        if not st.session_state["tickers"]:
            st.toast("Please select at least one ticker to collect data.", icon="❗️")
        else:
            with st.spinner("Collecting data.."):
                if len(st.session_state["metrics_list"]) > 0:
                    st.session_state["financetoolkit"] = metrics_view.plot_data(
                        variables=st.session_state["metrics_list"],
                        finance_toolkit=st.session_state["financetoolkit"],
                        include_table=st.session_state["include_table"],
                        growth=st.session_state["growth_rates"],
                        trailing=st.session_state["ttm_rates"],
                    )

        st.session_state["perform_initalization"] = False

    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"][aria-expanded="true"]{
                display: none;
            }
        </style>
    """,
        unsafe_allow_html=True,
    )


socials_model.create_socials_sidebar()
