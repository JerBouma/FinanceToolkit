"""Dashboard Module"""

import streamlit as st

from pages import helpers, initalization_model, metrics_view, socials_model

st.set_page_config(page_title="Finance Toolkit Dashboard", page_icon="🛠️", layout="wide")

initalization_model.initalize_session_state(st.session_state)

if "is_expanded" not in st.session_state:
    st.session_state["is_expanded"] = True

if st.session_state["invalid_api_key"]:
    helpers.load_css("pages/assets/style.css")

    st.session_state["welcome_msg"].markdown(
        body="""
        ## Welcome to the Finance Toolkit Dashboard! 👋

        This dashboard is powered by the Finance Toolkit, an open-source toolkit in which 200+ relevant financial ratios,
        indicators and performance measurements are written down in the most simplistic way allowing for complete
        transparency of the method of calculation. Once you've entered you're API key you are ready to go!

        The key features of this dashboard are as follows:

        - 🏢 Compare a small or large selection of tickers for any period (10+ years for Premium users, max 5 years for
        Free users).
        - 📃 Plot any item from the Financial Statements, e.g. Shareholder's Equity from the Balance Sheet, Revenue from
        the Income Statement and Operating Cash Flow from the Cash Flow Statement.
        - 💸 Compare 150+ financial ratios, technical indicators, performance metrics and risk metrics.
        - 📈 Show quarterly results, growth rates, trailing twelve months (TTM) as well as tables (with the option
        to download the data)

        Find below a demonstration of what is possible once you have entered your API key.
        """
    )

    st.session_state["intro_video"].video(
        "pages/assets/financetoolkit-dashboard.mov",
        autoplay=True,
        muted=True,
        start_time=15,
        end_time=66,
    )

    st.session_state["mobile_user"].markdown(
        """
        <div class="mobile-only"><h3>📱 Mobile Users</h3><b>Given that you are viewing this page on mobile,
        please open the sidebar (arrow in the top left) to get started! Once you've entered your API key it
        should automatically proceed.</b></div>
        """,
        unsafe_allow_html=True,
    )

    initalization_model.create_api_key_sidebar(st.session_state)


if "perform_initalization" not in st.session_state:
    st.session_state["perform_initalization"] = False
    st.session_state["collect_data"] = False

if not st.session_state["invalid_api_key"]:
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

    st.session_state["welcome_msg"].empty()
    st.session_state["intro_video"].empty()
    st.session_state["mobile_user"].empty()

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
        - **TI** (Technical Indicators) includes all technical indicators such as Moving Averages, RSI, and more.

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
        initalization_model.initalize_financetoolkit(st.session_state)

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

socials_model.create_socials_sidebar()
