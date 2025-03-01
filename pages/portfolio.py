"""Portfolio Page Module"""

import tempfile

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from financetoolkit import Portfolio

MAX_ROWS = 2000

# Configure the page
st.set_page_config(page_title="Portfolio Dashboard", page_icon="💼", layout="wide")

if not st.session_state.get("portfolio"):
    st.session_state["instructions"] = st.markdown(
        """
        ## Welcome to the Portfolio Dashboard! 👋

        This dashboard allows you to upload a portfolio file, analyze performance insights, and review detailed
        transaction data.
        Please use the sidebar to upload your file and enter your API key, then click the "Load Portfolio"
        button to begin.

        With this dashboard you can:
        - View an overall portfolio overview including returns and benchmark comparisons.
        - Dive into detailed asset performance with easy-to-read tables and visualizations.
        - Analyze portfolio performance trends over customizable time periods.
        - Compare returns between different assets using interactive bar charts.
        - Evaluate transactions performance to highlight top and worst trades.
        - Inspect comprehensive transaction details including cumulative PnL and trade statistics.

        Find below a video showcasing the capabilities of this dashboard once a portfolio file is uploaded
        and the API key is entered.
        """
    )

    st.session_state["intro_video"] = st.video(
        "pages/assets/portfolio-dashboard.mov",
        autoplay=True,
        muted=True,
    )

# Sidebar details
with st.sidebar:
    st.image(
        "https://github.com/JerBouma/FinanceToolkit/assets/46355364/198d47bd-e1b3-492d-acc4-5d9f02d1d009"
    )
    st.title("💼 Portfolio Dashboard")
    st.markdown(
        "This dashboard allows you to upload a portfolio file, analyze performance insights,"
        "and review detailed transaction data. Please use the sidebar to upload your file "
        "and enter your API key, then click the 'Load Portfolio' button to begin."
    )

    try:
        url = "https://raw.githubusercontent.com/JerBouma/FinanceToolkit/main/financetoolkit/portfolio/example_datasets/example_portfolio.xlsx"
        response = requests.get(url, timeout=200)
        st.session_state["download button"] = st.download_button(
            label="Download Portfolio Template",
            data=response.content,
            file_name="example_portfolio.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception:
        st.markdown(
            "Please download the following file: https://github.com/JerBouma/FinanceToolkit/raw/refs/heads/main/"
            "financetoolkit/portfolio/example_datasets/example_portfolio.xlsx"
        )

    file_uploader_placeholder = st.sidebar.empty()

    st.sidebar.markdown(
        """
        **To obtain an API key (with a 15% discount), please visit [this page](https://www.jeroenbouma.com/fmp).**
        """
    )

    st.session_state["api_key"] = st.text_input(
        "API Key", key="api_key_sidebar", type="password"
    )

    st.session_state["portfolio_file"] = file_uploader_placeholder.file_uploader(
        "Upload a Portfolio file", type=["xlsx", "xls", "csv"]
    )

    def trigger():
        return not st.session_state["portfolio_file"] or not str(
            st.session_state["api_key"]
        )

    st.session_state["load_portfolio"] = st.button("Load Portfolio", disabled=trigger())

# Load Portfolio if button pressed
if (
    st.session_state["load_portfolio"]
    and st.session_state["portfolio_file"]
    and st.session_state["api_key"]
):
    st.session_state["instructions"].empty()
    st.session_state["intro_video"].empty()

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=st.session_state["portfolio_file"].name
    ) as tmp:
        tmp.write(st.session_state["portfolio_file"].getvalue())
        file_path = tmp.name
        file_name = st.session_state["portfolio_file"].name

    with st.spinner("Loading Portfolio..."):
        st.session_state["portfolio"] = Portfolio(
            portfolio_dataset=file_path, api_key=st.session_state["api_key"]
        )
        st.session_state["portfolio"].get_portfolio_overview()
    st.session_state["load_portfolio"] = False

# Show header once portfolio is loaded
if st.session_state.get("portfolio"):
    st.session_state["instructions"].empty()

    st.sidebar.header("Portfolio Settings")
    frequency = st.sidebar.selectbox(
        "Frequency",
        ["daily", "weekly", "monthly", "quarterly", "yearly"],
        index=3,
        key="frequency_sidebar",
    )

    # Portfolio Overview
    try:
        portfolio_overview = st.session_state["portfolio"].get_portfolio_overview()
        portfolio_overview = portfolio_overview.reset_index()

        summary_parts = []
        overall_portfolio = portfolio_overview[
            portfolio_overview["Identifier"] == "Portfolio"
        ]
        if not overall_portfolio.empty:
            overall = overall_portfolio.iloc[0]
            diff = overall["Return"] - overall["Benchmark Return"]
            summary_parts.append(
                f"Overall portfolio return: **{overall['Return']:.2%}** vs benchmark: "
                f"**{overall['Benchmark Return']:.2%}**. "
                + (
                    f"It outperformed the benchmark by **{diff:.2%}**."
                    if diff > 0
                    else (
                        f"It underperformed by **{-diff:.2%}**."
                        if diff < 0
                        else "It matched the benchmark."
                    )
                )
            )
        else:
            summary_parts.append("Overall portfolio data is not available.")

        performers = portfolio_overview[portfolio_overview["Identifier"] != "Portfolio"]
        if not performers.empty:
            top = performers.sort_values("Return", ascending=False).iloc[0]
            worst = performers.sort_values("Return", ascending=True).iloc[0]
            summary_parts.append(
                f"Top performer is **{top['Identifier']}** (Weight: **{top['Weight']:.2%}**, "
                f"Return: **{top['Return']:.2%}**), while the worst performer is "
                f"**{worst['Identifier']}** (Weight: **{worst['Weight']:.2%}**, "
                f"Return: **{worst['Return']:.2%}**)."
            )
        else:
            summary_parts.append("Performers' details are not available.")

        st.markdown(" ".join(summary_parts))

        styled_portfolio_overview = (
            portfolio_overview.style.map(
                lambda x: (
                    "color: green"
                    if isinstance(x, (int, float)) and x > 0
                    else "color: red" if isinstance(x, (int, float)) and x < 0 else ""
                ),
                subset=["Alpha", "Return", "Benchmark Return"],
            )
            .format(
                {
                    "Alpha": "{:.2%}",
                    "Return": "{:.2%}",
                    "Return Value": "{:.2f}",
                    "Benchmark Return": "{:.2%}",
                    "Weight": "{:.2%}",
                    "Volume": "{:,.0f}",
                    "Price": "{:,.2f}",
                    "Costs": "{:,.2f}",
                    "Invested": "{:,.2f}",
                    "Latest Price": "{:,.2f}",
                    "Latest Value": "{:,.2f}",
                }
            )
            .apply(
                lambda row: [
                    (
                        "background-color: #f0f0f0"
                        if row.name == portfolio_overview.index[-1]
                        else ""
                    )
                    for _ in row
                ],
                axis=1,
            )
        )
        st.dataframe(
            styled_portfolio_overview, use_container_width=True, hide_index=True
        )

        st.markdown(
            "Below is a bar chart comparing the portfolio's return to the benchmark. "
            "If the portfolio (blue) exceeds the benchmark (orange), it suggests outperformance."
        )

        if portfolio_overview["Identifier"].nunique() > 10:  # noqa
            portfolio_overview = portfolio_overview.sort_values(
                "Weight", ascending=False
            ).head(10)
            portfolio_overview = portfolio_overview.assign(
                is_portfolio=portfolio_overview["Identifier"].eq("Portfolio")
            )
            portfolio_overview = portfolio_overview.sort_values(by="is_portfolio").drop(
                columns="is_portfolio"
            )

        x_labels = portfolio_overview.apply(
            lambda row: (
                f"{row['Identifier']} ({row['Weight']:.2%})"
                if row["Identifier"] != "Portfolio"
                else row["Identifier"]
            ),
            axis=1,
        )

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=portfolio_overview["Benchmark Return"],
                name="Benchmark Return",
                text=portfolio_overview["Benchmark Return"],
                texttemplate="%{text:.2%}",
            )
        )
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=portfolio_overview["Return"],
                name="Return",
                text=portfolio_overview["Return"],
                texttemplate="%{text:.2%}",
            )
        )
        fig.update_layout(
            title="Benchmark vs Portfolio Return",
            barmode="group",
            xaxis_title="Identifier (Weight)",
            legend_title="Metric",
            showlegend=True,
        )
        fig.update_yaxes(title_text="Return", tickformat=".2%")
        for trace in fig.data:
            trace.hovertemplate = f"{trace.name}: %{{y:.2%}}<extra></extra>"
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error fetching portfolio overview data: {e}")

    try:
        positions_overview = st.session_state["portfolio"].get_positions_overview()
        st.markdown(
            "Explore the evolution of various portfolio metrics over time using the interactive graphs below."
        )
        tab_columns = positions_overview.columns.get_level_values(0).unique()

        # Create a tab for each column in the positions overview and display a graph for numeric columns
        tabs = st.tabs(list(tab_columns))

        for col, tab in zip(tab_columns, tabs):
            with tab:
                # Write commentary based on what the graph depicts
                if col in ["Current Weight", "Invested Weight"]:
                    commentary = (
                        f"The graph above shows the evolution of {col} as a stacked area chart, "
                        "allowing you to observe how each company's allocation contributes to the "
                        "overall portfolio over time."
                    )
                elif col in ["Return", "Benchmark Return", "Alpha"]:
                    commentary = (
                        f"This line chart displays trends of {col} for each company. "
                        "Notice how peaks and troughs can indicate periods of strong or weak performance, "
                        "potentially influenced by market events or company-specific factors."
                    )
                else:
                    commentary = (
                        f"The graph illustrates the trend of {col} over time for all companies. "
                        "This visualization helps in identifying long-term trends, seasonal patterns, "
                        "or anomalies in the data."
                    )
                st.markdown(commentary)

                fig = go.Figure()

                for company in positions_overview.columns.get_level_values(1).unique():
                    if positions_overview[col, company].dtype in [int, float]:
                        trace_kwargs = dict(
                            x=positions_overview.index.to_timestamp(),
                            y=positions_overview[col, company],
                            name=company,
                            mode="lines",
                        )
                    if col in ["Current Weight", "Invested Weight"]:
                        trace_kwargs["stackgroup"] = "one"
                    else:
                        trace_kwargs["fill"] = "tozeroy"
                    fig.add_trace(go.Scatter(**trace_kwargs))

                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title=col,
                    height=600,
                    showlegend=True,
                )
                st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error fetching positions overview data: {e}")

    # Portfolio & Transactions Performance
    try:
        portfolio_performance = st.session_state["portfolio"].get_portfolio_performance(
            period=frequency
        )
        portfolio_performance = portfolio_performance.reset_index()

        slider_dates = portfolio_performance["Date"].dt.to_timestamp()
        min_date = slider_dates.min().date()
        max_date = slider_dates.max().date()
        start_date, end_date = st.sidebar.slider(
            "Select period",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            key="period_slider",
        )
        portfolio_performance = portfolio_performance[
            (portfolio_performance["Date"] >= str(start_date))
            & (portfolio_performance["Date"] <= str(end_date))
        ]

        best_period = portfolio_performance.loc[
            portfolio_performance["Return"].idxmax()
        ]
        worst_period = portfolio_performance.loc[
            portfolio_performance["Return"].idxmin()
        ]
        best_date = best_period["Date"]
        worst_date = worst_period["Date"]

        best_period_data = portfolio_performance[
            portfolio_performance["Date"] == best_date
        ].sort_values(by="Return", ascending=False)
        worst_period_data = portfolio_performance[
            portfolio_performance["Date"] == worst_date
        ].sort_values(by="Return", ascending=True)

        best_identifiers = best_period_data.head(3)[["Identifier", "Return"]]
        worst_identifiers = worst_period_data.head(3)[["Identifier", "Return"]]

        best_str = ", ".join(
            f"{row['Identifier']} ({row['Return']:.2%})"
            for _, row in best_identifiers.iterrows()
        )
        worst_str = ", ".join(
            f"{row['Identifier']} ({row['Return']:.2%})"
            for _, row in worst_identifiers.iterrows()
        )

        st.markdown(
            f"In {best_date}, the portfolio performed best, driven by {best_str}, "
            f"while in {worst_date} it performed worst, driven by {worst_str}."
        )

        portfolio_performance["Date"] = (
            portfolio_performance["Date"]
            .mask(
                portfolio_performance["Date"] == portfolio_performance["Date"].shift()
            )
            .fillna(" ")
        )

        if portfolio_performance.size > pd.get_option("styler.render.max_elements"):
            portfolio_performance = portfolio_performance.tail(MAX_ROWS)

        first_valid_index = portfolio_performance[
            portfolio_performance["Date"] != " "
        ].index.min()
        portfolio_performance = portfolio_performance.loc[first_valid_index:]

        styled_portfolio_performance = portfolio_performance.style.map(
            lambda x: (
                "color: green"
                if isinstance(x, (int, float)) and x > 0
                else "color: red" if isinstance(x, (int, float)) and x < 0 else ""
            ),
            subset=["Return"],
        ).format(
            {
                "Invested Weight": "{:.2%}",
                "Current Weight": "{:.2%}",
                "Return": "{:.2%}",
                "Volume": "{:,.0f}",
                "Costs": "{:,.2f}",
                "Invested Amount": "{:,.2f}",
                "Current Value": "{:,.2f}",
            }
        )
        st.dataframe(
            styled_portfolio_performance, use_container_width=True, hide_index=True
        )

    except Exception as e:
        st.error(f"Error fetching portfolio performance data: {e}")

    # Transactions Performance
    try:
        transactions_performance = st.session_state[
            "portfolio"
        ].get_transactions_performance(period=frequency)
        transactions_performance = transactions_performance.reset_index()
        transactions_performance = transactions_performance[
            (transactions_performance["Date"] >= str(start_date))
            & (transactions_performance["Date"] <= str(end_date))
        ]
        if not transactions_performance.empty:
            best_transactions = transactions_performance.sort_values(
                "Return", ascending=False
            ).head(3)
            worst_transactions = transactions_performance.sort_values(
                "Return", ascending=True
            ).head(3)
            best_sentences = (
                "The best 3 transactions are "
                + ", ".join(
                    f"{row['Identifier']} in {row['Date']} (Return: {row['Return']:.2%})"
                    for _, row in best_transactions.iterrows()
                )
                + "."
            )
            worst_sentences = (
                "The worst 3 transactions are "
                + ", ".join(
                    f"{row['Identifier']} in {row['Date']} (Return: {row['Return']:.2%})"
                    for _, row in worst_transactions.iterrows()
                )
                + "."
            )
            st.markdown(best_sentences + " " + worst_sentences)

        transactions_performance["Date"] = (
            transactions_performance["Date"]
            .mask(
                transactions_performance["Date"]
                == transactions_performance["Date"].shift()
            )
            .fillna(" ")
        )

        styled_transactions_performance = transactions_performance.style.map(
            lambda x: (
                "color: green"
                if isinstance(x, (int, float)) and x > 0
                else "color: red" if isinstance(x, (int, float)) and x < 0 else ""
            ),
            subset=["Alpha", "Return", "Benchmark Return"],
        ).format(
            {
                "Alpha": "{:.2%}",
                "Return": "{:.2%}",
                "Benchmark Return": "{:.2%}",
                "Volume": "{:,.0f}",
                "Price": "{:,.2f}",
                "Costs": "{:,.2f}",
                "Invested Amount": "{:,.2f}",
                "Current Value": "{:,.2f}",
            }
        )
        st.dataframe(
            styled_transactions_performance, use_container_width=True, hide_index=True
        )
    except Exception as e:
        st.error(f"Error fetching transactions performance data: {e}")

    # Transactions Overview
    try:
        transactions_overview = st.session_state[
            "portfolio"
        ].get_transactions_overview()
        if not transactions_overview.empty:
            current_pnl = transactions_overview["Cumulative PnL"].iloc[-1]
            current_pnl = f"{current_pnl:.2f}"
            top_transactions = transactions_overview.nlargest(3, "% Return")
            bottom_transactions = transactions_overview.nsmallest(3, "% Return")
            top_str = ", ".join(
                f"{row['Name']} on {index[0]} ({row['% Return']:.2%})"
                for index, row in top_transactions.iterrows()
            )
            bottom_str = ", ".join(
                f"{row['Name']} on {index[0]} ({row['% Return']:.2%})"
                for index, row in bottom_transactions.iterrows()
            )
            sentence = (
                f"The Cumulative PnL was **{current_pnl}** where the top 3 transactions were "
                f"{top_str} and the bottom 3 transactions were {bottom_str}."
            )
            st.markdown(sentence)

        transactions_overview = (
            transactions_overview.drop("Identifier", axis=1)
            if "Identifier" in transactions_overview.columns
            else transactions_overview
        )

        styled_transactions_overview = transactions_overview.style.map(
            lambda x: (
                "color: green"
                if isinstance(x, (int, float)) and x > 0
                else "color: red" if isinstance(x, (int, float)) and x < 0 else ""
            ),
            subset=["% Return", "Return", "PnL", "Cumulative PnL"],
        ).format(
            {
                "Volume": "{:,.0f}",
                "Price": "{:,.2f}",
                "Costs": "{:,.2f}",
                "Benchmark Price": "{:,.2f}",
                "Invested Amount": "{:,.2f}",
                "Current Value": "{:,.2f}",
                "% Return": "{:.2%}",
                "Return": "{:.2f}",
                "PnL": "{:,.2f}",
                "Cumulative PnL": "{:,.2f}",
            }
        )
        st.dataframe(styled_transactions_overview, use_container_width=True)
    except Exception as e:
        st.error(f"Error fetching transactions overview data: {e}")
