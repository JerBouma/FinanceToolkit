"""Metrics View"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from financetoolkit import Toolkit

# pylint: disable=R0914,W0212,R1735


def plot_data(
    variables: list[str],
    finance_toolkit: Toolkit,
    include_table: bool = False,
    growth: bool = False,
    trailing: int | None = None,
):
    """
    Plots financial data for the given variables using the Finance Toolkit.

    Args:
        variables (list[str]): A list of variables to plot. Each variable should be a string
            representing a specific financial statement or ratio. The variables should be
            formatted as follows:
            - Balance Sheet Statement: "variable_name (BS)"
            - Income Statement: "variable_name (IS)"
            - Cash Flow Statement: "variable_name (CF)"
            - Financial Ratios: "variable_name (FR)"
        finance_toolkit: An instance of the Finance Toolkit class that provides access to
            financial data.
        include_table (bool, optional): Whether to include a table with the plotted data.
            Defaults to False.
        growth (bool, optional): Whether to plot the growth rate of the data. Defaults to False.
        trailing (int | None, optional): The number of trailing periods to include in the plot.
            If None, all available periods will be included. Defaults to None.

    Returns:
        finance_toolkit: The Finance Toolkit instance used for plotting.
    """
    col1, col2 = st.columns(2)
    column_one = True

    trailing_value = 4 if trailing else None

    options = ["(BS)", "(IS)", "(CF)", "(FR)", "(MD)", "(TI)", "(PF)", "(RI)"]

    for variable in variables:
        if variable in ["TI", "PF", "RI"] and trailing_value:
            st.toast(
                f"No trailing data available for {variable}, showing regular chart.",
                icon="ℹ",
            )

        if "BS" in variable:
            # Balance Sheet Statement
            if trailing_value:
                st.toast(
                    f"No trailing data available for {variable}, showing regular chart.",
                    icon="❗️",
                )
            dataset = finance_toolkit.get_balance_sheet_statement(growth=growth)
        elif "IS" in variable:
            # Income Statement
            dataset = finance_toolkit.get_income_statement(
                growth=growth, trailing=trailing_value
            )
        elif "CF" in variable:
            # Cash Flow Statement
            dataset = finance_toolkit.get_cash_flow_statement(
                growth=growth, trailing=trailing_value
            )
        elif "FR" in variable:
            # Financial Ratios
            dataset = finance_toolkit.ratios.collect_all_ratios(
                growth=growth, trailing=trailing_value
            )
        elif "MD" in variable:
            if growth or trailing_value:
                st.toast(
                    f"No growth or trailing data available for {variable}, showing regular chart.",
                    icon="❗️",
                )
            # Historical Data
            dataset = finance_toolkit.get_historical_data().T.swaplevel(0, 1)

        elif "TI" in variable:
            # Technical Indicators
            dataset = finance_toolkit.technicals.collect_all_indicators(
                growth=growth
            ).T.swaplevel(0, 1)
        elif "PF" in variable:
            # Performance Metrics
            dataset = finance_toolkit.performance.collect_all_metrics(
                growth=growth
            ).T.swaplevel(0, 1)
        elif "RI" in variable:
            # Risk Metrics
            dataset = finance_toolkit.risk.collect_all_metrics(
                growth=growth
            ).T.swaplevel(0, 1)
        else:
            continue

        for option in options:
            if option in variable:
                variable_stripped = variable.replace(option, "").strip()
                break

        if (
            len(finance_toolkit._tickers) > 1
            or "(MD)" in variable
            or "(TI)" in variable
        ):
            dataset = dataset.loc[:, variable_stripped, :].T
        else:
            dataset = dataset.loc[variable_stripped, :].T

        dataset.index = dataset.index.to_timestamp()

        fig = px.line(dataset, title=variable, line_shape="linear")

        financetoolkit_template = go.layout.Template()
        financetoolkit_template.layout.annotations = [
            dict(
                name="Finance Toolkit",
                text="🛠️ Finance Toolkit",
                opacity=0.35,
                font=dict(color="black", size=12),
                xref="paper",
                yref="paper",
                x=1,
                y=-0.2,
                showarrow=False,
            )
        ]

        # Customize line chart appearance
        plot_mode = (
            "lines" if "(MD)" in variable or "(TI)" in variable else "lines+markers"
        )

        fig.update_traces(
            mode=plot_mode,
            texttemplate="%{y:.2s}",
            hovertemplate=None,
        )
        fig.update_layout(
            title={
                "text": variable,
                "y": 0.9,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            xaxis_title="Date",
            yaxis_title=None,
            legend_title=None,
            template=financetoolkit_template,
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.375, xanchor="right", x=1
            ),
            modebar_add=[
                "drawline",
                "drawopenpath",
                "drawclosedpath",
                "drawcircle",
                "drawrect",
                "eraseshape",
            ],
            modebar_remove=["toImage"],
            hovermode="x",
        )

        # In case there is no data at all, it shouldn't be shown either
        dataset = dataset.dropna(axis="rows", how="all")

        if not dataset.empty:
            freq = (
                "D"
                if "MD" in variable
                else (
                    "D"
                    if "TI" in variable
                    else "Q" if finance_toolkit._quarterly else "Y"
                )
            )

            if column_one:
                with col1:
                    column_one = False
                    st.plotly_chart(fig, use_container_width=True)

                if include_table:
                    with col2:
                        dataset.index = pd.PeriodIndex(dataset.index, freq=freq)
                        dataset.index.name = "Date"
                        st.dataframe(dataset, use_container_width=True)
            else:
                if include_table:
                    with col1:
                        dataset.index = pd.PeriodIndex(dataset.index, freq=freq)
                        dataset.index.name = "Date"
                        dataset = dataset.fillna(np.nan)
                        st.dataframe(dataset, use_container_width=True)

                with col2:
                    column_one = True
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.toast(f"No data available for {variable}", icon="❗️")

    return finance_toolkit
