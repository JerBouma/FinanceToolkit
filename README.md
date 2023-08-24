![FinanceToolkitCropped](https://github.com/JerBouma/FinanceToolkit/assets/46355364/198d47bd-e1b3-492d-acc4-5d9f02d1d009)

[![BuyMeACoffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Donate-brightgreen?logo=buymeacoffee)](https://www.buymeacoffee.com/jerbouma)
[![Documentation](https://img.shields.io/badge/Read_More-Documentation?logo=readme&label=Documentation)](https://www.jeroenbouma.com/projects/financetoolkit)
[![Discord](https://img.shields.io/badge/JerBouma-hello?logo=discord&label=Discord)](https://discord.com/)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/financetoolkit)](https://pypi.org/project/FinanceToolkit/)
[![PYPI Version](https://img.shields.io/pypi/v/FinanceToolkit)](https://pypi.org/project/FinanceToolkit/)
[![PYPI Downloads](https://static.pepy.tech/badge/financetoolkit/month)](https://pepy.tech/project/financetoolkit)

While browsing a variety of websites, I kept finding that the same financial metric can greatly vary per source and so do the financial statements reported while little information is given how the metric was calculated.

For example, Microsoft's Price-to-Earnings (PE) ratio on the 6th of May, 2023 is reported to be 28.93 (Stockopedia), 32.05 (Morningstar), 32.66 (Macrotrends), 33.09 (Finance Charts), 33.66 (Y Charts), 33.67 (Wall Street Journal), 33.80 (Yahoo Finance) and 34.4 (Companies Market Cap). All of these calculations are correct, however the method applied varies leading to different results. Therefore, collecting data from multiple sources can lead to wrong interpretation of the results given that one source could be applying a different calculation method than another. And that is, if it is even freely available. Often the calculation is hidden behind a paid subscription.

**This is why I designed the FinanceToolkit**, this is an open-source toolkit in which all relevant financial ratios (50+), indicators and performance measurements are written down in the most simplistic way allowing for complete transparency of the calculation method. This allows you to not have to rely on metrics from other providers and, given a financial statement, allow for efficient manual calculations. This leads to one uniform method of calculation being applied that is available and understood by everyone.

The Finance Toolkit is complimented very well with the [Finance Database 🌎](https://github.com/JerBouma/FinanceDatabase), a database that features 300.000+ symbols containing Equities, ETFs, Funds, Indices, Currencies, Cryptocurrencies and Money Markets. By utilising both, it is possible to do a fully-fledged competitive analysis with the tickers found from the FinanceDatabase inputted into the FinanceToolkit.

<p align="center">
    <img src="examples/Finance Toolkit - 6. Video Demo.gif" alt="Finance Toolkit Illustration" width="100%" onerror="this.style.display = 'none'"/>
</p>

---

# Table of Contents

1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
    1. [Using the Finance Toolkit](#using-the-finance-toolkit)
    2. [Defining Custom Ratios](#defining-custom-ratios)
    3. [Calling Functions Directly](#calling-functions-directly)
    4. [Working with other Datasets](#working-with-other-datasets)
3. [Contact](#contact)

# Installation

Before installation, consider starring the project on GitHub which helps others find the project as well. 

<img width="1415" alt="image" src="https://github.com/JerBouma/FinanceToolkit/assets/46355364/014109fe-0c68-47d4-99bd-217c69dcea8d">

To install the FinanceToolkit it simply requires the following:

```
pip install financetoolkit
````

Then within Python use:

```python
from financetoolkit import Toolkit
```
To be able to get started, you need to obtain an API Key from FinancialModelingPrep. This is used to gain access to 30+ years of financial statement both annually and quarterly. Note that the Free plan is limited to 250 requests each day, 5 years of data and only features companies listed on US exchanges.

___ 

<b><div align="center">Obtain an API Key from FinancialModelingPrep <a href="https://site.financialmodelingprep.com/developer/docs/pricing/jeroen/">here</a>.</div></b>
___

Through the link you are able to subscribe for the free plan and also premium plans at a **15% discount**. This is an affiliate link and thus supports the project at the same time. I have chosen FinancialModelingPrep as a source as I find it to be the most transparent, reliable and at an affordable price. I have yet to find a platform offering such low prices for the amount of data offered. When you notice that the data is inaccurate or have any other issue related to the data, note that I simply provide the means to access this data and I am not responsible for the accuracy of the data itself. For this, use [their contact form](https://site.financialmodelingprep.com/contact) or provide the data yourself. 

# Basic Usage

This section explains in detail how the Finance Toolkit can utilitised effectively. Also see the Jupyter Notebook in which you can run the examples also demonstrated here.

___ 

<b><div align="center">Find a variety of How-To Guides including Code Documentation for the FinanceToolkit <a href="https://www.jeroenbouma.com/projects/financetoolkit">here</a>.</div></b>
___

Within this package the following things are included:

- Company profiles (`get_profile`), including country, sector, ISIN and general characteristics (from FinancialModelingPrep)
- Company quotes (`get_quote`), including 52 week highs and lows, volume metrics and current shares outstanding (from FinancialModelingPrep)
- Company ratings (`get_rating`), based on key indicators like PE and DE ratios (from FinancialModelingPrep)
- Historical market data (`get_historical_data`), which can be retrieved on a daily, weekly, monthly and yearly basis. This includes OHLC, dividends, returns, cumulative returns and volatility calculations for each corresponding period. (from Yahoo Finance)
- Treasury Rates (`get_treasury_data`) for several months and several years over the last 3 months which allows yield curves to be constructed (from FinancialModelingPrep)
- Analyst Estimates (`get_analyst_estimates`) that show the expected EPS and Revenue from the past and future from a range of analysts (from FinancialModelingPrep)
- Earnings Calendar (`get_earnings_calendar`) which shows the exact dates earnings are released in the past and in the future including expectations (from FinancialModelingPrep)
- Revenue Geographic Segmentation (`get_revenue_geographic_segmentation`) which shows the revenue per company from each country and Revenue Product Segmentation (`get_revenue_product_segmenttion`) which shows the revenue per company from each product (from FinancialModelingPrep)
- Balance Sheet Statements (`get_balance_sheet_statement`), Income Statements (`get_income_statement`), Cash Flow Statements (`get_cash_flow_statement`) and Statistics Statement (`get_statistics_statement`), obtainable from FinancialModelingPrep or the source of your choosing through custom input. These functions are accompanied with a normalization function so that for any source, the same ratio analysis can be performed. Please see [this Jupyter Notebook](https://www.jeroenbouma.com/projects/financetoolkit/external-datasets) that explains how to use a custom source.
- Efficiency ratios (`ratios.collect_efficiency_ratios`), liquidity ratios (`ratios.collect_liquidity_ratios`), profitability ratios (`ratios._collect_profitability_ratios`), solvency ratios (`ratios.collect_solvency_ratios`) and valuation ratios (`ratios.collect_valuation_ratios`) functionality that automatically calculates the most important ratios based on the inputted balance sheet, income and cash flow statements. Any of the underlying ratios can also be called individually such as `ratios.get_return_on_equity`. Next to that, it is also possible to input your own custom ratios (`ratios.collect_custom_ratios`). See also [this Notebook](https://www.jeroenbouma.com/projects/financetoolkit/custom-ratios) or [this section](#defining-custom-ratios) for more information.
- Models like DUPONT analysis (`models.get_extended_dupont_analysis`) or Enterprise Breakdown (`models.get_enterprise_value_breakdown`) that can be used to perform in-depth financial analysis through a single function. These functions combine much of the functionality throughout the Toolkit to provide advanced calculations. 

The dependencies of the package are on purpose *very slim* so that it will work well with any combination of packages and not result in conflicts. I've also been careful with my selection in which I leave out functionality like technical analysis in which [ta-lib](https://ta-lib.org/) does an excellent job as well as portfolio attribution and optimisation in which [Riskfolio-lib](https://riskfolio-lib.readthedocs.io/en/latest/index.html) shines.

## Using the Finance Toolkit

A basic example of how to initialise the Finance Toolkit is shown below, also see [this notebook](https://www.jeroenbouma.com/projects/financetoolkit/getting-started) for a detailed Getting Started guide as well as [this notebook](https://www.jeroenbouma.com/projects/financetoolkit/finance-database) that includes the [Finance Database 🌎](https://www.jeroenbouma.com/projects/financedatabase) and a proper financial analysis.

````python
from financetoolkit import Toolkit

companies = Toolkit(['AAPL', 'MSFT'], api_key="FMP_KEY", start_date='2017-12-31')

# a Historical example
historical_data = companies.get_historical_data()

# a Financial Statement example
balance_sheet_statement = companies.get_balance_sheet_statement()

# a Ratios example
profitability_ratios = companies.ratios.collect_profitability_ratios()

# a Models example
extended_dupont_analysis = companies.models.get_extended_dupont_analysis()

# Show the profitability ratios for Apple
profitability_ratios.loc['AAPL']
````

This returns the following output for `profitability_ratios.loc['AAPL]`. Omitting `.loc['AAPL']` will return the result for both AAPL and MSFT.


|                                             |     2018 |     2019 |     2020 |     2021 |     2022 |
|:--------------------------------------------|---------:|---------:|---------:|---------:|---------:|
| Gross Margin                                | 0.383437 | 0.378178 | 0.382332 | 0.417794 | 0.433096 |
| Operating Margin                            | 0.26694  | 0.24572  | 0.241473 | 0.297824 | 0.302887 |
| Net Profit Margin                           | 0.224142 | 0.212381 | 0.209136 | 0.258818 | 0.253096 |
| Interest Burden Ratio                       | 1.02828  | 1.02827  | 1.01211  | 1.00237  | 0.997204 |
| Income Before Tax Profit Margin             | 0.274489 | 0.252666 | 0.244398 | 0.298529 | 0.30204  |
| Effective Tax Rate                          | 0.183422 | 0.159438 | 0.144282 | 0.133023 | 0.162045 |
| Return on Assets (ROA)                      | 0.162775 | 0.16323  | 0.177256 | 0.269742 | 0.282924 |
| Return on Equity (ROE)                      | 0.555601 | 0.610645 | 0.878664 | 1.50071  | 1.96959  |
| Return on Invested Capital (ROIC)           | 0.269858 | 0.293721 | 0.344126 | 0.503852 | 0.562645 |
| Return on Capital Employed (ROCE)           | 0.305968 | 0.297739 | 0.320207 | 0.495972 | 0.613937 |
| Return on Tangible Assets                   | 0.555601 | 0.610645 | 0.878664 | 1.50071  | 1.96959  |
| Income Quality Ratio                        | 1.30073  | 1.25581  | 1.4052   | 1.09884  | 1.22392  |
| Net Income per EBT                          | 0.816578 | 0.840562 | 0.855718 | 0.866977 | 0.837955 |
| Free Cash Flow to Operating Cash Flow Ratio | 0.828073 | 0.848756 | 0.909401 | 0.893452 | 0.912338 |
| EBT to EBIT Ratio                           | 0.957448 | 0.948408 | 0.958936 | 0.976353 | 0.975982 |
| EBIT to Revenue                             | 0.286688 | 0.26641  | 0.254864 | 0.305759 | 0.309473 |

## Defining Custom Ratios

The Finance Toolkit has an abundance of financial ratios, however it could be that you are looking for a specific ratio that is currently not provided. First and foremost, I encourage you to [create a Pull Request](https://github.com/JerBouma/FinanceToolkit/pulls) to add these ratios in but there is also an option to add custom ratios as follows. Find a Notebook example [here](https://www.jeroenbouma.com/projects/financetoolkit/custom-ratios).

Define how each custom ratio needs to be calculated. This can be any of the following structures:

- **Simple operations such as:** `'Quick Assets': 'Cash and Short Term Investments + Accounts Receivable'`
- **Working with multiple operations:** `'Cash Op Expenses':'Cost of Goods Sold + Selling, General and Administrative Expenses - Depreciation and Amortization'`,
- **Using curly brackets:** `'WC / Net Income as %': '(Working Capital / Net Income) * 100'`,
- **Defining a criteria:** `'Large Revenues': 'Revenue > 1000000000'`,
- **Using actual numbers:**  `'Daily Cash Op Expenses': 'Cash Op Expenses / 365'`,
- **Combining earlier defined formulas:** `'Defensive Interval':'Quick Assets / Daily Cash Op Expenses'`

Not that it is important you follow the NAME - FORMULA format and that you adhere to the normalization files naming. This can be viewed relatively easy by initializing the Toolkit and running for example `get_balance_sheet_statement`.

```python
custom_ratios = {
    'WC / Net Income as %': '(Working Capital / Net Income) * 100',
    'Large Revenues': 'Revenue > 1000000000',
    'Quick Assets': 'Cash and Short Term Investments + Accounts Receivable',
    'Cash Op Expenses':'Cost of Goods Sold + Selling, General and Administrative Expenses '
    '- Depreciation and Amortization',
    'Daily Cash Op Expenses': 'Cash Op Expenses / 365',
    'Defensive Interval':'Quick Assets / Daily Cash Op Expenses'
}
```

Initializing the Finance Toolkit. Make sure to set the parameter `custom_ratios` with the above dictionary. Note that `quarterly=True` doesn't work without a Premium plan.

```python
from financetoolkit import Toolkit

# Initialize the Finance Toolkit
companies = Toolkit(
    ["AAPL", "MSFT", "GOOGL", "AMZN"], api_key="FMP_KEY", start_date="2022-08-10",
    custom_ratios=custom_ratios, quarterly=True
)

# Obtain the custom ratios
companies.ratios.collect_custom_ratios()
```

By then running `ratios.collect_custom_ratios` it automatically calculates the given ratios. Note the ratios `Daily Cash Op Expenses` and `Defensive Interval` which rely on an earlier defined ratio (`Cash Op Expenses`). This is an example of how you can create a custom ratio based on another custom ratio.

Below you can find the custom ratios from AMZN (Amazon) which has been selected with `.loc['AMZN]`. The Notebook as found [here](https://www.jeroenbouma.com/projects/financetoolkit/getting-started) shows the full output.

|                        |         2022Q4 |         2023Q1 |         2023Q2 |   2023Q3 |
|:-----------------------|---------------:|---------------:|---------------:|---------:|
| Cash Op Expenses       |    2.1856e+10  |    1.9972e+10  |    2.1322e+10  |      nan |
| Daily Cash Op Expenses |    5.98795e+07 |    5.47178e+07 |    5.84164e+07 |      nan |
| Defensive Interval     | 2260.22        | 2592.34        | 2738.1         |      nan |
| Large Revenues         |    1           |    1           |    1           |        0 |
| Quick Assets           |    1.35341e+11 |    1.41847e+11 |    1.5995e+11  |      nan |
| WC / Net Income as %   |  463.349       |  427.335       |  398.924       |      nan |

In case you forgot a ratio or would like to insert new ratios, you can directly pass the custom_ratios dictionary to the function.

```python
new_custom_ratios = {
    'WC / Net Income': 'Working Capital / Net Income',
    'Low Revenues': 'Revenue < 1000000000',
    'Quick Assets': 'Cash and Short Term Investments + Accounts Receivable',
    'Cash Op Expenses':'Cost of Goods Sold + Selling, General and Administrative Expenses '
    '- Depreciation and Amortization',
    'Monthly Cash Op Expenses': 'Cash Op Expenses / 30',
    'Defensive Interval':'Quick Assets / Monthly Cash Op Expenses'
}

# Return the new ratios by passing the dictionary into the function
companies.ratios.collect_custom_ratios(custom_ratios_dict=new_custom_ratios)
```

Which returns the following for GOOGL (Google):

|                          |        2022Q3 |       2022Q4 |        2023Q1 |        2023Q2 |   2023Q3 |
|:-------------------------|--------------:|-------------:|--------------:|--------------:|---------:|
| Cash Op Expenses         |   3.6925e+10  |   4.2495e+10 |   3.6722e+10  |   3.7896e+10  |      nan |
| Defensive Interval       | 123.847       | 108.733      | 123.472       | 124.395       |      nan |
| Low Revenues             |   0           |   0          |   0           |   0           |        0 |
| Monthly Cash Op Expenses |   1.23083e+09 |   1.4165e+09 |   1.22407e+09 |   1.2632e+09  |      nan |
| Quick Assets             |   1.52435e+11 |   1.5402e+11 |   1.51138e+11 |   1.57136e+11 |      nan |
| WC / Net Income          |   7.19842     |   7.00932    |   6.1877      |   4.95857     |      nan |


## Calling Functions Directly

It also possible to call any ratio or model directly as shown below. This allows access to 50+ ratios with custom data. Also see [this notebook](https://www.jeroenbouma.com/projects/financetoolkit/functional-toolkit).

```python
import pandas as pd
import numpy as np

from financetoolkit.models import dupont

years = [2018, 2019, 2020, 2021, 2022]

dupont.get_dupont_analysis(
    net_income=pd.Series(
        [59531000000, 55256000000, 57411000000, 94680000000, 99803000000], index=years
    ),
    total_revenue=pd.Series(
        [265595000000, 260174000000, 274515000000, 365817000000, 394328000000],
        index=years,
    ),
    total_assets_begin=pd.Series(
        [np.nan, 365725000000, 338516000000, 323888000000, 351002000000],
        index=years,
    ),
    total_assets_end=pd.Series(
        [365725000000, 338516000000, 323888000000, 351002000000, 352755000000],
        index=years,
    ),
    total_equity_begin=pd.Series(
        [np.nan, 107147000000, 90488000000, 65339000000, 63090000000], index=years
    ),
    total_equity_end=pd.Series(
        [107147000000, 90488000000, 65339000000, 63090000000, 50672000000], index=years
    ),
)
```

This returns the following table which closely resembles a proper Dupont analysis for Apple at their given reporting dates in October:

|                   |       2018 |     2019 |     2020 |     2021 |     2022 |
|:------------------|-----------:|---------:|---------:|---------:|---------:|
| Net Profit Margin |   0.224142 | 0.212381 | 0.209136 | 0.258818 | 0.253096 |
| Asset Turnover    | nan        | 0.738878 | 0.828845 | 1.08408  | 1.12064  |
| Equity Multiplier | nan        | 3.56334  | 4.25089  | 5.25497  | 6.18622  |
| Return on Equity  | nan        | 0.559172 | 0.736856 | 1.47443  | 1.75459  |

## Working with other Datasets

The Finance Toolkit has the ability to leverage custom datasets from any data provider as well. This makes it possible to work with your preferred data and not be limited to the data source the Finance Toolkit currently provides. A detailed example can be found [here](https://www.jeroenbouma.com/projects/financetoolkit/external-datasets) but to get started see the code below.

```python
from financetoolkit import Toolkit

# Initialize the Finance Toolkit
companies = Toolkit(['AAPL', 'MSFT'])

# Copy the normalization files
companies.get_normalization_files()
```
This copies over three files, `balance.csv`, `income.csv`, `cash.csv` and `statistics.csv` (not required) which will contain a structure like the following:

![Normalization Format](https://github.com/JerBouma/FinanceToolkit/assets/46355364/ea4ebf87-1a27-4c40-a1fb-40d0eb0634bc)

By replacing the first column with the names from your dataset (e.g. replace `cashAndCashEquivalents` with `Cash` if this is how it is called in your dataset), it will automatically normalize the dataset when you initialize the Finance Toolkit. Note that the DataFrame needs to be a multi-index in case you use multiple tickers structured as `Ticker x Financial Statement Item x Periods`.

As an example:

![Dataset Example](https://github.com/JerBouma/FinanceToolkit/assets/46355364/c2fb69d9-d599-4ba4-8163-e69abc164e88)

If you have individual DataFrames for each company, you can do the following which will return the DataFrame structure that is required:

```python
from financetoolkit.base import helpers

balance_sheets = helpers.combine_dataframes(
    {
        "TSLA": tsla_balance,
        "GOOGL": googl_balance,
    },
)
income_statements = helpers.combine_dataframes(
    {
        "TSLA": tsla_income,
        "GOOGL": googl_income,
    },
)
cash_flow_statements = helpers.combine_dataframes(
    {
        "TSLA": tsla_cash,
        "GOOGL": googl_cash
    },
)
```

Once all of this is set-up you can feed this information to the Toolkit and use the Toolkit as normally.

```python

# Initialize the Toolkit
companies = Toolkit(
    tickers=["TSLA", "GOOGL"],
    balance=balance_sheets,
    income=income_statements,
    cash=cash_flow_statements,
    format_location="examples/external_datasets",
    reverse_dates=True, # Important when the dates are descending
) 

# Return all Ratios
companies.ratios.collect_all_ratios()
```

This will return all financial ratios that can be collected based on the provided data and the format. See [this notebook](https://www.jeroenbouma.com/projects/financetoolkit/external-datasets) to understand how to work with actual datasets.

![Output of Result](https://github.com/JerBouma/FinanceToolkit/assets/46355364/4ce78cb1-1d37-4f71-ac53-4186cb8b1abf)

# Contact
If you have any questions about the FinanceToolkit or would like to share with me what you have been working on, feel free to reach out to me via:

- **Website**: https://jeroenbouma.com/
- **LinkedIn:** https://www.linkedin.com/in/boumajeroen/
- **Email:** jer.bouma@gmail.com
- **Discord:** add me on Discord **`JerBouma`**

If you'd like to support my efforts, either help me out by contributing to the package or [Buy me a Coffee](https://www.buymeacoffee.com/jerbouma).

[![Star History Chart](https://api.star-history.com/svg?repos=JerBouma/FinanceToolkit&type=Date)](https://star-history.com/#JerBouma/FinanceToolkit&Date)
