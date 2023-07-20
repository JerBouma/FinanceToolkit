![FinanceToolkitCropped](https://github.com/JerBouma/FinanceToolkit/assets/46355364/198d47bd-e1b3-492d-acc4-5d9f02d1d009)

[![BuyMeACoffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Donate-brightgreen?logo=buymeacoffee)](https://www.buymeacoffee.com/jerbouma)
[![Website](https://img.shields.io/badge/Website-Learn%20more%20about%20me%20here-brightgreen?logo=ReadMe)](https://jeroenbouma.com/)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/financetoolkit)](https://pypi.org/project/FinanceToolkit/)
[![PYPI Version](https://img.shields.io/pypi/v/FinanceToolkit)](https://pypi.org/project/FinanceToolkit/)
[![PYPI Downloads](https://img.shields.io/pypi/dm/FinanceToolkit)](https://pypi.org/project/FinanceToolkit/)

While browsing a variety of websites, I kept finding that the same financial metric can greatly vary per source and so do the financial statements reported while little information is given how the metric was calculated.

For example, Microsoft's Price-to-Earnings (PE) ratio on the 6th of May, 2023 is reported to be 28.93 (Stockopedia), 32.05 (Morningstar), 32.66 (Macrotrends), 33.09 (Finance Charts), 33.66 (Y Charts), 33.67 (Wall Street Journal), 33.80 (Yahoo Finance) and 34.4 (Companies Market Cap). All of these calculations are correct, however the method applied varies leading to different results. Therefore, collecting data from multiple sources can lead to wrong interpretation of the results given that one source could be applying a different calculation method than another. And that is, if it is even freely available. Often the calculation is hidden behind a paid subscription.

**This is why I designed the FinanceToolkit**, this is an open-source toolkit in which all relevant financial ratios (50+), indicators and performance measurements are written down in the most simplistic way allowing for complete transparency of the calculation method. This allows you to not have to rely on metrics from other providers and, given a financial statement, allow for efficient manual calculations. This leads to one uniform method of calculation being applied that is available and understood by everyone.

The Finance Toolkit is complimented very well with the [Finance Database 🌎](https://github.com/JerBouma/FinanceDatabase), a database that features 300.000+ symbols containing Equities, ETFs, Funds, Indices, Currencies, Cryptocurrencies and Money Markets. By utilising both, it is possible to do a fully-fledged competitive analysis with the tickers found from the FinanceDatabase inputted into the FinanceToolkit.

<p align="center">
    <img src="examples/Finance Toolkit - 5. Video Demo.gif" alt="Finance Toolkit Illustration" width="100%" onerror='this.style.display = "none"'/>
</p>

---

# Table of Contents

1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
    1. [Using the Finance Toolkit](#using-the-finance-toolkit)
    2. [Working with other Datasets](#working-with-other-datasets)
    3. [Calling Functions Directly](#calling-functions-directly)
3. [Contact](#contact)

# Installation

To install the FinanceToolkit it simply requires the following:

```
pip install financetoolkit
````

Then within Python use:

```python
from financetoolkit import Toolkit
```
To be able to get started, you need to obtain an API Key from FinancialModelingPrep. Use the following instructions to obtain a _free_ API Key. Note that these keys are limited to 250 requests per day but the premium plans are kept at a low cost in case you do run out of the limit or have a need for access to more data. **It is possible to use your own set of financial statements and not rely on FinancialModelingPrep, please have a look [here](#working-with-other-datasets).**

1. Go to [FinancialModellingPrep's API](https://financialmodelingprep.com/developer/docs/)
2. Under "Get your Free API Key Today!" click on "Get my API KEY here"
3. Sign-up to the website and select the Free Plan
4. Obtain the API Key as found [here](https://financialmodelingprep.com/developer/docs/)
5. Start using this package.

Note that I am in no way affiliated FinancialModelingprep and never will be. I have chosen their source as I find it to be the most transparent and reliable. When you notice that data is inaccurate or have any other issue related to the data, note that I simply provide the means to access this data and I am not responsible for the accuracy of the data itself. For this, use [their contact form](https://site.financialmodelingprep.com/contact) or provide the data yourself. 

# Basic Usage

This section explains in detail how the Finance Toolkit can utilitised effectively. Also see the Jupyter Notebook in which you can run the examples also demonstrated here.

___ 

<b><div align="center">Find a variety of How-To Guides for the FinanceToolkit <a href="https://github.com/JerBouma/FinanceToolkit/tree/main/examples">here</a>.</div></b>
___

Within this package the following things are included:

- Company profiles (`profile`), including country, sector, ISIN and general characteristics (from FinancialModelingPrep)
- Company quotes (`quote`), including 52 week highs and lows, volume metrics and current shares outstanding (from FinancialModelingPrep)
- Market cap and enterprise values (`enterprise`), including every intermediate step (from FinancialModelingPrep)
- Company ratings (`rating`), based on key indicators like PE and DE ratios (from FinancialModelingPrep)
- Historical market data (`historical_data`), which can be retrieved on a daily, weekly, monthly and yearly basis (from Yahoo Finance)
- Balance Sheet Statements (`balance_sheet_statement`), Income Statements (`income_statement`) and Cash Flow Statements (`cash_flow_statement`), obtainable from FinancialModelingPrep or the source of your choosing through custom input. These functions are accompanied with a normalization function so that for any source, the same ratio analysis can be performed. Please see this Jupyter Notebook that explains how to use a custom source.
- Efficiency ratios (`efficiency_ratios`), liquidity ratios (`liquidity_ratios`), profitability ratios (`profitability_ratios`), solvency ratios (`solvency_ratios`) and valuation ratios (`valuation_ratios`) functionality that automatically calculates the most important ratios based on the inputted balance sheet, income and cash flow statements.
- Models (`models`) like DUPONT analysis that can be used to perform in-depth financial analysis through a single function.

The dependencies of the package are on purpose *very slim* so that it will work well with any combination of packages and not result in conflicts. I've also been careful with my selection in which I leave out functionality like technical analysis in which [ta-lib](https://ta-lib.org/) does an excellent job as well as portfolio attribution and optimisation in which [Riskfolio-lib](https://riskfolio-lib.readthedocs.io/en/latest/index.html) shines and lastly allow for any source to be incorporated to allow for the incorporation of data sources from [OpenBB](https://openbb.co/).

## Using the Finance Toolkit

A basic example of how to initialise the Finance Toolkit is shown below, also see [this notebook](https://github.com/JerBouma/FinanceToolkit/blob/main/examples/Finance%20Toolkit%20-%201.%20Getting%20Started.ipynb) for a detailed Getting Started guide as well as [this notebook](https://github.com/JerBouma/FinanceToolkit/blob/main/examples/Finance%20Toolkit%20-%202.%20Combining%20the%20Finance%20Toolkit%20with%20the%20Finance%20Database.ipynb) that includes the [Finance Database 🌎](https://github.com/JerBouma/FinanceDatabase) and a proper financial analysis.

````python
from financetoolkit import Toolkit

companies = Toolkit(['AAPL', 'MSFT'], api_key="FMP_KEY")

# an Enterprise example
enterprise = companies.get_enterprise()

# a Historical example
historical_data = companies.get_historical_data(start='2000-01-01', end='2020-01-01')

# a Financial Statement example
balance_sheet_statement = companies.get_balance_sheet_statement()

# a Ratios example
profitability_ratios = companies.ratios.collect_profitability_ratios()

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

## Working with other Datasets

The Finance Toolkit shines in its ability to leverage custom datasets from any data provider as well. This makes it possible to work with your preferred data and not be limited to the data source the Finance Toolkit currently provides. A detailed example can be found [here](https://github.com/JerBouma/FinanceToolkit/blob/main/examples/Finance%20Toolkit%20-%203.%20Using%20External%20Datasets%20(OpenBB).ipynb) but to get started see the code below.

```python
from financetoolkit import Toolkit

# Initialize the Finance Toolkit
companies = Toolkit(['AAPL', 'MSFT'])

# Copy the normalization files
companies.get_normalization_files()
```
This copies over three files, `balance.csv`, `income.csv` and `cash.csv` which will contain a structure like the following:

![Normalization Format](https://github.com/JerBouma/FinanceToolkit/assets/46355364/ea4ebf87-1a27-4c40-a1fb-40d0eb0634bc)

By replacing the first column with the names from your dataset (e.g. replace `cashAndCashEquivalents` with `Cash` if this is how it is called in your dataset), it will automatically normalize the dataset when you initialize the Finance Toolkit. Note that the DataFrame needs to be a multi-index in case you use multiple tickers structured as `Ticker x Financial Statement Item x Periods`.

As an example:

![Dataset Example](https://github.com/JerBouma/FinanceToolkit/assets/46355364/fe0e3db0-3e88-41d2-a355-5f68110fdcf3)

If you have individual DataFrames for each company, you can do the following which will return the DataFrame structure that is required:

```python
from financetoolkit.base import helpers

balance_grouped = helpers.combine_dataframes({'AAPL': balance_apple, 'MSFT': balance_msft})
```

Once all of this is set-up you can feed this information to the Toolkit and use the Toolkit as normally.

```python

# Initialize the Toolkit
companies = Toolkit(
    tickers=['AAPL', 'MSFT'],
    balance=balance_grouped,
    income=income_grouped,
    cash=cash_grouped,
    format_location="FOLDER_PATH",
    reverse_dates=False) # Put this to True in case dates are descending

# Return all Ratios
companies.ratios.collect_all_ratios()
```

This will return all financial ratios that can be collected based on the provided data and the format.

![Output of Result](https://github.com/JerBouma/FinanceToolkit/assets/46355364/2a995430-b8d8-4236-892c-edb47042d6af)

## Calling Functions Directly

It also possible to call any ratio or model directly as shown below. This allows access to 50+ ratios with custom data. Also see [this notebook](https://github.com/JerBouma/FinanceToolkit/blob/main/examples/Finance%20Toolkit%20-%204.%20Calling%20Functions%20Directly.ipynb).

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

# Contact
If you have any questions about the FinanceToolkit or would like to share with me what you have been working on, feel free to reach out to me via:

- **LinkedIn:** https://www.linkedin.com/in/boumajeroen/
-  **Email:** jer.bouma@gmail.com

If you'd like to support my efforts, either help me out by contributing to the package or [Buy me a Coffee](https://www.buymeacoffee.com/jerbouma).

[![Star History Chart](https://api.star-history.com/svg?repos=JerBouma/FinanceToolkit&type=Date)](https://star-history.com/#JerBouma/FinanceToolkit&Date)
