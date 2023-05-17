![FinancialToolkit](https://github.com/JerBouma/FinancialToolkitAlpha/assets/46355364/a0f5e7af-d8d6-4ce8-9e71-47376baa35f2)

[![BuyMeACoffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Donate-brightgreen?logo=buymeacoffee)](https://www.buymeacoffee.com/jerbouma)
[![Issues](https://img.shields.io/github/issues/jerbouma/fundamentalanalysis)](https://github.com/JerBouma/FundamentalAnalysis/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/JerBouma/fundamentalanalysis?color=yellow)](https://github.com/JerBouma/FundamentalAnalysis/pulls)
[![PYPI Version](https://img.shields.io/pypi/v/fundamentalanalysis)](https://pypi.org/project/FundamentalAnalysis/)
[![PYPI Downloads](https://img.shields.io/pypi/dm/fundamentalanalysis)](https://pypi.org/project/FundamentalAnalysis/)

While browsing a variety of websites, I kept finding that the same financial metric can greatly vary per source and so do the financial statements reported while little information is given how the metric was calculated.

For example, Microsoft's Price-to-Earnings (PE) ratio on the 6th of May, 2023 is reported to be 28.93 (Stockopedia), 32.05 (Morningstar), 32.66 (Macrotrends), 33.09 (Finance Charts), 33.66 (Y Charts), 33.67 (Wall Street Journal), 33.80 (Yahoo Finance) and 34.4 (Companies Market Cap). All of these calculations are correct, however the method applied varies leading to different results. Therefore, collecting data from multiple sources can lead to wrong interpretation of the results given that one source could be applying a different calculation method than another. And that is, if it is even freely available. Often the calculation is hidden behind a paid subscription.

**This is why I designed the FinancialToolkit**, this is an open-source financial toolkit in which all relevant financial ratios (50+), indicators and performance measurements are written down in the most simplistic way allowing for complete transparency of the calculation method. This allows you to not have to rely on metrics from other providers and, given a financial statement, allow for efficient manual calculations. This leads to one uniform method of calculation being applied that is available and understood by everyone.

The Financial Toolkit is complimented very well with the [Finance Database 🌎](https://github.com/JerBouma/FinanceDatabase), a database that features 300.000+ symbols containing Equities, ETFs, Funds, Indices, Currencies, Cryptocurrencies and Money Markets. By utilising both, it is possible to do a fully-fledged competitive analysis with the tickers found from the FinanceDatabase inputted into the FinancialToolkit.

---

# Table of Contents

1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
    1. [Quick Start](#quick-start)
    3. [Collecting information from the database](#collecting-information-from-the-database)
    3. [Searching the database extensively](#searching-the-database-extensively)
    4. [Storing the database at a different location](#storing-the-database-at-a-different-location)
3. [Examples](#examples)
    1. [Companies in the Netherlands](#companies-in-the-netherlands)
    2. [Technical Analysis of Biotech ETFs](#technical-analysis-of-biotech-etfs)
    3. [Silicon Valley's Market Cap](#silicon-valleys-market-cap)
4. [Questions & Answers](#questions--answers)
6. [Contact](#contact)

## Installation

To install the FinancialToolkit it simply requires the following:

```cli
pip install financialtoolkit
````
 
The dependencies of the package are on purpose *very slim* so that it will work well with any combination of packages and not result in conflicts. I've also been careful with my selection in which I leave out functionality like technical analysis in which [ta-lib](https://ta-lib.org/) does an excellent job as well as portfolio attribution and optimisation in which [Riskfolio-lib](https://riskfolio-lib.readthedocs.io/en/latest/index.html) shines and lastly allow for any source to be incorporated to allow for the incorporation of data sources from [OpenBB](https://openbb.co/).

To be able to get started, you need to obtain an API Key from FinancialModelingPrep. Use the following instructions to obtain a _free_ API Key. Note that these keys are limited to 250 requests per day but the premium plans are kept at a low cost in case you do run out of the limit or have a need for access to more data. **It is possible to use your own set of financial statements and not rely on FinancialModelingPrep, please have a look LINK_TO_PARAGRAPH.**

1. Go to [FinancialModellingPrep's API](https://financialmodelingprep.com/developer/docs/)
2. Under "Get your Free API Key Today!" click on "Get my API KEY here"
3. Sign-up to the website and select the Free Plan
4. Obtain the API Key as found [here](https://financialmodelingprep.com/developer/docs/)
5. Start using this package.

Note that I am in no way affiliated FinancialModelingprep and never will be. I have chosen their source as I find it to be the most transparent and reliable. When you notice that data is inaccurate or have any other issue related to the data, note that I simply provide the means to access this data and I am not responsible for the accuracy of the data itself. For this, use [their contact form](https://site.financialmodelingprep.com/contact) or provide the data yourself. 

## Basic Usage

This section explains in detail how the Financial Toolkit can utilitised effectively. Also see the Jupyter Notebook in which you can run the examples also demonstrated here. You can find this document [here](https://github.com/JerBouma/FinanceDatabase/blob/main/examples.ipynb).


````python
from financialtoolkit import Toolkit

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
profitability_ratios.loc['AAPL]
````

This returns the following output for `profitability_ratios.loc['AAPL]`. Ommitting `.loc['AAPL']` will return the result for both AAPL and MSFT.


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

It also possible to call any ratio or model directly as shown below.

```python
import pandas as pd
import numpy as np

from financialtoolkit.models import dupont

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

## Examples

Find more information about the Jupyter Notebooks that includes examples below.

### The Analyzer

After installing the FinancialToolkit the Analyzer class can be used as follows:

```

```






Within the related Jupyter Notebook, the class as found in the analyzer module (`from financialtoolkit.analyzer import Analyzer`) is used to rapidly collect financial data and calculate the relevant financial metrics. Within this class it is possible to find:

- Company profiles (`profile`), including country, sector, ISIN and general characteristics (from FinancialModelingPrep)
- Company quotes (`quote`), including 52 week highs and lows, volume metrics and current shares outstanding (from FinancialModelingPrep)
- Market cap and enterprise values (`enterprise`), including every intermediate step (from FinancialModelingPrep)
- Company ratings (`rating`), based on key indicators like PE and DE ratios (from FinancialModelingPrep)
- Historical market data (`historical_data`), which can be retrieved on a daily, weekly, monthly and yearly basis (from Yahoo Finance)
- Balance Sheet Statements (`balance_sheet_statement`), Income Statements (`income_statement`) and Cash Flow Statements (`cash_flow_statement`), obtainable from FinancialModelingPrep or the source of your choosing through custom input. These functions are accompanied with a normalization function so that for any source, the same ratio analysis can be performed. Please see the [normalization](/normalization/) folder for the files if using a custom source.
- Efficiency ratios (`efficiency_ratios`), liquidity ratios (`liquidity_ratios`), profitability ratios (`profitability_ratios`), solvency ratios (`solvency_ratios`) and valuation ratios (`valuation_ratios`) functionality that automatically calculates the most important ratios based on the inputted balance sheet, income and cash flow statements.

___ 

<b><div align="center">Find code examples for the Analyzer functionality of the FinancialToolkit <a href="https://github.com/JerBouma/FinanceDatabase/blob/main/examples.ipynb">here</a>.</div></b>
___

### The Toolkit

After installing the FinancialToolkit access the related functionality with the following:

```
import numpy as np 
import pandas as pd

from financialtoolkit.ratios import efficiency, liquidity, profitability, solvency, valuation
from financialtoolkit.portfolio import portfolio
from financialtoolkit.historical import historical
from financialtoolkit.models import dupont

# Note: this requires your own data to be inputted, this example has dummy data.

# a Portfolio example
tracking_error = portfolio.tracking_error(
    portfolio_returns=np.array([0.05, 0.10, 0.01]),
    benchmark_returns=np.array([0.04, 0.09, 0.02])
) 

# a Historical example
sharpe_ratio = historical.sharpe_ratio(
    returns=np.array([0.10, 0.5, 0.075]),
    risk_free_rate=0.01
)

# a Model example
years = [2018, 2019, 2020, 2021, 2022]
dupont_analysis = dupont.get_dupont_analysis(
    net_income=pd.Series([59531000000, 55256000000, 57411000000, 94680000000, 99803000000], index=years),
    total_revenue=pd.Series([265595000000, 260174000000, 274515000000, 365817000000, 394328000000], index=years),
    total_assets=pd.Series([365725000000, 338516000000, 323888000000, 351002000000, 352755000000], index=years),
    total_equity=pd.Series([107147000000, 90488000000, 65339000000, 63090000000, 50672000000], index=years)
)

# Show the Dupont Analysis
dupont_analysis
```

This returns the following table which closely resembles the proper Dupont analysis for Apple at their given reporting dates in October:

|                    |     2018 |     2019 |     2020 |     2021 |     2022 |
|:-------------------|---------:|---------:|---------:|---------:|---------:|
| Net Profit Margin  | 0.224142 | 0.212381 | 0.209136 | 0.258818 | 0.253096 |
| Asset Turnover     | 0.726215 | 0.768572 | 0.847562 | 1.04221  | 1.11785  |
| Financial Leverage | 3.4133   | 3.741    | 4.95704  | 5.56351  | 6.96154  |
| Return on Equity   | 0.555601 | 0.610645 | 0.878664 | 1.50071  | 1.96959  |

With the related Jupyter Notebook, the vast collection of ratios, indicators and performance measurements are explored. This demonstrates the value of being able to call a function that does the calculation for you instead of needing to manually perform each step of the calculation. This includes:

- Efficiency ratios (`from financialtoolkit.ratios import efficiency`), which includes asset turnover ratio, days of sales outstanding, operating and cash conversion cycles and more.
- Liquidity ratios (`from financialtoolkit.ratios import liquidity`), which includes current ratio, quick ratio, working capital ratio, short term coverage ratio and more.
- Profitability ratios (`from financialtoolkit.ratios import profitability`), which include net profit margin, effective tax rate, return on assets (ROA), return on equity (ROE), return on capital employed (ROCE) and more.
- Solvency ratios (`from financialtoolkit.ratios import solvency`), which includes debt to assets ratio (debt ratio), interest coverage ratio, financial leverage, free cash flow yield and more.
- Valuation ratios (`from financialtoolkit.ratios import valuation`), which includes earnings per share (EPS), price to book ratio (PB), dividend yield, earnings yield, payout ratio and more.
- Variety of models (`from financialtoolkit.models import dupont`), which aggregate ratios and similar statistics to perform well-known methods and models like the Dupont analysis.
- Historical market data (`from financialtoolkit.historical import historical`), which includes the collection of historical data, returns and volatility calculations, sharpe ratio, sortino ratio and more.
- Portfolio performance metrics (`from financialtoolkit.portfolio import portfolio`), which include tracking error, profit factor, jensens alpha and more.

___

<b><div align="center">Find code examples for the Toolkit functionality of the FinancialToolkit <a href="https://github.com/JerBouma/FinanceDatabase/blob/main/examples.ipynb">here</a>.</div></b>
___


# Contact
If you have any questions about the FinancialToolkit or would like to share with me what you have been working on, feel free to reach out to me via:

- **LinkedIn:** https://www.linkedin.com/in/boumajeroen/
-  **Email:** jer.bouma@gmail.com

f you'd like to support my efforts, either help me out by contributing to the package or [Buy me a Coffee](https://www.buymeacoffee.com/jerbouma).

[![Star History Chart](https://api.star-history.com/svg?repos=JerBouma/FinancialToolkit&type=Date)](https://star-history.com/#JerBouma/FinancialToolkit&Date)
