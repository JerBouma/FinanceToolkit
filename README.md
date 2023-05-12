[IMAGE]()

[![BuyMeACoffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Donate-brightgreen?logo=buymeacoffee)](https://www.buymeacoffee.com/jerbouma)
[![Issues](https://img.shields.io/github/issues/jerbouma/fundamentalanalysis)](https://github.com/JerBouma/FundamentalAnalysis/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/JerBouma/fundamentalanalysis?color=yellow)](https://github.com/JerBouma/FundamentalAnalysis/pulls)
[![PYPI Version](https://img.shields.io/pypi/v/fundamentalanalysis)](https://pypi.org/project/FundamentalAnalysis/)
[![PYPI Downloads](https://img.shields.io/pypi/dm/fundamentalanalysis)](https://pypi.org/project/FundamentalAnalysis/)

While browsing a variety of websites, I kept finding that the same financial metric can greatly vary per source and so do the financial statements reported while little information is given how the metric was calculated.

For example, Microsoft's Price-to-Earnings (PE) ratio on the 6th of May, 2023 is reported to be 28.93 (Stockopedia), 32.05 (Morningstar), 32.66 (Macrotrends), 33.09 (Finance Charts), 33.66 (Y Charts), 33.67 (Wall Street Journal), 33.80 (Yahoo Finance) and 34.4 (Companies Market Cap). All of these calculations are correct, however the method applied varies leading to different results. Therefore, collecting data from multiple sources can lead to wrong interpretation of the results given that one source could be applying a different calculation method than another.

**This is why I designed the FinancialToolkit**, this is an open-source financial toolkit in which all relevant financial ratios, indicators and performance measurements are written down in the most simplistic way allowing for complete transparency of the calculation method. This allows you to not have to rely on metrics from other providers and, given a financial statement, allow for efficient manual calculations. This leads to one uniform method of calculation being applied that is available and understood by everyone.

This package has two paths that can be explored further:

## [The Analyzer 🔍](#the-analyzer)

Making use of the `Analyzer` class that allows for any amount of companies to be analyzed and has the option to collect data from FinancialModelingPrep or by using custom financial statements which are then normalized to be able to calculate efficiency, liquidity, profitability, solvency and valuation metrics in an instant. With the custom input, this package offers support for virtually any platform including Bloomberg, Eikon Refinitiv, OpenBB or any data provider like Polygon, YCharts and AlphaVantage. 

**This path is recommended for those looking to find a tool that collects data and provides the calculations for further analysis while benefiting from the open source and transparent nature of the package.**


## [The Toolkit 🛠️](#the-toolkit)

Making use of the extensive collection of ratios, indicators and performance measurements (50+) by directly calling functions like `price_to_earnings_ratio`, `extended_dupont_analysis`, `debt_to_assets_ratio`, `sortino_ratio` and `tracking_error` which tend to accept any input, let it be a float, an array or DataFrame. This allows the package to truly be used as a toolkit, as you are in need of certain models and metric calculations, you call the function directly and input the relevant variables.

**This path is recommended for those looking to delve deeper into the calculations as part of their own models or research being able to rely on validated methods allowing for efficiency.**
___

The Financial Toolkit is complimented very well with the [Finance Database 🌎](https://github.com/JerBouma/FinanceDatabase), a database that features 300.000+ symbols containing Equities, ETFs, Funds, Indices, Currencies, Cryptocurrencies and Money Markets. By utilising both, it is possible to do a fully-fledged competitive analysis with the tickers found from the FinanceDatabase inputted into the FinancialToolkit.

---

## Installation

To install the FinancialToolkit it simply requires the following:

 ```cli
 pip install financialtoolkit
 ````
 
 The dependencies of the package are on purpose *very slim* so that it will work well with any combination of packages and not result in conflicts.

Note the following:

- **Technical indicators are on purpose not included within this package as I find that [ta-lib](https://ta-lib.org/) does an excellent job in providing these.** I recommend using their package alongside the FinancialToolkit.
- **I've not included any portfolio optimization techniques as I believe that [Riskfolio-lib](https://riskfolio-lib.readthedocs.io/en/latest/index.html) should be used for this purpose.** The author has spend countless of hours on desiging this package and is therefore a true domain export in this area. The FinancialToolkit does however compliment well with this package through providing key metrics to perform portfolio attribution.
- **I've used merely one source as a data provider because I believe that platforms like [OpenBB](https://openbb.co/) do an excellent job at data aggregation.** With the 'Custom' option, you are able to combine OpenBB with the FinancialToolkit to collect data and perform the necessary calculations efficiently.
- **I am in no way affiliated FinancialModelingprep and never will be.** I have chosen their source as I find it to be the most transparent and reliable. When you notice that data is inaccurate or have any other issue related to the data, note that I simply provide the means to access this data and I am not responsible for the accuracy of the data itself. For this, use [their contact form](https://site.financialmodelingprep.com/contact) or provide the data yourself.

**To be able to use the functionality of FinancialModelingPrep you need an API Key from their website.** Use the following instructions to obtain a _free_ API Key. Note that these keys are limited to 250 requests per day but the premium plans are kept at a low cost in case you do run out of the limit or have a need for access to more data.

1. Go to [FinancialModellingPrep's API](https://financialmodelingprep.com/developer/docs/)
2. Under "Get your Free API Key Today!" click on "Get my API KEY here"
3. Sign-up to the website and select the Free Plan
4. Obtain the API Key as found [here](https://financialmodelingprep.com/developer/docs/)
5. Start using this package.

## Examples

Find more information about the Jupyter Notebooks that includes examples below.

### The Analyzer

After installing the FinancialToolkit the Analyzer class can be used as follows:

```
from financialtoolkit.analyzer import Analyzer

companies = Analyzer(['AAPL', 'MSFT'], api_key="FMP_KEY")

# an Enterprise example
enterprise = companies.get_enterprise()

# a Historical example
historical_data = companies.get_historical_data(start='2000-01-01', end='2020-01-01')

# a Financial Statement example
balance_sheet_statement = companies.get_balance_sheet_statement()

# a Ratios example
profitability_ratios = companies.get_profitability_ratios()

# Show the profitability ratios
profitability_ratios
```

Which returns the following output for the `profitability_ratios` variable:

|                |   Gross Margin |   Operating Margin |   Net Profit Margin |   Pre Tax Profit |   Effective Tax Rate |   Return on Assets (ROA) |   Return on Equity (ROE) |   Return on Invested Capital (ROIC) |   Return on Tangible Assets (ROTA) |   Return on Capital Employed (ROCE) |   Income Quality Ratio |
|:---------------|---------------:|-------------------:|--------------------:|-----------------:|---------------------:|-------------------------:|-------------------------:|------------------------------------:|-----------------------------------:|------------------------------------:|-----------------------:|
| AAPL, 2018 |       0.383437 |           0.26694  |            0.224142 |         0.274489 |             0.183422 |                0.162775  |                 0.555601 |                            0.330474 |                           0.555601 |                            0.305968 |                1.30073 |
| AAPL, 2019 |       0.378178 |           0.24572  |            0.212381 |         0.252666 |             0.159438 |                0.16323   |                 0.610645 |                            0.349435 |                           0.610645 |                            0.297739 |                1.25581 |
| AAPL, 2020 |       0.382332 |           0.241473 |            0.209136 |         0.244398 |             0.144282 |                0.177256  |                 0.878664 |                            0.402149 |                           0.878664 |                            0.320207 |                1.4052  |
| AAPL, 2021 |       0.417794 |           0.297824 |            0.258818 |         0.298529 |             0.133023 |                0.269742  |                 1.50071  |                            0.58116  |                           1.50071  |                            0.495972 |                1.09884 |
| AAPL, 2022 |       0.433096 |           0.302887 |            0.253096 |         0.30204  |             0.162045 |                0.282924  |                 1.96959  |                            0.67145  |                           1.96959  |                            0.613937 |                1.22392 |
| MSFT, 2018 |       0.652474 |           0.317669 |            0.150154 |         0.3305   |             0.545676 |                0.0640183 |                 0.200331 |                            0.177905 |                           0.221938 |                            0.195683 |                2.64824 |
| MSFT, 2019 |       0.65902  |           0.34137  |            0.311817 |         0.347163 |             0.101813 |                0.136937  |                 0.383465 |                            0.293593 |                           0.414887 |                            0.213571 |                1.32989 |
| MSFT, 2020 |       0.67781  |           0.370304 |            0.309625 |         0.370842 |             0.165077 |                0.146961  |                 0.374298 |                            0.313879 |                           0.397974 |                            0.242912 |                1.37023 |
| MSFT, 2021 |       0.689258 |           0.415949 |            0.364517 |         0.423005 |             0.138266 |                0.183568  |                 0.431522 |                            0.370857 |                           0.456606 |                            0.299639 |                1.25247 |
| MSFT, 2022 |       0.684017 |           0.420553 |            0.366863 |         0.422232 |             0.131134 |                0.19937   |                 0.436755 |                            0.398895 |                           0.46854  |                            0.317985 |                1.22405 |


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
