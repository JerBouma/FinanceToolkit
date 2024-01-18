[![FinanceToolkit](https://github.com/JerBouma/FinanceToolkit/assets/46355364/198d47bd-e1b3-492d-acc4-5d9f02d1d009)](https://github.com/JerBouma/FinanceToolkit)

[![GitHub Sponsors](https://img.shields.io/badge/Sponsor_this_Project-grey?logo=github)](https://github.com/sponsors/JerBouma)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy_Me_a_Coffee-grey?logo=buymeacoffee)](https://www.buymeacoffee.com/jerbouma)
[![Twitter](https://img.shields.io/badge/Twitter-grey?logo=x)](https://twitter.com/JerBouma)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-grey?logo=Linkedin&logoColor=white)](https://www.linkedin.com/in/boumajeroen/)
[![Documentation](https://img.shields.io/badge/Documentation-grey?logo=readme)](https://www.jeroenbouma.com/projects/financetoolkit/docs)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/financetoolkit)](https://pypi.org/project/FinanceToolkit/)
[![PYPI Version](https://img.shields.io/pypi/v/FinanceToolkit)](https://pypi.org/project/FinanceToolkit/)
[![PYPI Downloads](https://static.pepy.tech/badge/financetoolkit/month)](https://pepy.tech/project/financetoolkit)

While browsing a variety of websites, I kept finding that the same financial metric can greatly vary per source and so do the financial statements reported while little information is given how the metric was calculated.

For example, Microsoft's Price-to-Earnings (PE) ratio on the 6th of May, 2023 is reported to be 28.93 (Stockopedia), 32.05 (Morningstar), 32.66 (Macrotrends), 33.09 (Finance Charts), 33.66 (Y Charts), 33.67 (Wall Street Journal), 33.80 (Yahoo Finance) and 34.4 (Companies Market Cap). All of these calculations are correct, however the method applied varies leading to different results. Therefore, collecting data from multiple sources can lead to wrong interpretation of the results given that one source could be applying a different calculation method than another. And that is, if it is even freely available. Often the calculation is hidden behind a paid subscription.

**This is why I designed the FinanceToolkit**, this is an open-source toolkit in which all relevant financial ratios ([150+](#available-metrics)), indicators and performance measurements are written down in the most simplistic way allowing for complete transparency of the calculation method ([proof](https://github.com/JerBouma/FinanceToolkit/blob/main/financetoolkit/ratios/valuation_model.py)). This allows you to not have to rely on metrics from other providers and, given a financial statement, allow for efficient manual calculations. This leads to one uniform method of calculation being applied that is available and understood by everyone.

The Finance Toolkit not only supports Equities. Even for Options, Currencies, Cryptocurrencies, ETFs, Mutual Funds, Indices, Money Markets, Commodities, Key Economic Indicators and more, the Finance Toolkit can be used to obtain historical data as well as important performance and risk measurements such as the Sharpe Ratio and Value at Risk.

The Finance Toolkit is complimented very well with the [Finance Database 🌎](https://github.com/JerBouma/FinanceDatabase), a database that features 300.000+ symbols containing Equities, ETFs, Funds, Indices, Currencies, Cryptocurrencies and Money Markets. By utilising both, it is possible to do a fully-fledged competitive analysis with the tickers found from the FinanceDatabase inputted into the FinanceToolkit.

<p align="center">
    <img src="examples/Finance Toolkit - Video Demo.gif" alt="Finance Toolkit Illustration" width="100%" onerror="this.style.display = 'none'"/>
</p>

---

# Table of Contents

1. [Installation](#installation)
2. [Functionality](#functionality)
2. [Basic Usage](#basic-usage)
3. [Available Metrics](#available-metrics)
4. [Questions & Answers](#questions--answers)
5. [Contributing](#contributing)
6. [Contact](#contact)

# Installation

Before installation, consider starring the project on GitHub which helps others find the project as well. 

<a href="https://github.com/JerBouma/FinanceToolkit" target="_blank"><img width="1415" alt="image" src="https://github.com/JerBouma/FinanceToolkit/assets/46355364/014109fe-0c68-47d4-99bd-217c69dcea8d"></a>

To install the FinanceToolkit it simply requires the following:

```
pip install financetoolkit -U
````

Then within Python use:

```python
from financetoolkit import Toolkit

companies = Toolkit(
    tickers=['GOOGL', 'MSFT', 'AMZN'],
    api_key="FINANCIAL_MODELING_PREP_KEY",
)
```
To be able to get started, you need to obtain an API Key from FinancialModelingPrep. This is used to gain access to 30+ years of financial statement both annually and quarterly. Note that the Free plan is limited to 250 requests each day, 5 years of data and only features companies listed on US exchanges.

___ 

<b><div align="center">Obtain an API Key from FinancialModelingPrep <a href="https://www.jeroenbouma.com/fmp" target="_blank">here</a>.</div></b>
___

Through the link you are able to subscribe for the free plan and also premium plans at a **15% discount**. This is an affiliate link and thus supports the project at the same time. I have chosen FinancialModelingPrep as a source as I find it to be the most transparent, reliable and at an affordable price. I have yet to find a platform offering such low prices for the amount of data offered. When you notice that the data is inaccurate or have any other issue related to the data, note that I simply provide the means to access this data and I am not responsible for the accuracy of the data itself. For this, use [their contact form](https://site.financialmodelingprep.com/contact) or provide the data yourself. 

___ 

<b><div align="center">Find a variety of How-To Guides including Code Documentation for the FinanceToolkit <a href="https://www.jeroenbouma.com/projects/financetoolkit" target="_blank">here</a>.</div></b>
___

In the next pargraphs you can find information about the available functionality, basic usage examples, a complete list of all 150+ metrics and how you can contribute and reach out to me.

# Functionality

The Finance Toolkit features the following functionality, also see [Basic Usage](#basic-usage) to see some of these functions in action and the [Code Documentation](https://www.jeroenbouma.com/projects/financetoolkit/docs) for a detailed explanation of each function.

| **Feature** | **Description** |
|:--------|:------------|
| **Discover Instruments** | Obtain lists of companies, cryptocurrencies, forex, commodities, etfs and indices including screeners, quotes, performance metrics and more to find and select tickers to use in the Finance Toolkit. For more information have a look [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/discovery). |
| **Company Overview** | Obtain company profiles including a general description about the company as well as the current quote, the rating and recommendation based on a variety of different metrics such as the Discounted Cash Flow (DCF) and Return on Equity (RoE) |
| **Historical Market Data** | Obtain a full history of OHLC, dividends, (excess) returns, cumulative returns and (excess) volatility calculations on a daily, weekly, quarterly or yearly basis. This includes the normalization of the data as well as auto-adjustments for missing values. It can also be obtained from both FinancialModelingPrep and Yahoo Finance. |
| **Treasury Rates** | Just like the historical market data, obtain a full history for the treasury rates which also serve as risk-free rate by default allowing for calculations such as the Sharpe Ratio. This also includes normalization of the data as well as auto-adjustments for missing values. It can also be obtained from both FinancialModelingPrep and Yahoo Finance. |
| **Dividends Calendar** | Obtain the (adjusted) dividend and the the related record, payment and declaration dates. This represent both past and future dividend cashflows based on official statements from the selected companies. |
| **Analyst Estimates** | Obtain the expected Revenue, Earnings-per-Share (EPS), EBITDA, EBIT, Net Income, and SGA Expense from the past and future from a large collection of analysts. For each metric, it includes the lower bound, upper bound and median based on the analyst consensus. |
| **Earnings Calendar** | Retrieve precise dates for past and future earnings releases, including analyst expectations. These metrics are presented for both Earnings-per-Share (EPS) and Revenue.|
| **Revenue Segmentation** | Retrieve revenue segmentation based on both product design and geographic location for each company. Normalization is applied to ensure accurate comparisons of revenue streams over time among different companies. |
| **ESG Scores** | Retrieve comprehensive Environmental, Social, and Governance (ESG) scores for each company. The scores encompass various factors such as carbon footprint, diversity and inclusion efforts, ethical business practices, and board effectiveness. This information provides a holistic view of a company's sustainability and ethical practices in alignment with ESG criteria.|
| **Financial Statements** | Acquire a full history of both annual and quarterly financial statements, including balance sheets, income statements, and cash flow statements. These financial statements are normalized to facilitate straightforward comparisons between companies such as automatic currency conversions of the entire of the financial statement. Additionally, variations in accounting periods among companies are automatically standardized, ensuring that each company's year or quarter aligns with the corresponding calendar period. |
| **Financial Ratios** | Acquire 50+ financial ratios categorized into their key financial domains, including Efficiency, Liquidity, Profitability, Solvency, and Valuation.<br><br>*Efficiency ratios*, such as Inventory Turnover and Asset Turnover provide insights into operational effectiveness. *Liquidity ratios*, including the Current and Quick Ratios, gauge the company's short-term financial viability. *Profitability ratios*, for example Return on Equity (RoE) and Net Profit Margin, shed light on the company's ability to generate profits. *Solvency ratios*, which include the Debt-to-Equity ratio or Interest Coverage ratio, assess the company's long-term financial stability. *Valuation ratios*, such as the Price-to-Earnings (P/E) ratio and Earnings Yield, offer perspectives on the company's market valuation.<br><br>Find all of them [here](#financial-ratios). |
| **Custom Ratios** | Define custom ratios and calculate these automatically based on the balance sheet, income and cash flow statements. With this, it is possible to calculate any collection of custom ratios without needing to understanding the backend of the Finance Toolkit. Learn how [here](https://www.jeroenbouma.com/projects/financetoolkit/custom-ratios).|
| **Financial Models** | Access a variety of financial models that combine the historical market data and financial statements. This is for example the DUPONT analysis, Weighted Average Cost of Capital (WACC), and Altman Z-Score. Find all of them [here](#financial-models). |
| **Options and Greeks** | Access valuation methods such as the Black Scholes Model as well as real market prices for both call and put options. Next to that, calculate all First, Second and Third-Order Greeks such as Delta, Gamma, Theta, Vega, Rho, Veta and Ultima that explain option sensitivities. Find all of them [here](#options-and-greeks). |
| **Performance Metrics** | Access a variety of performance metrics, including Jensens Alpha, Capital Asset Pricing Model (CAPM), and (Rolling) Sharpe Ratio. These metrics provide a comprehensive evaluation of a company's financial performance, risk-adjusted returns, and its relationship with market benchmarks. Find all of them [here](#performance-metrics). |
| **Risk Metrics** | Access a diverse set of risk metrics, including the Value at Risk (VaR), GARCH, and Entropic Value at Risk (eVaR). These metrics are meant to gauge the potential major losses that can occur over time as well as estimate future volatilities. Find all of them [here](#risk-metrics). |
| **Technical Indicators** | Access an extensive array of more than 40 technical indicators such as the Relative Strength Index (RSI), Exponential Moving Average (EMA), and Bollinger Bands. These technical indicators are key to understanding short-term market trends, momentum, and volatility. Find all of them [here](#technical-indicators). |
| **Key Economic Indicators** | Access an extensive collection of over 50 macro-economic indicators such as Gross Domestic Product (GDP), Consumer Price Index (CPI), and Federal Funds Rate (FED). These indicators serve as barometers to understand the economic, environmental and social climate of 60+ countries. Find all of them [here](#key-economic-indicators). |

The dependencies of the package are on purpose *very slim* so that it will work well with any combination of packages and not result in conflicts.

# Basic Usage

This section is an introduction to the Finance Toolkit. Also see [this notebook](https://www.jeroenbouma.com/projects/financetoolkit/getting-started) for a detailed Getting Started guide as well as [this notebook](https://www.jeroenbouma.com/projects/financetoolkit/finance-database) that includes the [Finance Database 🌎](https://www.jeroenbouma.com/projects/financedatabase) and a proper financial analysis. Next to that, find below a fully-fledged code documentation as well as Jupyter Notebooks in which you can see many examples ranging from basic examples to creating custom ratios to working with your own datasets.

___ 

<b><div align="center">Find a variety of How-To Guides including Code Documentation for the FinanceToolkit <a href="https://www.jeroenbouma.com/projects/financetoolkit">here</a>.</div></b>
___


A basic example of how to use the Finance Toolkit is shown below.


````python
from financetoolkit import Toolkit

companies = Toolkit(["AAPL", "MSFT"], api_key=API_KEY, start_date="2017-12-31")

# a Historical example
historical_data = companies.get_historical_data()

# a Financial Statement example
income_statement = companies.get_income_statement()

# a Ratios example
profitability_ratios = companies.ratios.collect_profitability_ratios()

# a Models example
extended_dupont_analysis = companies.models.get_extended_dupont_analysis()

# a Options example
all_greeks = companies.options.collect_all_greeks(expiration_time_range=180)

# a Performance example
factor_asset_correlations = companies.performance.get_factor_asset_correlations(
    period="quarterly"
)

# a Risk example
value_at_risk = companies.risk.get_value_at_risk(period="weekly")

# a Technical example
ichimoku_cloud = companies.technicals.get_ichimoku_cloud()

# an Economics example
unemployment_rates = companies.economics.get_unemployment_rate()
````

Generally, the functions return a DataFrame with a multi-index in which all tickers, in this case Apple and Microsoft, are presented. To keep things manageable for this README, I select just Apple but in essence the list of tickers can be endless as I've seen DataFrames with thousands of tickers. The filtering is done through `.loc['AAPL']` and `.xs('AAPL', level=1, axis=1)` based on whether it's fundamental data or historical data respectively.

### Obtaining Historical Data

Obtain historical data on a daily, weekly, monthly or yearly basis. This includes OHLC, volumes, dividends, returns, cumulative returns and volatility calculations for each corresponding period. For example, the a portion of the historical data for Apple is shown below.

| date       |    Open |    High |     Low |   Close |   Adj Close |      Volume |   Dividends |   Return |   Volatility |   Excess Return |   Excess Volatility |   Cumulative Return |
|:-----------|--------:|--------:|--------:|--------:|------------:|------------:|------------:|---------:|-------------:|----------------:|--------------------:|--------------------:|
| 2018-01-02 | 42.54   | 43.075  | 42.315  | 43.065  |       40.78 | 1.02224e+08 |           0 |   0      |       0.0202 |         -0.0067 |              0.0233 |              1      |
| 2018-01-03 | 43.1325 | 43.6375 | 42.99   | 43.0575 |       40.77 | 1.17982e+08 |           0 |  -0.0002 |       0.0202 |         -0.0247 |              0.0233 |              0.9998 |
| 2018-01-04 | 43.135  | 43.3675 | 43.02   | 43.2575 |       40.96 | 8.97384e+07 |           0 |   0.0047 |       0.0202 |         -0.0198 |              0.0233 |              1.0044 |
| 2018-01-05 | 43.36   | 43.8425 | 43.2625 | 43.75   |       41.43 | 9.46401e+07 |           0 |   0.0115 |       0.0202 |         -0.0133 |              0.0233 |              1.0159 |
| 2018-01-08 | 43.5875 | 43.9025 | 43.4825 | 43.5875 |       41.27 | 8.22711e+07 |           0 |  -0.0039 |       0.0202 |         -0.0287 |              0.0233 |              1.012  |

And below the cumulative returns are plotted which include the S&P 500 as benchmark:

![HistoricalData](https://github.com/JerBouma/FinanceToolkit/assets/46355364/cd7b5029-0e66-4592-9822-42b652e7deed)


### Obtaining Financial Statements

Obtain an Income Statement on an annual or quarterly basis. This can also be an income statement (`companies.get_income_statement()`) or cash flow statement (`companies.get_cash_flow_statement()`). For example, the first 5 rows of the Income Statement for Apple are shown below.

|                                   |        2017 |        2018 |        2019 |        2020 |        2021 |        2022 |        2023 |
|:----------------------------------|------------:|------------:|------------:|------------:|------------:|------------:|------------:|
| Revenue                           | 2.29234e+11 | 2.65595e+11 | 2.60174e+11 | 2.74515e+11 | 3.65817e+11 | 3.94328e+11 | 3.83285e+11 |
| Cost of Goods Sold                | 1.41048e+11 | 1.63756e+11 | 1.61782e+11 | 1.69559e+11 | 2.12981e+11 | 2.23546e+11 | 2.14137e+11 |
| Gross Profit                      | 8.8186e+10  | 1.01839e+11 | 9.8392e+10  | 1.04956e+11 | 1.52836e+11 | 1.70782e+11 | 1.69148e+11 |
| Gross Profit Ratio                | 0.3847      | 0.3834      | 0.3782      | 0.3823      | 0.4178      | 0.4331      | 0.4413      |
| Research and Development Expenses | 1.1581e+10  | 1.4236e+10  | 1.6217e+10  | 1.8752e+10  | 2.1914e+10  | 2.6251e+10  | 2.9915e+10  |

And below the Earnings Before Interest, Taxes, Depreciation and Amortization (EBITDA) are plotted for both Apple and Microsoft.

![FinancialStatements](https://github.com/JerBouma/FinanceToolkit/assets/46355364/a4ba0629-0832-4dc0-a5c1-9cf2c9bd13ce)

### Obtaining Financial Ratios

Get Profitability Ratios based on the inputted balance sheet, income and cash flow statements. This can be any of the 50+ ratios within the `ratios` module. The `get_` functions show a single ratio whereas the `collect_` functions show an aggregation of multiple ratios. For example, see some of the profitability ratios of Microsoft below.

|                                 |    2017 |    2018 |    2019 |    2020 |    2021 |    2022 |    2023 |
|:--------------------------------|--------:|--------:|--------:|--------:|--------:|--------:|--------:|
| Gross Margin                    |  0.6191 |  0.6525 |  0.659  |  0.6778 |  0.6893 |  0.684  |  0.6892 |
| Operating Margin                |  0.2482 |  0.3177 |  0.3414 |  0.3703 |  0.4159 |  0.4206 |  0.4177 |
| Net Profit Margin               |  0.2357 |  0.1502 |  0.3118 |  0.3096 |  0.3645 |  0.3669 |  0.3415 |
| Interest Coverage Ratio         | 13.9982 | 16.5821 | 20.3429 | 25.3782 | 34.7835 | 47.4275 | 52.0244 |
| Income Before Tax Profit Margin |  0.2574 |  0.3305 |  0.3472 |  0.3708 |  0.423  |  0.4222 |  0.4214 |

And below a few of the profitability ratios are plotted for Microsoft.

![FinancialRatios](https://github.com/JerBouma/FinanceToolkit/assets/46355364/93221f7a-face-4035-87c7-e43815e89eb4)

### Obtaining Financial Models

Get an Extended DuPont Analysis based on the inputted balance sheet, income and cash flow statements. This can also be an Enterprise Value Breakdown, Weighted Average Cost of Capital (WACC), Altman Z-Score and many more models. For example, this shows the Extended DuPont Analysis for Apple:

|                         |     2017 |   2018 |   2019 |   2020 |   2021 |   2022 |   2023 |
|:------------------------|---------:|-------:|-------:|-------:|-------:|-------:|-------:|
| Interest Burden Ratio   |   0.9572 | 0.9725 | 0.9725 | 0.988  | 0.9976 | 1.0028 | 1.005  |
| Tax Burden Ratio        |   0.7882 | 0.8397 | 0.8643 | 0.8661 | 0.869  | 0.8356 | 0.8486 |
| Operating Profit Margin |   0.2796 | 0.2745 | 0.2527 | 0.2444 | 0.2985 | 0.302  | 0.2967 |
| Asset Turnover          | nan      | 0.7168 | 0.7389 | 0.8288 | 1.0841 | 1.1206 | 1.0868 |
| Equity Multiplier       | nan      | 3.0724 | 3.5633 | 4.2509 | 5.255  | 6.1862 | 6.252  |
| Return on Equity        | nan      | 0.4936 | 0.5592 | 0.7369 | 1.4744 | 1.7546 | 1.7195 |

And below each component of the Extended Dupont Analysis is plotted including the resulting Return on Equity (ROE).

![Models](https://github.com/JerBouma/FinanceToolkit/assets/46355364/f5e1cab3-d1bd-455d-a4ba-92e1348163be)

### Obtaining Options and Greeks

Get the Black Scholes Model for both call and put options including the relevant Greeks, in this case Delta, Gamma, Theta and Vega. This can be any of the First, Second or Third Order Greeks as found in the the `options` module. The `get_` functions show a single Greek whereas the `collect_` functions show an aggregation of Greeks. For example, see the delta of the Call options for Apple for multiple expiration times and strike prices below (Stock Price: 185.92, Volatility: 31.59%, Dividend Yield: 0.49% and Risk Free Rate: 3.95%):

|     |   1 Month |   2 Months |   3 Months |   4 Months |   5 Months |   6 Months |
|----:|----------:|-----------:|-----------:|-----------:|-----------:|-----------:|
| 175 |    0.7686 |     0.7178 |     0.6967 |     0.6857 |     0.6794 |     0.6759 |
| 180 |    0.6659 |     0.64   |     0.6318 |     0.629  |     0.6285 |     0.6291 |
| 185 |    0.5522 |     0.5583 |     0.5648 |     0.571  |     0.5767 |     0.5816 |
| 190 |    0.4371 |     0.4762 |     0.4977 |     0.513  |     0.5249 |     0.5342 |
| 195 |    0.3298 |     0.3971 |     0.4324 |     0.4562 |     0.474  |     0.4875 |

Which can also be plotted together with Gamma, Theta and Vega as follows:

![Greeks](https://github.com/JerBouma/FinanceToolkit/assets/46355364/3aebe116-c4ac-4845-9801-54d2b4bde0f5)

### Obtaining Performance Metrics

Get the correlations with the factors as defined by Fama-and-French. These include market, size, value, operating profitability and investment. The beauty of all functionality here is that it can be based on any period as the function accepts the period 'weekly', 'monthly', 'quarterly' and 'yearly'. For example, this shows the quarterly correlations for Apple:

|        |   Mkt-RF |     SMB |     HML |     RMW |     CMA |
|:-------|---------:|--------:|--------:|--------:|--------:|
| 2022Q2 |   0.9177 | -0.1248 | -0.5077 | -0.3202 | -0.2624 |
| 2022Q3 |   0.8092 |  0.1528 | -0.5046 | -0.1997 | -0.5231 |
| 2022Q4 |   0.8998 |  0.2309 | -0.5968 | -0.1868 | -0.5946 |
| 2023Q1 |   0.7737 |  0.1606 | -0.3775 | -0.228  | -0.5707 |
| 2023Q2 |   0.7416 | -0.1166 | -0.2722 |  0.0093 | -0.4745 |

And below the correlations with each factor are plotted over time for both Apple and Microsoft.

![Performance](https://github.com/JerBouma/FinanceToolkit/assets/46355364/9c1eff76-b5c8-4bd2-9f47-8ce70bf002db)

### Obtaining Risk Metrics

Get the Value at Risk for each week. Here, the days within each week are considered for the Value at Risk. This makes it so that you can understand within each period what is the expected Value at Risk (VaR) which can again be any period but also based on distributions such as Historical, Gaussian, Student-t, Cornish-Fisher.

|                       |    AAPL |    MSFT |   Benchmark |
|:----------------------|--------:|--------:|------------:|
| 2023-09-25/2023-10-01 | -0.0205 | -0.0133 |     -0.0122 |
| 2023-10-02/2023-10-08 | -0.0048 | -0.0206 |     -0.0108 |
| 2023-10-09/2023-10-15 | -0.0089 | -0.0092 |     -0.0059 |
| 2023-10-16/2023-10-22 | -0.0135 | -0.0124 |     -0.0131 |
| 2023-10-23/2023-10-29 | -0.0224 | -0.0293 |     -0.0139 |

And below the Value at Risk (VaR) for Apple, Microsoft and the benchmark (S&P 500) are plotted also demonstrating the impact of COVID-19.

![Risk](https://github.com/JerBouma/FinanceToolkit/assets/46355364/a95e5b51-f7fc-4a70-bbb4-bf88b346523e)

### Obtaining Technical Indicators

Get the Ichimoku Cloud parameters based on the historical market data. This can be any of the 30+ technical indicators within the `technicals` module. The `get_` functions show a single indicator whereas the `collect_` functions show an aggregation of multiple indicators. For example, see some of the parameters for Apple below:

| Date       |   Base Line |   Conversion Line |   Leading Span A |   Leading Span B |
|:-----------|------------:|------------------:|-----------------:|-----------------:|
| 2023-10-30 |     174.005 |           171.755 |          176.245 |            178.8 |
| 2023-10-31 |     174.005 |           171.755 |          176.37  |            178.8 |
| 2023-11-01 |     174.005 |           170.545 |          176.775 |            178.8 |
| 2023-11-02 |     174.005 |           171.725 |          176.235 |            178.8 |
| 2023-11-03 |     174.005 |           171.725 |          175.558 |            178.8 |

And below the Ichimoku Cloud parameters are plotted for Apple and Microsoft side-by-side.

![Technicals](https://github.com/JerBouma/FinanceToolkit/assets/46355364/1ced5b34-2410-4206-8ddf-bb053bcb21b2)

### Understanding Key Economic Indicators

Get insights for 60+ countries into key economic indicators such as the Consumer Price Index (CPI), Gross Domestic Product (GDP), Unemployment Rates and 3-month and 10-year Government Interest Rates. This is done through the `economics` module and can be used as a standalone module as well by using `from financetoolkit import Economics`. For example see a selection of the countries below:

|      |   Colombia |   United States |   Sweden |   Japan |   Germany |
|:-----|-----------:|----------------:|---------:|--------:|----------:|
| 2017 |     0.093  |          0.0435 |   0.0686 |  0.0281 |    0.0357 |
| 2018 |     0.0953 |          0.039  |   0.0648 |  0.0244 |    0.0321 |
| 2019 |     0.1037 |          0.0367 |   0.0691 |  0.0235 |    0.0298 |
| 2020 |     0.1586 |          0.0809 |   0.0848 |  0.0278 |    0.0362 |
| 2021 |     0.1381 |          0.0537 |   0.0889 |  0.0282 |    0.0358 |
| 2022 |     0.1122 |          0.0365 |   0.0748 |  0.026  |    0.0307 |

And below these Unemployment Rates are plotted over time:

![Economics](https://github.com/JerBouma/FinanceToolkit/assets/46355364/bb44bd4e-07a1-4ecf-a4eb-7fc09a960930)

# Available Metrics

The Finance Toolkit has the ability to calculate 150+ financial metrics. The following list shows all of the available metrics. Note that the Finance Toolkit is not limited to these ratios and indicators as it is possible to add custom ratios as well. See [this Notebook](https://www.jeroenbouma.com/projects/financetoolkit/custom-ratios) for more information.

Each ratio and indicator has a corresponding function that can be called directly for example `ratios.get_return_on_equity` or `technicals.get_relative_strength_index`.

___ 

<b><div align="center">Find a variety of How-To Guides including Code Documentation for the FinanceToolkit <a href="https://www.jeroenbouma.com/projects/financetoolkit">here</a>.</div></b>
___

## Financial Ratios

The Ratios Module contains over 50+ ratios that can be used to analyse companies. These ratios are divided into 5 categories which are efficiency, liquidity, profitability, solvency and valuation. Each ratio is calculated using the data from the Toolkit module.  **Find the Notebook [here](https://www.jeroenbouma.com/projects/financetoolkit/ratios-module) and the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios) which includes an explanation about the ratio, the parameters and an example.**

All of these ratios can be calculated based on (lagged) growth as well as trailing (e.g. TTM) metrics. This is embedded in all ratios as well as the financial statements themselves which means it is possible to calculate revenue growth and 12-month (TTM) Price-to-Earnings with the parameters `growth=True` and `trailing=4` respectively. Note that trailing is based on periods therefore TTM can only be calculated by setting `quarterly=True` in the Toolkit initialization.

<details>
  <summary><b>Efficiency Ratios ⚙️</b></summary>

The efficiency ratios are used to assess how well a company utilizes its assets and liabilities to generate revenue. They provide insight into the company’s operational efficiency and its ability to manage its assets and liabilities.

> **Asset Turnover Ratio**

The asset turnover ratio is calculated by dividing the company’s net sales (revenue) by its average total assets. It measures how well a company utilizes its assets to generate revenue. A higher asset turnover ratio indicates that the company is generating more revenue per unit of assets, which is generally seen as a positive sign of operational efficiency. See documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_asset_turnover_ratio).

> **Inventory Turnover Ratio**

The inventory turnover ratio is calculated by dividing the cost of goods sold (COGS) by the average inventory value. It indicates how many times a company’s inventory is sold and replaced over a period. A higher inventory turnover ratio suggests that a company is effectively managing its inventory by quickly converting it into sales. See documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_inventory_turnover_ratio).

> **Days of Inventory Outstanding**

The days sales in inventory ratio (DSI) is calculated by dividing the average inventory by the cost of goods sold (COGS) and then multiplying by the number of days in the period. It represents the average number of days it takes for a company to sell its inventory. A lower DSI indicates that the company is selling its inventory more quickly. See documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_days_of_inventory_outstanding).

> **Days of Sales Outstanding**

The days of sales outstanding (DSO) ratio is calculated by dividing the accounts receivable by the total credit sales and then multiplying by the number of days in the period. It represents the average number of days it takes for a company to collect payment on its credit sales. A lower DSO indicates that the company is collecting payments more quickly. See documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_days_of_sales_outstanding).

> **Operating Cycle**

The operating cycle represents the total time required to purchase inventory, convert it into finished goods, sell the goods to customers, and collect the accounts receivable. It is calculated by adding the days sales in inventory (DSI) and the days of sales outstanding (DSO). See documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_operating_cycle).

> **Accounts Payables Turnover Ratio**

The accounts payable turnover ratio indicates how many times, on average, a company pays off its accounts payable during a specific period. A higher turnover ratio is generally favorable, as it suggests that the company is efficiently managing its payments to suppliers. See documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_accounts_payables_turnover_ratio).

> **Days of Accounts Payable Outstanding**

The days payables outstanding (DPO) ratio is used to assess how efficiently a company manages its accounts payable. It calculates the average number of days it takes for a company to pay its suppliers after receiving an invoice. A higher DPO ratio indicates that the company is taking longer to pay its suppliers, which may have implications for its relationships with suppliers. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_days_of_accounts_payable_outstanding).

> **Cash Conversion Cycle (CCC)**

The Cash Conversion Cycle (CCC) is an important measure of a company’s liquidity management and efficiency in managing its working capital. It takes into account the time it takes to sell inventory, collect payments from customers, and pay suppliers. A shorter CCC indicates that a company is able to quickly convert its investments into cash, which can be a positive sign of efficient operations. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_cash_conversion_cycle).

> **Cash Conversion Efficiency (CCE)**

The cash conversion efficiency ratio is calculated by dividing the operating cash flow by the revenue. It indicates how much of a company’s sales are converted into cash. A higher cash conversion efficiency ratio is generally favorable, as it suggests that the company is able to convert its sales into cash more efficiently. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_cash_conversion_efficiency).

> **Receivables Turnover**

The receivables turnover ratio is an important measure of how well a company manages its accounts receivable. It indicates how quickly a company collects payments from its customers. A higher turnover ratio is generally favorable as it suggests that the company is collecting payments more quickly, which improves its cash flow and working capital management. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_receivables_turnover).

> **SGA to Revenue Ratio**

The SG&A to revenue ratio is calculated by dividing the total SG&A expenses by the company’s revenue and then multiplying by 100 to express it as a percentage. It provides insight into the efficiency of a company’s cost management and its ability to control its overhead costs. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_sga_to_revenue_ratio).

> **Fixed Asset Turnover**

The Fixed Asset Turnover ratio is calculated by dividing the company’s net sales by the average fixed assets. It indicates how well a company is utilizing its fixed assets to generate revenue. A higher ratio suggests more efficient utilization of fixed assets. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_fixed_asset_turnover).

> **Operating Ratio**

The operating ratio is calculated by dividing the company’s operating expenses by its net sales and multiplying by 100 to express it as a percentage. It provides insight into how efficiently a company is managing its operations. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_operating_ratio).

</details>
<details>
  <summary><b>Liquidity Ratios 💧</b></summary>

The liquidity ratios are used to assess a company’s ability to meet its short-term obligations using its short-term assets. They provide insight into the company’s short-term financial health and its ability to cover its current obligations using its liquid assets.

> **Current Ratio**

The current ratio is calculated by dividing a company’s current assets by its current liabilities. It indicates whether a company can meet its short -term obligations using its short-term assets. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_current_ratio).

> **Quick Ratio**

The quick ratio is calculated by subtracting inventory from current assets and then dividing the result by current liabilities. It provides insight into a company’s ability to cover its short -term liabilities using its most liquid assets without relying on inventory. This ratio is also referred to as the Acid Test Ratio. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_quick_ratio).

> **Cash Ratio**

The cash ratio is calculated by dividing the sum of cash and cash equivalents by current liabilities. It provides insight into a company’s immediate ability to cover its short -term obligations using its most liquid assets. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_cash_ratio).

> **Working Capital**

The working capital is calculated by subtracting total current liabilities from total current assets. It represents the company’s short -term financial health and its ability to cover its current obligations using its liquid assets. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_working_capital).

> **Operating Cash Flow Ratio**

The operating cash flow ratio is calculated by dividing operating cash flow by current liabilities. It indicates whether a company’s operating cash flow is sufficient to cover its short-term obligations. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_operating_cash_flow_ratio).

> **Operating Cash Flow Sales Ratio**

The operating cash flow to sales ratio is calculated by dividing operating cash flow by sales revenue. It indicates the proportion of sales revenue that is converted into cash from operating activities. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_operating_cash_flow_sales_ratio).

> **Short Term Coverage Ratio**

The short-term coverage ratio is calculated by dividing operating cash flow by short -term debt. It assesses the company’s ability to meet its short -term obligations using its operating cash flow. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_short_term_coverage_ratio).

</details>

<details>
  <summary><b>Profitability Ratios 💰</b></summary>

The profitability ratios are used to assess a company’s ability to generate profits from its operations. They provide insight into the company’s operational efficiency and its ability to generate profits from its revenue.

> **Gross Margin**

The gross margin ratio is calculated by subtracting the cost of goods sold (COGS) from the total revenue and then dividing the result by the total revenue. It represents the portion of revenue that contributes to covering other expenses and generating profit. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_gross_margin).

> **Operating Margin**

The operating margin ratio is calculated by subtracting the operating expenses from the total revenue and then dividing the result by the total revenue. It indicates how efficiently a company is managing its operating expenses in relation to its revenue. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_operating_margin).

> **Net Profit Margin**

The net profit margin ratio is calculated by dividing the net income by the total revenue. It indicates the portion of each dollar of revenue that represents profit after all expenses have been deducted. A higher net profit margin is generally considered favorable. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_net_profit_margin).

> **Interest Burden Ratio**

The Interest Burden or Interest Coverage Ratio is calculated by dividing the earnings before interest and taxes (EBIT) by the interest expenses. A higher ratio indicates that the company has more earnings to cover its interest expenses, which is generally considered favorable. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_interest_burden_ratio).

> **Income Before Tax Profit Margin**

The Income Before Tax Profit Margin or Pretax Profit Margin is calculated by dividing the pre -tax profit by the revenue. It provides insight into how efficiently a company is able to generate profits from its revenue. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_income_before_tax_profit_margin).

> **Effective Tax Rate**

The effective tax rate is calculated by dividing the income tax expense by the pre-tax income. A financial ratio that measures the percentage of pretax income that is paid as taxes. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_effective_tax_rate).

> **Return on Assets (RoA)**

The return on assets is calculated by dividing the net income by the average total assets. Note that it is false to take the total assets at the end of the period given that income statements report over the period whereas a balance sheet reports on the period. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_return_on_assets).

> **Return on Equity (RoE)**

The return on equity is calculated by dividing the net income by the average shareholders’ equity. Shareholders’ equity represents the residual interest in the assets of a company after deducting liabilities. Note that it is false to take the total assets at the end of the period given that income statements report over the period whereas a balance sheet reports on the period. ROE provides insight into the company’s ability to generate profits from the investments made by its shareholders. A higher ROE indicates that the company is using its equity effectively to generate higher returns for its shareholders. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_return_on_equity).

> **Return on Invested Capital (RoIC)**

The return on invested capital is calculated by dividing the net operating profit after taxes (NOPAT) by the average invested capital. Invested capital includes both equity and debt, making this ratio a valuable measure of how efficiently a company generates returns for all of its investors. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_return_on_invested_capital).


> **Income Quality Ratio**

The income quality ratio provides insights into the quality of a company’s reported earnings. By comparing the cash flow from operating activities to the net income, this ratio helps assess whether a company’s reported profits are backed by actual cash flow. A higher income quality ratio suggests higher earnings quality and a better ability to convert profits into cash flow. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_income_quality_ratio).

> **Return on Tangible Assets (RoTA)**

The return on tangible assets (ROTA) provides insights into the efficiency with which a company utilizes its tangible assets to generate profits. Tangible assets include physical assets such as buildings, machinery, and equipment. ROTA indicates how well a company can generate profits from its core operational assets. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_return_on_tangible_assets).

> **Return on Capital Employed (RoCE)**

The return on capital employed (ROCE) is a crucial financial metric that evaluates the efficiency and profitability of a company’s utilization of both equity and debt capital to generate profits. It assesses how well the company generates earnings relative to the total capital invested in the business. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_return_on_capital_employed).

> **Net Income per EBT**

The net income per earnings before taxes (EBT) ratio helps evaluate the extent to which a company’s net income is generated from its operating activities before considering the impact of income taxes. It gives insights into how effectively a company generates profit relative to its taxable income. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_net_income_per_ebt).

> **Free Cash Flow Operating Cash Flow Ratio**

The free cash flow to operating cash flow ratio helps assess how well a company’s operating activities translate into free cash flow, which is the cash available after all expenses and investments. A higher ratio indicates that the company is generating strong free cash flow relative to its operating cash flow, which could signify efficient capital management. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_free_cash_flow_operating_cash_flow_ratio).

> **Tax Burden Ratio**

The tax burden ratio measures the portion of a company’s earnings that is paid as taxes. A higher ratio indicates that a larger portion of the income is being retained by the company after taxes. This ratio provides insights into the tax efficiency of the company and its ability to manage its tax liabilities. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_tax_burden_ratio).

> **EBT to EBIT**

The EBT to EBIT ratio measures the proportion of a company’s earnings that is generated before paying interest and taxes. It provides insights into how a company’s operating performance is impacted by interest expenses and tax obligations. A higher ratio indicates that a larger portion of the company’s earnings is generated from its core operations before considering interest payments and taxes. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_ebt_to_ebit).

> **EBIT to Revenue**

The EBIT to Revenue ratio measures the company’s ability to generate profit from its core operations relative to its revenue. It provides insights into the operational efficiency and profitability of the company, as it excludes the impact of interest expenses and taxes on its earnings. A higher ratio indicates that a larger portion of the company’s revenue is converted into operating profit. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_ebit_to_revenue).

</details>

<details>
  <summary><b>Solvency Ratios 🪨</b></summary>

The solvency ratios are used to assess a company’s ability to meet its long-term obligations using its long-term assets. They provide insight into the company’s long-term financial health and its ability to cover its long-term obligations using its long-term assets.

> **Debt to Assets Ratio**

This ratio, also known as the Debt Ratio, indicates the percentage of a company’s total assets that are funded by debt. It is a measure of a company’s financial leverage and indicates the extent to which a company relies on borrowed funds to finance its operations. A higher ratio implies a higher level of debt in the company’s capital structure, which could increase financial risk. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_debt_to_assets_ratio).

> **Debt to Equity Ratio**

The debt to equity ratio, for short the D/E ratio, indicates the relative contribution of debt and equity to a company’s capital structure. It helps assess the level of financial risk a company carries due to its debt obligations. A higher ratio implies a higher reliance on debt to finance the business, which could increase risk but also potentially lead to higher returns for shareholders. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_debt_to_equity_ratio).

> **Interest Coverage Ratio**

The interest coverage ratio evaluates a company’s ability to meet its interest obligations from its operating income. A higher ratio indicates a company’s stronger ability to cover its interest payments using its earnings, implying lower financial risk. Conversely, a lower ratio suggests a company may have difficulty meeting its interest obligations and could be at higher risk of default. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_interest_coverage_ratio).

> **Equity Multiplier**

The equity multiplier helps assess the financial leverage of a company by indicating how much of its assets are financed by equity versus debt. A higher equity multiplier suggests that the company relies more on debt financing, which can amplify returns but also increases financial risk. Conversely, a lower equity multiplier indicates a larger portion of assets is financed by equity, potentially lowering financial risk. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_equity_multiplier).

> **Debt Service Coverage Ratio**

The debt service coverage ratio provides insights into a company’s ability to meet its debt obligations from its operating income. It is especially important for companies with significant debt obligations, as a lower ratio indicates higher financial risk and potential difficulties in servicing debt payments. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_debt_service_coverage_ratio).

> **Free Cash Flow Yield**

The free cash flow yield ratio is a measure of how efficiently a company generates free cash flow relative to its market value. It provides insights into whether the company’s valuation is reasonable compared to the amount of cash it generates. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_free_cash_flow_yield).

> **Net Debt to EBITDA Ratio**

The net debt to EBITDA ratio is a measure of a company’s ability to manage its debt obligations in relation to its earnings and cash flow. A lower ratio indicates better financial health and a stronger ability to manage debt. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_net_debt_to_ebitda_ratio).

> **Cash Flow Coverage Ratio**

The cash flow coverage ratio assesses a company’s ability to meet its debt obligations by comparing its operating cash flow to its total debt. A higher ratio indicates a stronger ability to cover its debt with cash generated from operations. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_cash_flow_coverage_ratio).

> **CAPEX Coverage Ratio**

The capex coverage ratio evaluates a company’s ability to fund its capital expenditures, which are essential for maintaining and growing its business, using the cash generated from its operations. A higher ratio indicates a stronger ability to fund capital investments from operating cash flow. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_capex_coverage_ratio).

> **CAPEX Dividend Coverage Ratio**

The capex dividend coverage ratio assesses whether a company’s cash flow from operations is sufficient to cover both its capital expenditures (which are essential for maintaining and growing its business) and its dividend payments to shareholders. A higher ratio indicates a stronger ability to fund both capex and dividends from operating cash flow. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_capex_dividend_coverage_ratio).

</details>

<details>
  <summary><b>Valuation Ratios ⚖️</b></summary>

The valuation ratios are used to assess the relative value of a company’s stock. They provide insight into the company’s valuation and its attractiveness as an investment opportunity.

> **Earnings per Share (EPS)**

The earnings per share (EPS) is a widely used financial metric that helps investors understand the profitability of a company on a per -share basis. It provides insight into the portion of a company’s earnings that is allocated to each outstanding share of its common stock. EPS is an important measure for investors and analysts when assessing a company’s financial performance and comparing it to other companies. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_earnings_per_share).


> **Revenue per Share (RPS)**

The revenue per share is an important metric that provides insight into a company’s ability to generate revenue on a per -share basis. It can help investors understand the company’s revenue -generation efficiency and its overall financial health. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_revenue_per_share).

> **Price Earnings Ratio (PE)**

The price earnings ratio is a widely used valuation metric that helps investors assess the relative value of a company’s stock. A higher P/E ratio may indicate that the market has high expectations for the company’s future growth, while a lower P/E ratio may suggest that the company is undervalued. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_price_earnings_ratio).

> **Price to Earnings Growth Ratio (PEG)**

The price--to-earnings growth (PEG) ratio provides a more comprehensive valuation measure compared to the P/E ratio alone. It takes into account a company’s earnings growth rate, allowing investors to assess whether a stock is overvalued or undervalued relative to its growth prospects. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_price_to_earnings_growth_ratio).

> **Book Value per Share**

The book value per share is a fundamental valuation metric that reflects the net worth of a company attributed to each outstanding share of common stock. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_book_value_per_share).

> **Price to Book Ratio (PB)**

The price to book ratio is a key valuation metric that helps investors assess whether a company’s stock is overvalued or undervalued relative to its underlying net asset value. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_price_to_book_ratio).

> **Interest Debt per Share**

The interest debt per share ratio provides insight into how much interest a company pays on its debt relative to its shareholder base. It can help investors assess the financial burden of interest expenses on the company’s profitability. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_interest_debt_per_share).

> **CAPEX per Share**

The capex per share ratio provides insight into how much capital a company invests in its operations and growth initiatives relative to its shareholder base. It can help investors assess the level of reinvestment into the business. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_capex_per_share).

> **Dividend Yield**

The dividend yield ratio is used by investors to assess the income potential of an investment in a company’s stock based on the dividends it pays out. A higher dividend yield can be attractive to income-seeking investors. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_dividend_yield).

> **Weighted Dividend Yield**

This dividend yield ratio takes into account the (diluted) weighted average shares and actual dividends paid as found in the cash flow statement. It provides a more accurate reflection of the dividends paid out per share, considering any changes in the number of shares. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_weighted_dividend_yield).

> **Price to Cash Flow Ratio (P/CF)**

The price to cash flow ratio is a key valuation metric that helps investors assess the relative value of a company’s stock. It is similar to the price to earnings ratio, but uses cash flow instead of earnings in the denominator. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_price_to_cash_flow_ratio).


> **Price to Free Cash Flow Ratio (P/FCF)**

This price to free cash flow ratio provides insight into how the market values a company’s ability to generate free cash flow. It is a valuation ratio that compares a company’s market price to its free cash flow per share. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_price_to_free_cash_flow_ratio).

> **Market Capitalization**

Market capitalization, often referred to as “market cap,” is the total value of a company’s outstanding shares of stock in the stock market. It is calculated by multiplying the current market price per share by the total number of outstanding shares. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_market_cap).

> **Enterprise Value**

The Enterprise Value (EV) is a measure of a company’s total value, often used as a more comprehensive alternative to market capitalization. It is calculated as the sum of a company’s market capitalization, outstanding debt, minority interest, and preferred equity, minus the cash and cash equivalents. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_enterprise_value).

> **EV to Sales Ratio**

This EV to Sales ratio compares the enterprise value (EV) to the total revenue generated by the company. It can provide insights into how efficiently a company is using its revenue to generate value for its investors. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_ev_to_sales_ratio).

> **EV to EBITDA Ratio**

This EV to EBITDA ratio is a valuation ratio that measures a company’s total value (including debt and equity) relative to its EBITDA. It helps investors understand how many times the enterprise value exceeds the company’s EBITDA, providing insights into the company’s debt load and operating performance. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_ev_to_ebitda_ratio).

> **EV to Operating Cashflow Ratio**

The EV to Operating Cashflow Ratio is a valuation ratio that measures a company’s total value (including debt and equity) relative to its operating cash flow. This ratio provides insights into how many times the enterprise value exceeds the company’s operating cash flow, indicating the company’s ability to generate cash from its operations. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_ev_to_operating_cashflow_ratio).

> **EV to EBIT**

Calculate the enterprise value over earnings before interest and taxes (EBIT) ratio, which is a valuation metric that compares a company’s total value (including debt and equity) relative to its earnings before interest and taxes. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_ev_to_ebit).

> **Earnings Yield**

The earnings yield ratio is a valuation metric that provides insights into how much a company’s earnings contribute to its stock price. It compares the earnings per share to the market price per share, helping investors understand the earnings potential of the company relative to its current market value. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_earnings_yield).

> **Dividend Payout Ratio**

The payout ratio is a financial metric that helps investors assess the portion of a company’s earnings that is being distributed to shareholders in the form of dividends. It’s a valuable indicator for dividend investors as it indicates the sustainability of dividend payments and the company’s approach to distributing profits. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_dividend_payout_ratio).

> **Reinvestment Rate**

The reinvestment rate is a financial metric that helps investors assess the portion of a company’s earnings that is being retained by the company for future growth. It’s a valuable indicator for dividend investors as it indicates the sustainability of dividend payments and the company’s approach to distributing profits. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_reinvestment_rate).

> **Tangible Asset Value**

The tangible asset value is a financial metric that represents the total value of a company’s assets that can be used to generate revenue. Tangible assets are those physical assets that have a finite monetary value and can be sold, used, or consumed. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_tangible_asset_value).

> **Net Current Asset Value**

The net current asset value is a financial metric that represents the total value of a company’s current assets minus its current liabilities. It indicates the extent to which a company’s short -term assets exceed its short-term liabilities. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/ratios#get_net_current_asset_value).

</details>

## Financial Models

The Models module is meant to execute well-known models such as DUPONT and the Discounted Cash Flow (DCF) model. These models are also directly related to the data retrieved from the Toolkit module. **Find the Notebook [here](https://www.jeroenbouma.com/projects/financetoolkit/models-module) and the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/models) which includes an explanation about the model, the parameters and an example.**

<details>
  <summary><b>Financial Models</b></summary>

The financial models are used to analyze a company’s financial performance and assess its financial health. They provide insights into the company’s profitability, efficiency, liquidity, solvency, and valuation.

> **DuPont Analysis**

The Dupont analysis is a method used to dissect and understand the factors that drive a company’s return on equity (ROE). It breaks down the ROE into three key components: Profit Margin, Asset Turnover, and Financial Leverage. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/models#get_dupont_analysis).

> **Extended DuPont Analysis**

The Extended Dupont analysis is an advanced method used to break down the return on equity (ROE) into multiple components, providing a more detailed insight into the factors influencing a company’s profitability. It considers additional metrics such as Return on Assets (ROA), Total Asset Turnover, Financial Leverage, and more. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/models#get_extended_dupont_analysis).

> **Enterprise Value Breakdown**

The Enterprise Value (EV) breakdown provides a detailed view of its components. This include the market capitalization, the total debt, the minority interest, the preferred equity, the cash and cash equivalents, and the enterprise value. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/models#get_enterprise_value_breakdown).

> **Weighted Average Cost of Capital (WACC)**

The Weighted Average Cost of Capital (WACC) is a financial metric used to estimate the cost of capital for a company. It represents the average rate of return a company must pay to its investors for using their capital. WACC takes into account the cost of both equity and debt, weighted by their respective proportions in the company’s capital structure. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/models#get_weighted_average_cost_of_capital).

> **Intrinsic Valuation**

Intrinsic value is a fundamental concept in finance and investing that represents the true worth or value of an asset, security, or investment, independent of its current market price or prevailing market sentiment. It is a concept often associated with the value investing philosophy, made famous by legendary investors like Benjamin Graham and Warren Buffett. Understanding intrinsic value is crucial for investors looking to make informed decisions about where to allocate their capital.

This functionality uses DCF, or Discounted Cash Flow which is a widely used financial valuation method that allows investors and analysts to estimate the intrinsic value of an investment or business based on its expected future cash flows. It is a fundamental tool in finance and investment analysis, providing a systematic way to assess the present value of future cash flows while considering the time value of money. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/models#get_intrinsic_valuation).

> **Altman Z-Score**

The Altman Z -Score is a financial metric used to predict the likelihood of a company going bankrupt. The Altman Z -Score is calculated using several financial ratios, including working capital to total assets, retained earnings to total assets, earnings before interest and taxes (EBIT) to total assets, market value of equity to book value of total liabilities, and sales to total assets. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/models#get_altman_z_score).


> **Piotroski F-Score**

The Piotroski Score is a comprehensive financial assessment tool that helps investors and analysts evaluate a company’s financial health and fundamental strength. The Piotroski Score was developed by Joseph Piotroski and is based on a set of nine fundamental financial criteria. Each criterion is assigned a score of 0 or 1, and the scores are then summed to calculate the Piotroski Score. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/models#get_piotroski_f_score).

> **Present Value of Growth Opportunities (PVGO)**

The Present Value of Growth Opportunities (PVGO) is a financial metric that represents the present value of a company’s future growth opportunities. It is calculated as the difference between the company’s current stock price and the discounted value of its future cash flows. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/models#get_present_value_of_growth_opportunities).

</details>

## Options and Greeks

The Options module is meant to calculate theoretical values based on Black Scholes for European Options. This includes the theoretical value of the Option for each Strike Price and Expiration Date which is intelligently chosen. It also includes the First, Second and Third Order Greeks such as Delta, Gamma, Vega, Vomma and Ultima. **Find the Notebook [here](https://www.jeroenbouma.com/projects/financetoolkit/options-module) and the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/options) which includes an explanation about the model, the parameters and an example.**

<details>
  <summary><b>Option Pricing 📊</b></summary>

Option pricing are mathematical models that calculate the theoretical value of an option contract. It is widely used by traders and investors to determine the theoretical value of an option, and to assess the potential risks and rewards of a position.

> **Option Chains**

Get the Option Chains which gives information about the currently available options as reported by Yahoo Finance. This returns the Contract Symbol, Strike Currency, Last Price, Absolute Change, Percent Change, Volume, Open Interest, Bid Pirce, Ask Price, Expiration, Last Trade Date, Implied Volatility and whether the option is In The Money. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/options#get_option_chains).

> **Black Scholes Model**

The Black Scholes Model is a mathematical model used to estimate the price of European-style options. It is widely used by traders and investors to determine the theoretical value of an option, and to assess the potential risks and rewards of a position.

Within Risk Management, defining the theoretical value of an option is important to assess the potential risk and rewards of an option position. A position that could be used to hedge a portfolio, for example, is a long put option. The theoretical value of this option can be used to determine the potential risk and rewards of this position.

The Black Scholes Model is based on several assumptions, including the following:

- The option is European and can only be exercised at expiration.
- The underlying stock follows a lognormal distribution.
- The risk-free rate and volatility of the underlying stock are known and constant.
- The returns on the underlying stock are normally distributed.

Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/options#get_black_scholes_model).

</details>

<details>
    <summary><b>First-Order Greeks</b></summary>

The first-order greeks are Delta, Dual Delta, Vega, Theta, Rho, Epsilon and Lambda. These greeks are used to assess the sensitivity of an option’s price to changes in the underlying asset’s price, volatility, time to expiration, risk-free interest rate, dividend yield, and strike price.

> **Delta**

The delta is the rate of change of the option price with respect to the price of the underlying asset. The Delta can be interpreted as follows:

- For call options, Delta is positive, indicating that the option price tends to move in the same direction as the underlying asset’s price.
- For put options, Delta is negative, indicating that the option price tends to move in the opposite direction to the underlying asset’s price.

Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/options#get_delta).

> **Dual Delta**

The dual delta is the actual probability of an option finishing in the money which is the first derivative of option price with respect to strike. The Dual Delta can be interpreted as the probability of an option finishing in the money. For example, if the Dual Delta is 0.5, then the probability of the option finishing in the money is 50%. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/options#get_dual_delta).

> **Vega**

The vega is the rate of change of the option price with respect to the volatility of the underlying asset. The Vega can be interpreted as follows:

- If Vega is positive, it indicates that the option value will increase as the volatility increases, and vice versa.
- If Vega is negative, it implies that the option value will decrease as the volatility increases, and vice versa.

Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/options#get_vega).

> **Theta**

The theta is the rate of change of the option price with respect to the passage of time. The Theta can be interpreted as follows:

- If Theta is positive, it indicates that the option value will increase as the time to expiration increases, and vice versa.
- If Theta is negative, it implies that the option value will decrease as the time to expiration increases, and vice versa.

Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/options#get_theta).

> **Rho**

The rho is the rate of change of the option price with respect to the risk free interest rate. The Rho can be interpreted as follows:

- If Rho is positive, it indicates that the option value will increase as the risk free rate increases, and vice versa.
- If Rho is negative, it implies that the option value will decrease as the risk free rate increases, and vice versa.

Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/options#get_rho).

> **Epsilon**

The epsilon is the rate of change of the option price with respect to the dividend yield. The Epsilon can be interpreted as follows:

- If Epislon is positive, it indicates that the option value will increase as the dividend yield increases, and vice versa.
- If Epislon is negative, it implies that the option value will decrease as the dividend yield increases, and vice versa.

Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/options#get_epsilon).

> **Lambda**

The lambda is the rate of change of the option price with respect to the underlying price. The Lambda can be interpreted as follows:

- If Lambda is positive, it indicates that the option value will increase as the underlying price increases, and vice versa.
- If Lambda is negative, it implies that the option value will decrease as the underlying price increases, and vice versa.

Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/options#get_lambda).

</details>

<details>
    <summary><b>Second-Order Greeks</b></summary>:

The second-order greeks are are Gamma, Dual Gamma, Vanna, Charm, Vomma, Vera and Veta. These greeks are used to assess the sensitivity of the first-order greeks to small changes in the underlying asset’s price, volatility, time to expiration, risk-free interest rate, dividend yield, and strike price.

> **Gamma**

The gamma is the rate of change of the delta with respect to the price of the underlying asset. The Gamma can be interpreted as follows:

- If Gamma is high, it indicates that the option’s Delta is highly sensitive to changes in the underlying asset’s price. The option’s Delta will change more significantly with small movements in the stock price.
- If Gamma is low, it suggests that the option’s Delta is relatively insensitive to changes in the underlying asset’s price. The option’s Delta changes more gradually with movements in the stock price.

Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/options#get_gamma).

> **Dual Gamma**

Similar to Dual Delta, the Dual Gamma measures how fast dual delta changes with respect to the strike price. It therefore measures the change in the probability of an option finishing in the money with respect to the strike price. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/options#get_dual_gamma).

> **Vanna**

The vanna is the rate of change of the vega with respect to the price of the underlying asset. The Vanna can be interpreted as follows:

- If Vanna is positive, it indicates that the Delta of the option becomes more positive as both the underlying asset’s price and implied volatility increase, and more negative as they both decrease.
- If Vanna is negative, it suggests that the Delta of the option becomes more negative as both the underlying asset’s price and implied volatility increase, and more positive as they both decrease.

Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/options#get_vanna).

> **Charm**

The charm is the rate of change of the delta with respect to the time to expiration. The Charm can be interpreted as follows:

- If Charm is positive, it suggests that the option’s Delta is becoming more positive over time. In other words, the option is gaining sensitivity to changes in the underlying asset’s price as time passes.
- If Charm is negative, it indicates that the option’s Delta is becoming more negative over time. The option is losing sensitivity to changes in the underlying asset’s price as time passes.

Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/options#get_charm).

> **Vomma**

The vomma is the rate of change of the vega with respect to the volatility of the underlying asset. The vomma can be interpreted as follows:

- If Vomma is high, it indicates that the option’s Vega is highly sensitive to changes in implied volatility. The option’s value will experience more significant fluctuations with variations in implied volatility.
- If Vomma is low, it suggests that the option’s Vega is relatively less sensitive to changes in implied volatility.

Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/options#get_vomma).

> **Vera**

The vera is the rate of change of the rho with respect to volatility. The Vera can be interpreted as follows:

- If Vera is positive, it indicates that the option’s Rho is becoming more positive over time. In other words, the option is gaining sensitivity to changes in the risk free rate as time passes.
- If Vera is negative, it suggests that the option’s Rho is becoming more negative over time. The option is losing sensitivity to changes in the risk free rate as time passes.

Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/options#get_vera).

> **Veta**

The veta is the rate of change of the vega with respect to the time to expiration. The Veta can be interpreted as follows:

- If Veta is positive, it indicates that the option’s Vega is becoming more positive over time. In other words, the option is gaining sensitivity to changes in implied volatility as time passes.
- If Veta is negative, it suggests that the option’s Vega is becoming more negative over time. The option is losing sensitivity to changes in implied volatility as time passes.

Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/options#get_veta).

> **Partial Derivative**

The partial derivative is the rate of change of the option price with respect to the strike price. The partial derivative is used in the Breeden -Litzenberger theorem is used for risk -neutral valuation and was developed by Fischer Black and Robert Litzenberger in 1978. The theorem states that the price of any derivative security can be calculated by finding the expected value of the derivative under a risk -neutral measure. The theorem is based on the Black -Scholes model and the assumption that the underlying asset follows a lognormal distribution. See the paper: https://www.jstor.org/stable/2352653. Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/options#get_partial_derivative).

</details>

<details>
    <summary><b>Third-Order Greeks</b></summary>

The third-order greeks are Speed, Zomma, Color and Ultima. These greeks are used to assess the sensitivity of the second-order greeks to small changes in the underlying asset’s price, volatility, time to expiration, risk-free interest rate, dividend yield, and strike price.

> **Speed**

The speed is the rate of change of the gamma with respect to the price of the underlying asset. The Speed can be interpreted as follows:

- If Speed is positive, it indicates that the option’s Gamma is becoming more positive over time. In other words, the option is gaining sensitivity to changes in the underlying price as time passes.
- If Speed is negative, it suggests that the option’s Gamma is becoming more negative over time. The option is losing sensitivity to changes in the underlying price as time passes.

Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/options#get_speed).

> **Zomma**

The zomma is the rate of change of the gamma with respect to volatility. The Zomma can be interpreted as follows:

- If Zomma is positive, it indicates that the option’s Gamma is becoming more positive over time. In other words, the option is gaining sensitivity to changes in volatility as time passes.
- If Zomma is negative, it suggests that the option’s Gamma is becoming more negative over time. The option is losing sensitivity to changes in volatility as time passes.

Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/options#get_zomma).

> **Color**

The color is the rate of change of the gamma with respect to time to expiration. The Color can be interpreted as follows:

- If Color is positive, it indicates that the option’s Gamma is becoming more positive over time. In other words, the option is gaining sensitivity to changes in time to expiration as time passes.
- If Color is negative, it suggests that the option’s Gamma is becoming more negative over time. The option is losing sensitivity to changes in time to expiration as time passes.

Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/options#get_color).

> **Ultima**

The ultima is the rate of change of the vomma with respect to volatility. The Ultima can be interpreted as follows:

- If Ultima is positive, it indicates that the option’s vomma is becoming more positive over time. In other words, the option is gaining sensitivity to changes in volatility as time passes.
- If Ultima is negative, it suggests that the option’s vomma is becoming more negative over time. The option is losing sensitivity to changes in volatility as time passes.

Find the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/options#get_ultima).

</details>

## Performance Metrics

The Performance module is meant to calculate important performance metrics such as Sharpe Ratio, Sortino Ratio, Treynor Ratio, Information Ratio, Jensen's Alpha, Beta, Capital Asset Pricing Model, Factor Models and more. **Find the Notebook [here](https://www.jeroenbouma.com/projects/financetoolkit/performance-module) and the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/performance) which includes an explanation about the performance metrics, the parameters and an example.**

- Alpha
- Jensen's Alpha 
- Beta
- Capital Asset Pricing Model (CAPM)
- Factor Asset Correlations
- Factor Correlations
- Fama-French 5 Factor Model (FF)
- Treynor Ratio
- Sharpe Ratio
- Sortino Ratio
- Ulcer Performance Index (UPI)
- M2 Ratio
- Tracking Error
- Information Error
- Compound Annual Growth Rate (CAGR)

## Risk Metrics

The Risk module is meant to calculate important risk metrics such as Value at Risk (VaR), Conditional Value at Risk (cVaR), Maximum Drawdown, Correlations, Beta, GARCH, EWMA and more. **Find the Notebook [here](https://www.jeroenbouma.com/projects/financetoolkit/risk-module) and the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/risk) which includes an explanation about the risk metrics, the parameters and an example.**

- Value at Risk (VaR) with distributions Historical, Gaussian, Student-t, Cornish-Fisher.
- Conditional Value at Risk (cVaR) with distributions Historical, Gaussian, Laplace, Logistic.
- Entropic Value at Risk (eVaR) with a Gaussian distribution. 
- GARCH (Generalized Autoregressive Conditional Heteroskedasticity) with Forecasting
- Ulcer Index (UI)
- Maximum Drawdown (MDD)
- Skewness
- Kurtosis

## Technical Indicators

The Technicals Module contains 30+ Technical Indicators that can be used to analyse companies. These ratios are divided into 4 categories which are breadth, momentum, overlap and volatility. Each indicator is calculated using the data from the Toolkit module. **Find the Notebook [here](https://www.jeroenbouma.com/projects/financetoolkit/technicals-module) and the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/technicals) which includes an explanation about the indicator, the parameters and an example.**

### Breadth Indicators 💨
- McClellan Oscillator
- Advancers/Decliners Ratio
- On-Balance Volume (OBV)
- Accumulation/Distribution Line (ADL)
- Chaikin Oscillator

### Momentum Indicators ⏱️
- Money Flow Index
- Williams %R
- Aroon Indicator
- Commodity Channel Index
- Relative Vigor Index
- Force Index
- Ultimate Oscillator
- Percentage Price Oscillator
- Detrended Price Oscillator
- Average Directional Index (ADX)
- Chande Momentum Oscillator (CMO)
- Ichimoku Cloud
- Stochastic Oscillator
- Moving Average Convergence Divergence (MACD)
- Relative Strength Index (RSI)
- Balance of Power (BOP)

### Overlap Indicators 🌈
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Double Exponential Moving Average (DEMA)
- Triple Exponential Moving Average (TRIX)
- Triangular Moving Average (TMA)

### Volatility Indicators ⚠️
- True Range (TR)
- Average True Range (ATR)
- Keltners Channels
- Bollinger Bands

## Key Economic Indicators

The Economics Module contains a variety of Key Economic Indicators that help in understanding the health and performance of more than 60 different countries. This module can be called directly via the Toolkit but also separately if desired through `from financetoolkit import Economics`. **Find the Notebook [here](https://www.jeroenbouma.com/projects/financetoolkit/economics-module) and the documentation [here](https://www.jeroenbouma.com/projects/financetoolkit/docs/economics) which includes an explanation about each indicator, the parameters and an example.**

### Economy 💵
- Gross Domestic Product (GDP) including Growth and Forecasts
- Consumer Confidence Index (CCI)
- Business Confidence Index (BCI)
- Composite Leading Indicator (CLI)
- Consumer Price Index (CPI)
- Producer Price Index (PPI)
- House and Rent Prices
- Share Prices

### Finance 💹
- Long Term Interest Rates (10 year)
- Short Term Interest Rates (3 month)
- Narrow Money (M1)
- Broad Money (M3)
- Purchasing Power Parity (PPP)
- Exchange Rates

### Central Banks 🏦
- Main Refinancing Operations (ECB)
- Marginal Lending Facility (ECB)
- Deposit Facility (ECB)
- Effective Federal Funds Rate (FED)
- Overnight Bank Funding Rate (FED)
- Tri-Party General Collateral Rate (FED)
- Broad General Collateral Rate (FED)
- Secured Overnight Financing Rate (FED)

### Environment 💚
- Renewable Energy as % of Total Energy
- Environmental Tax as % of GDP
- Greenhouse Gas Emissions
- Crude Oil Production and Prices

### Government 🏛️
- Deficit as % of GDP
- Revenue as % of GDP
- Spending as % of GDP
- Debt as % of GDP
- Financial Wealth as % of GDP
- Production Costs as % of GDP
- Central Spending based on the System of National accounts (SNA)
- Trust in Government

### Jobs 🗂️
- Unemployment Rates
- Labour Productivity
- Income Inequality

### Society 👪
- Population
- Young Population
- Working Age Population
- Elderly Population
- Fertility Rates
- Old-Age Dependency Ratio
- Poverty Rate

# Questions & Answers

This section includes frequently asked questions and is meant to clear up confusion about certain results and/or deviations from other sources. If you have any questions that are not answered here, feel free to reach out to me via the contact details below.

> **How do you deal with companies that have different fiscal years?**

For any financial statement, I make sure to line it up with the corresponding *calendar* period. For example, Apple's Q4 2023 relates to July to September of 2023. This corresponds to the calendar period Q3 which is why I normalize Apple's numbers to Q3 2023 instead. This is done to allow for comparison between companies that have different fiscal years.

> **Why do the numbers in the financial statements sometimes deviate from the data from FinancialModelingPrep?**

When looking at a company such as Hyundai Motor Company (ticker: 005380.KS), you will notice that the financial statements are reported in KRW (South Korean won). As this specific ticker is listed on the Korean Exchange, the historical market data will also be reported in KRW. However, if you use the ticker HYMTF, which is listed on the American OTC market, the historical market data will be reported in USD. To deal with this discrepancy, the end of year or end of quarter exchange rate is retrieved which is used to convert the financial statements to USD. This is done to prevent ratio calculations such as the Free Cash Flow Yield (which is based on the market capitalization) or Price Earnings Ratio (which is based on the stock price) from being incorrect. This can be disabled by setting `convert_currency=False` in the Toolkit initialization. It is recommended to always use the ticker that is listed on the exchange where the company is based.

> **How can I get TTM (Trailing Twelve Months) and Growth metrics?**

Most functions will have the option to define the `trailing` parameter. This lets you define the number of periods that you want to use to calculate the trailing metrics. For example, if you want to calculate the trailing 12-month (TTM) Price-to-Earnings Ratio, you can set `trailing=4` when you have set `quarterly=True` in the Toolkit initialization. The same goes for growth metrics which can be calculated by setting `growth=True`. This will calculate the growth for each period based on the previous period. This also includes a `lag` parameter in which you can define lagged growth. Furthermore, you can also combine the trailing and growth parameters to get trailing growth. For example, set `trailing=4` and `growth=True`  for the Price-to-Earnings Ratio which will then calculate the TTM growth.

> **How can I save the data periodically so that I don't have to retrieve it every single time again?**

The Toolkit initialization has the ability to add custom datasets which can serve as input for external datasets from different providers but also to prevent downloading all of the same data again. Have a look at [this comment](https://github.com/JerBouma/FinanceToolkit/issues/68#issuecomment-1748684501) that explains this further.

> **What is the "Benchmark" that is automatically obtained when acquiring historical data?**

This is related to the `benchmark_ticker` parameter which is set to "SPY" (S&P 500) by default. This is important when calculating performance metrics such as the Sharpe Ratio or Treynor Ratio that require a market return. This can be disabled by setting `benchmark_ticker=None` in the Toolkit initialization.

> **Data collection seems to be slow, what could be the issue?**

Generally, it should take less than 15 seconds to retrieve the historical data of 100 tickers. If it takes much longer, this could be due to reaching the API limit (the Starter plan has 250 requests per minute), due to a slow internet connection or due to unoptimized code. As the Finance Toolkit makes use of threading, initializing the Toolkit with a single ticker will result in a slow process. This is because the Toolkit will have to wait for the previous request to finish before it can start the next one. Therefore, it is recommended to initialize the Toolkit with all tickers you want to analyze. If it is taking 10+ minutes consider having a look at [this issue](https://github.com/JerBouma/FinanceToolkit/issues/99#issuecomment-1889726000) that managed to resolve the problem.

> **Are you part of FinancialModelingPrep?**

*No, I am not*. I've merely picked them as the primary data provider given that they have a generous free tier and fair pricing compared to other providers. Therefore, any questions related to the data should go through [their contact form](https://site.financialmodelingprep.com/contact). When it comes to any type of ratios, performance metrics, risk metrics, technical indicators or economic indicators, feel free to reach out to me as this is my own work.

# Contributing
First off all, thank you for taking the time to contribute (or at least read the Contributing Guidelines)! 🚀

___ 

<b><div align="center">Find the Contributing Guidelines <a href="/CONTRIBUTING.md">here</a>.</div></b>
___

The goal of the Finance Toolkit is to make any type of financial calculation as transparent and efficient as possible. I want to make these type of calculations as accessible to anyone as possible and seeing how many websites exists that do the same thing (but instead you have to pay) gave me plenty of reasons to work on this.

# Contact
If you have any questions about the FinanceToolkit or would like to share with me what you have been working on, feel free to reach out to me via:

- **Website**: https://jeroenbouma.com/
- **Twitter**: https://twitter.com/JerBouma
- **LinkedIn:** https://www.linkedin.com/in/boumajeroen/
- **Email:** jer.bouma@gmail.com
- **Discord:** add me on Discord **`JerBouma`**

If you'd like to support my efforts, either help me out by contributing to the package or [Sponsor Me](https://github.com/sponsors/JerBouma).

[![Star History Chart](https://api.star-history.com/svg?repos=JerBouma/FinanceToolkit&type=Date)](https://star-history.com/#JerBouma/FinanceToolkit&Date)
