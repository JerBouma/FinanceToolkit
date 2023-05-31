These files can be downloaded by clicking [here](). Set these up by defining the paths within the `Analyzer` module as follows:

```python
from financetoolkit.analyzer import Analyzer

companies = Analyzer(
    tickers=['AAPL', 'MSFT'],
    source='Custom',
    balance_sheet_format_location="LOCATION_TO_CSV_FILE",
    income_statement_format_location="LOCATION_TO_CSV_FILE",
    cash_flow_statement_format_location="LOCATION_TO_CSV_FILE"
)
```

Once defined, it will automatically take the "Custom" and "Generic" columns of the CSV and use that for the normalization process. Then, by providing your own balance sheet, income statement and cash flow statement you are able to calculate the ratios in your desired format. So, if you would use:


```python

companies.efficiency_ratios(
    custom_balance_sheet="YOUR_BALANCE_SHEET_AS_DATAFRAME",
    custom_income_statement="YOUR_INCOME_STATEMENT_AS_DATAFRAME"
)
```

It will do the calculations based on your input. As an example, this would return something like the following:

|                |   Asset Turnover Ratio |   Inventory Turnover Ratio |   Payables Turnover Ratio |   Days of Inventory Outstanding (DIO) |   Days of Sales Outstanding (DSO) |   Operating Cycle (CC) |   Days of Payables Outstanding (DPO) |   Cash Conversion Cycle (CCC) |
|:---------------|-----------------------:|---------------------------:|--------------------------:|--------------------------------------:|----------------------------------:|-----------------------:|-------------------------------------:|------------------------------:|
| AAPL, 2018 |               0.726215 |                    34.6222 |                   3.10971 |                              10.5424  |                           67.3325 |                77.8749 |                             117.374  |                    -39.4995   |
| AAPL, 2019 |               0.768572 |                    34.2048 |                   3.07222 |                              10.671   |                           64.2588 |                74.9298 |                             118.807  |                    -43.8767   |
| AAPL, 2020 |               0.847562 |                    35.8491 |                   3.21991 |                              10.1816  |                           49.7875 |                59.9691 |                             113.357  |                    -53.3882   |
| AAPL, 2021 |               1.04221  |                    45.0296 |                   4.04449 |                               8.10578 |                           51.391  |                59.4967 |                              90.2463 |                    -30.7496   |
| AAPL, 2022 |               1.11785  |                    47.2633 |                   4.24511 |                               7.72269 |                           56.4002 |                64.1229 |                              85.9812 |                    -21.8583   |
| MSFT, 2018 |               0.426351 |                    14.7534 |                   2.96428 |                              24.74    |                           87.5821 |               112.322  |                             123.133  |                    -10.8107   |
| MSFT, 2019 |               0.439157 |                    16.5064 |                   3.31648 |                              22.1127  |                           85.6326 |               107.745  |                             110.056  |                     -2.31108  |
| MSFT, 2020 |               0.474642 |                    17.725  |                   3.56134 |                              20.5923  |                           81.6978 |               102.29   |                             102.49   |                     -0.199432 |
| MSFT, 2021 |               0.503591 |                    20.0923 |                   4.03698 |                              18.1661  |                           82.6097 |               100.776  |                              90.4142 |                     10.3616   |
| MSFT, 2022 |               0.543444 |                    24.0999 |                   4.84218 |                              15.1453  |                           81.4811 |                96.6265 |                              75.3793 |                     21.2471   |