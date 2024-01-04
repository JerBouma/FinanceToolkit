"""Discovery Module"""
__docformat__ = "google"

import pandas as pd

from financetoolkit.discovery import discovery_model

# pylint: disable=too-many-instance-attributes,too-few-public-methods,too-many-lines,
# pylint: disable=too-many-locals,line-too-long,too-many-public-methods
# ruff: noqa: E501


class Discovery:
    """
    The Discovery module contains a collection of functions that are meant to get
    find companies and other financial instruments. Given that the Toolkit itself expects
    a ticker symbol, these functions are meant to help find the ticker symbol for a given
    company or financial instrument.
    """

    def __init__(
        self,
        api_key: str | None = None,
    ):
        """
        Initializes the Discovery Controller Class.

        Args:
            api_key (str): An API key from FinancialModelingPrep. Obtain one here: https://www.jeroenbouma.com/fmp

        As an example:

        ```python
        from financetoolkit import Discovery

        discovery = Discovery(api_key="FINANCIAL_MODELING_PREP_KEY")

        stock_list = discovery.get_stock_list()

        # The total list equals over 60.000 rows
        stock_list.iloc[48000:48010]
        ```

        Which returns:

        | Symbol     | Name                                |   Price | Exchange                         | Exchange Code   |
        |:-----------|:------------------------------------|--------:|:---------------------------------|:----------------|
        | RBL.AX     | Redbubble Limited                   |   0.54  | Australian Securities Exchange   | ASX             |
        | RBL.BO     | Rane Brake Lining Limited           | 870.05  | Bombay Stock Exchange            | BSE             |
        | RBL.NS     | Rane Brake Lining Limited           | 870.05  | National Stock Exchange of India | NSE             |
        | RBLAY      | Robinsons Land Corporation          |   4.61  | Other OTC                        | PNK             |
        | RBLBANK.BO | RBL Bank Limited                    | 280.9   | Bombay Stock Exchange            | BSE             |
        | RBLBANK.NS | RBL Bank Limited                    | 280.9   | National Stock Exchange of India | NSE             |
        | RBLN-B.CO  | Roblon A/S                          |  91.8   | Copenhagen                       | CPH             |
        | RBLX       | Roblox Corporation                  |  45.72  | New York Stock Exchange          | NYSE            |
        | RBMNF      | Rugby Resources Ltd.                |   0.065 | Other OTC                        | PNK             |
        | RBMS.JK    | PT Ristia Bintang Mahkotasejati Tbk |  50     | Jakarta Stock Exchange           | JKT             |
        """
        if not api_key:
            raise ValueError(
                "Please enter an API key from FinancialModelingPrep. "
                "For more information, look here: https://www.jeroenbouma.com/fmp"
            )

        self._api_key = api_key

    def search_instruments(self, query: str | None = None) -> pd.DataFrame:
        """
        The search instruments function allows you to search for a company or financial instrument
        by name. It returns a dataframe with all the symbols that match the query.

        Args:
            query (str): A query to search for, e.g. 'META'.

        Returns:
            pd.DataFrame: A dataframe with all the symbols that match the query.

        As an example:

        ```python
        from financetoolkit import Discovery

        discovery = Discovery(api_key="FINANCIAL_MODELING_PREP_KEY")

        discovery.search_instruments(query='META')
        ```

        Which returns:

        | Symbol   | Name                                  | Currency   | Exchange               | Exchange Code   |
        |:---------|:--------------------------------------|:-----------|:-----------------------|:----------------|
        | META     | Meta Platforms, Inc.                  | USD        | NASDAQ Global Select   | NASDAQ          |
        | META.L   | WisdomTree Industrial Metals Enhanced | USD        | London Stock Exchange  | LSE             |
        | METAUSD  | Metadium USD                          | USD        | CCC                    | CRYPTO          |
        | META.MI  | WisdomTree Industrial Metals Enhanced | EUR        | Milan                  | MIL             |
        | META.JK  | PT Nusantara Infrastructure Tbk       | IDR        | Jakarta Stock Exchange | JKT             |

        """
        if not query:
            raise ValueError(
                "Please enter a query to search for, e.g. search_instruments(query='META'). "
            )

        symbol_list = discovery_model.get_instruments(
            api_key=self._api_key, query=query
        )

        return symbol_list

    def get_stock_screener(
        self,
        market_cap_higher: int | None = None,
        market_cap_lower: int | None = None,
        price_higher: int | None = None,
        price_lower: int | None = None,
        beta_higher: int | None = None,
        beta_lower: int | None = None,
        volume_higher: int | None = None,
        volume_lower: int | None = None,
        dividend_higher: int | None = None,
        dividend_lower: int | None = None,
        is_etf: bool | None = None,
    ):
        """
        Screen stocks based on a set of criteria. This can be useful to find companies that match
        a specific criteria or your analysis. Further filtering can be done by utilising the
        Finance Toolkit and calculating the relevant ratios to filter by. This can be:

        - Market capitalization (market_cap_higher, market_cap_lower)
        - Price (price_higher, price_lower)
        - Beta (beta_higher, beta_lower)
        - Volume (volume_higher, volume_lower)
        - Dividend (dividend_higher, dividend_lower)

        Note that the limit is 1000 companies. Thus if you hit the 1000, it is recommended
        to narrow down your search to prevent companies from being excluded simply because
        of this limit.

        Args:
            market_cap_higher (int): The minimum market capitalization of the stock.
            market_cap_lower (int): The maximum market capitalization of the stock.
            price_higher (int): The minimum price of the stock.
            price_lower (int): The maximum price of the stock.
            beta_higher (int): The minimum beta of the stock.
            beta_lower (int): The maximum beta of the stock.
            volume_higher (int): The minimum volume of the stock.
            volume_lower (int): The maximum volume of the stock.
            dividend_higher (int): The minimum dividend of the stock.
            dividend_lower (int): The maximum dividend of the stock.

        Returns:
            pd.DataFrame: A dataframe with all the symbols that match the query.

        As an example:

        ```python
        from financetoolkit import Discovery

        discovery = Discovery(api_key="FINANCIAL_MODELING_PREP_KEY")

        discovery.get_stock_screener(
            market_cap_higher=1000000,
            market_cap_lower=200000000000,
            price_higher=100,
            price_lower=200,
            beta_higher=1,
            beta_lower=1.5,
            volume_higher=100000,
            volume_lower=2000000,
            dividend_higher=1,
            dividend_lower=2,
            is_etf=False
        )
        ```

        Which returns:

        | Symbol   | Name              |   Market Cap | Sector            | Industry               |   Beta |   Price |   Dividend |   Volume | Exchange                | Exchange Code   | Country   |
        |:---------|:------------------|-------------:|:------------------|:-----------------------|-------:|--------:|-----------:|---------:|:------------------------|:----------------|:----------|
        | NKE      | NIKE, Inc.        | 163403295604 | Consumer Cyclical | Footwear & Accessories |  1.079 | 107.36  |       1.48 |  1045865 | New York Stock Exchange | NYSE            | US        |
        | SAF.PA   | Safran SA         |  66234006559 | Industrials       | Aerospace & Defense    |  1.339 | 160.16  |       1.35 |   119394 | Paris                   | EURONEXT        | FR        |
        | ROST     | Ross Stores, Inc. |  46724188589 | Consumer Cyclical | Apparel Retail         |  1.026 | 138.785 |       1.34 |   169879 | NASDAQ Global Select    | NASDAQ          | US        |
        | HES      | Hess Corporation  |  44694706090 | Energy            | Oil & Gas E&P          |  1.464 | 145.51  |       1.75 |   123147 | New York Stock Exchange | NYSE            | US        |

        """
        stock_screener = discovery_model.get_stock_screener(
            api_key=self._api_key,
            market_cap_higher=market_cap_higher,
            market_cap_lower=market_cap_lower,
            price_higher=price_higher,
            price_lower=price_lower,
            beta_higher=beta_higher,
            beta_lower=beta_lower,
            volume_higher=volume_higher,
            volume_lower=volume_lower,
            dividend_higher=dividend_higher,
            dividend_lower=dividend_lower,
            is_etf=is_etf,
        )

        return stock_screener

    def get_stock_list(self) -> pd.DataFrame:
        """
        The stock list function returns a complete list of all the symbols that can be used
        in the FinanceToolkit. These are over 60.000 symbols.

        Returns:
            pd.DataFrame: A dataframe with all the symbols in the toolkit.

        As an example:

        ```python
        from financetoolkit import Discovery

        discovery = Discovery(api_key="FINANCIAL_MODELING_PREP_KEY")

        stock_list = discovery.get_stock_list()

        # The total list equals over 60.000 rows
        stock_list.iloc[38000:38010]
        ```

        Which returns:

        | Symbol      | Name                         |   Price | Exchange                        | Exchange Code   |
        |:------------|:-----------------------------|--------:|:--------------------------------|:----------------|
        | LEO.V       | Lion Copper and Gold Corp.   |   0.09  | Toronto Stock Exchange Ventures | TSX             |
        | LEOF.TA     | Lewinsky-Ofer Ltd.           | 263.1   | Tel Aviv                        | TLV             |
        | LEON        | Leone Asset Management, Inc. |   0.066 | Other OTC                       | OTC             |
        | LEON.SW     | Leonteq AG                   |  34.35  | Swiss Exchange                  | SIX             |
        | LER.AX      | Leaf Resources Limited       |   0.014 | Australian Securities Exchange  | ASX             |
        | LERTHAI.BO  | LERTHAI FINANCE LIMITED      | 265     | Bombay Stock Exchange           | BSE             |
        | LES.WA      | Less S.A.                    |   0.22  | Warsaw Stock Exchange           | WSE             |
        | LESAF       | Le Saunda Holdings Limited   |   0.071 | Other OTC                       | PNK             |
        | LESHAIND.BO | Lesha Industries Limited     |   4.68  | Bombay Stock Exchange           | BSE             |
        | LESL        | Leslie's, Inc.               |   6.91  | NASDAQ Global Select            | NASDAQ          |
        """

        stock_list = discovery_model.get_stock_list(api_key=self._api_key)

        return stock_list

    def get_stock_quotes(self) -> pd.DataFrame:
        """
        Returns the real time stock prices for each company. This includes the bid and ask size,
        the volume, the bid and ask price, the last sales price and the last sales size.

        Returns:
            pd.DataFrame: A dataframe with quotes for each company.

        As an example:

        ```python
        from financetoolkit import Discovery

        discovery = Discovery(api_key="FINANCIAL_MODELING_PREP_KEY")

        stock_quotes = discovery.get_stock_quotes()

        stock_quotes.iloc[3000:3010]
        ```

        Which returns:

        | Symbol   |  Bid Size |   Ask Price |           Volume |   Ask Size |   Bid Price |   Last Sale Price |   Last Sale Size |   Last Sale Time |
        |:---------|----------:|------------:|-----------------:|-----------:|------------:|------------------:|-----------------:|-----------------:|
        | EIPX     |         0 |        0    |  59676           |          0 |        0    |           21.28   |                0 |      1.7039e+12  |
        | EIRL     |         2 |       64.67 |   5455           |          2 |       57.7  |           61.1316 |                0 |      1.7039e+12  |
        | EIS      |        10 |       61.71 |  15886           |          2 |       56.2  |           58.1909 |                0 |      1.7039e+12  |
        | EIX      |         1 |       75.7  |      1.41398e+06 |          1 |       50.1  |           71.49   |                0 |      1.70389e+12 |
        | EJAN     |         1 |       31.42 | 252595           |          1 |       28.1  |           28.67   |                0 |      1.7039e+12  |
        | EJH      |         6 |        3.83 |      0           |          8 |        3.82 |            3.82   |              100 |      1.7042e+12  |
        | EJUL     |         2 |       27.97 |  10226           |          2 |       23.16 |           23.63   |                0 |      1.7039e+12  |
        | EKG      |         4 |       20    |   1197           |          1 |        6.38 |           15.9357 |                0 |      1.70388e+12 |
        | EKSO     |         3 |        2.54 |      0           |          5 |        2.31 |            2.31   |              100 |      1.7042e+12  |
        | EL       |         1 |      143.9  |      0           |          1 |      142.5  |          143      |              100 |      1.7042e+12  |
        """

        stock_quotes = discovery_model.get_stock_quotes(api_key=self._api_key)

        return stock_quotes

    def get_stock_shares_float(self) -> pd.DataFrame:
        """
        Returns the shares float for each company. The shares float is the number of shares
        available for trading for each company. It also includes the number of shares
        outstanding and the date.

        Returns:
            pd.DataFrame: A dataframe with the shares float for each company.

        As an example:

        ```python
        from financetoolkit import Discovery

        discovery = Discovery(api_key="FINANCIAL_MODELING_PREP_KEY")

        shares_float = discovery.get_stock_shares_float()

        shares_float.iloc[50000:50010]
        ```

        Which returns:

        | Symbol   | Date                |   Free Float |   Float Shares |   Outstanding Shares |
        |:---------|:--------------------|-------------:|---------------:|---------------------:|
        | OPY.AX   | NaT                 |     51.4746  |      119853548 |          2.3284e+08  |
        | OPYGY    | NaT                 |      4.49504 |       60892047 |          1.35465e+09 |
        | OQAL     | 2024-01-01 13:12:23 |      0       |              0 |     226543           |
        | OQLGF    | 2023-12-31 21:48:07 |      0.6765  |        1150607 |          1.70082e+08 |
        | OR       | 2024-01-02 05:18:03 |     99.3281  |      183921869 |          1.85166e+08 |
        | OR-R.BK  | 2024-01-01 05:29:30 |     23.153   |     2778360000 |          1.2e+10     |
        | OR.BK    | 2024-01-02 03:52:39 |     22.7847  |     2734164000 |          1.2e+10     |
        | OR.PA    | 2024-01-02 07:57:35 |     45.2727  |      242084445 |          5.34725e+08 |
        | OR.SW    | 2023-12-31 13:38:10 |     45.2727  |      355743960 |          7.8578e+08  |
        | OR.TO    | 2023-12-31 17:56:33 |     99.3317  |      183928535 |          1.85166e+08 |

        """

        stock_shares_float = discovery_model.get_stock_shares_float(
            api_key=self._api_key
        )

        return stock_shares_float

    def get_sectors_performance(self) -> pd.DataFrame:
        """
        Returns the sectors performance for each sector. This features the sector
        performance over the last months.

        Returns:
            pd.DataFrame: A dataframe with the sectors performance for each sector.

        As an example:

        ```python
        from financetoolkit import Discovery

        discovery = Discovery(api_key="FINANCIAL_MODELING_PREP_KEY")

        sectors_performance = discovery.get_sectors_performance()

        sectors_performance.tail()
        ```

        Which returns:

        | Date       |   Utilities |   Basic Materials |   Communication Services |   Consumer Cyclical |   Consumer Defensive |   Energy |   Financial Services |   Healthcare |   Industrials |   Real Estate |   Technology |
        |:-----------|------------:|------------------:|-------------------------:|--------------------:|---------------------:|---------:|---------------------:|-------------:|--------------:|--------------:|-------------:|
        | 2023-12-27 |     0.13511 |           0.40986 |                 -0.23963 |             0.10358 |              0.48048 | -0.27499 |              0.30153 |      0.75715 |       0.30234 |       0.35946 |      0.02372 |
        | 2023-12-28 |     0.80513 |          -0.45131 |                 -0.15858 |            -0.45874 |              0.03828 | -0.81641 |              0.02954 |     -0.01345 |       0.22808 |       0.59612 |     -0.15283 |
        | 2023-12-29 |    -0.01347 |          -0.14525 |                 -0.15072 |            -0.58879 |              0.18141 | -0.42463 |             -0.34718 |     -0.082   |      -0.2181  |      -0.52222 |     -0.57062 |
        | 2024-01-01 |    -0.01347 |          -0.14536 |                 -0.15074 |            -0.58877 |              0.18141 | -0.41917 |             -0.34753 |     -0.08193 |      -0.21821 |      -0.52216 |     -0.5708  |
        | 2024-01-02 |    -0.01347 |          -0.14536 |                 -0.15074 |            -0.58877 |              0.18141 | -0.41917 |             -0.34779 |     -0.08193 |      -0.21823 |      -0.52281 |     -0.57073 |
        """
        sectors_performance = discovery_model.get_sectors_performance(
            api_key=self._api_key
        )

        return sectors_performance

    def get_biggest_gainers(self) -> pd.DataFrame:
        """
        Returns the biggest gainers for the day. This includes the symbol, the name,
        the price, the change and the change percentage.

        Returns:
            pd.DataFrame: A dataframe with the biggest gainers for the day.

        As an example:

        ```python
        from financetoolkit import Discovery

        discovery = Discovery(api_key="FINANCIAL_MODELING_PREP_KEY")

        biggest_gainers = discovery.get_biggest_gainers()

        biggest_gainers.head(10)
        ```

        Which returns:

        | Symbol   | Name                                                   |   Change |   Price |   Change % |
        |:---------|:-------------------------------------------------------|---------:|--------:|-----------:|
        | AAME     | Atlantic American Corporation                          |   0.3001 |  2.4501 |    13.9581 |
        | ADAP     | Adaptimmune Therapeutics plc                           |   0.1029 |  0.793  |    14.9109 |
        | ADTX     | Aditxt, Inc.                                           |   1.81   |  6.63   |    37.5519 |
        | AFMD     | Affimed N.V.                                           |   0.0861 |  0.625  |    15.977  |
        | AIH      | Aesthetic Medical International Holdings Group Limited |   0.1016 |  0.6896 |    17.2789 |
        | ANTE     | AirNet Technology Inc.                                 |   0.1229 |  0.8299 |    17.3833 |
        | APRE     | Aprea Therapeutics, Inc.                               |   1.04   |  4.7    |    28.4153 |
        | ASTR     | Astra Space, Inc.                                      |   0.55   |  2.28   |    31.7919 |
        | BHG      | Bright Health Group, Inc.                              |   2.37   |  7.63   |    45.057  |
        | BROG     | Brooge Energy Limited                                  |   0.73   |  3.68   |    24.7458 |
        """
        biggest_gainers = discovery_model.get_biggest_gainers(api_key=self._api_key)

        return biggest_gainers

    def get_biggest_losers(self) -> pd.DataFrame:
        """
        Returns the biggest losers for the day. This includes the symbol, the name,
        the price, the change and the change percentage.

        Returns:
            pd.DataFrame: A dataframe with the biggest losers for the day.

        As an example:

        ```python
        from financetoolkit import Discovery

        discovery = Discovery(api_key="FINANCIAL_MODELING_PREP_KEY")

        biggest_losers = discovery.get_biggest_losers()

        biggest_losers.head(10)
        ```

        Which returns:

        | Symbol   | Name                                       |   Change |   Price |   Change % |
        |:---------|:-------------------------------------------|---------:|--------:|-----------:|
        | AGAE     | Allied Gaming & Entertainment Inc.         |  -0.2    |  1.06   |   -15.873  |
        | AVTX     | Avalo Therapeutics, Inc.                   |  -2.7339 |  9.1    |   -23.1023 |
        | BAYAR    | Bayview Acquisition Corp Right             |  -0.03   |  0.12   |   -20      |
        | BBLG     | Bone Biologics Corporation                 |  -1.48   |  4.52   |   -24.6667 |
        | BKYI     | BIO-key International, Inc.                |  -0.6    |  3      |   -16.6667 |
        | BREA     | Brera Holdings PLC Class B Ordinary Shares |  -0.2064 |  0.6112 |   -25.2446 |
        | BTBT     | Bit Digital, Inc.                          |  -0.86   |  4.23   |   -16.8959 |
        | BTCS     | BTCS Inc.                                  |  -0.69   |  1.63   |   -29.7414 |
        | BTDR     | Bitdeer Technologies Group                 |  -3.36   |  9.86   |   -25.416  |
        | BYN      | Banyan Acquisition Corporation             |  -2.035  | 10.9    |   -15.7325 |
        """

        biggest_losers = discovery_model.get_biggest_losers(api_key=self._api_key)

        return biggest_losers

    def get_most_active_stocks(self) -> pd.DataFrame:
        """
        Returns the most active stocks for the day. This includes the symbol, the name,
        the price, the change and the change percentage.

        Returns:
            pd.DataFrame: A dataframe with the most active stocks for the day.

        As an example:

        ```python
        from financetoolkit import Discovery

        discovery = Discovery(api_key="FINANCIAL_MODELING_PREP_KEY")

        most_active_stocks = discovery.get_most_active_stocks()

        most_active_stocks.head(10)
        ```

        Which returns:

        | Symbol   | Name                           |   Change |   Price |   Change % |
        |:---------|:-------------------------------|---------:|--------:|-----------:|
        | AAPL     | Apple Inc.                     |    -1.05 |  192.53 |    -0.5424 |
        | ADTX     | Aditxt, Inc.                   |     1.81 |    6.63 |    37.5519 |
        | AMD      | Advanced Micro Devices, Inc.   |    -1.35 |  147.41 |    -0.9075 |
        | AMZN     | Amazon.com, Inc.               |    -1.44 |  151.94 |    -0.9388 |
        | BAC      | Bank of America Corporation    |    -0.21 |   33.67 |    -0.6198 |
        | BITF     | Bitfarms Ltd.                  |    -0.41 |    2.91 |   -12.3494 |
        | BITO     | ProShares Bitcoin Strategy ETF |    -0.33 |   20.49 |    -1.585  |
        | CAN      | Canaan Inc.                    |    -0.5  |    2.31 |   -17.7936 |
        | CLSK     | CleanSpark, Inc.               |    -2.08 |   11.03 |   -15.8657 |
        | DISH     | DISH Network Corporation       |     0.11 |    5.77 |     1.9435 |
        """

        most_active_stocks = discovery_model.get_most_active_stocks(
            api_key=self._api_key
        )

        return most_active_stocks

    def get_delisted_stocks(self) -> pd.DataFrame:
        """
        The delisted stocks function returns a complete list of all delisted stocks including
        the IPO and delisted date.

        Returns:
            pd.DataFrame: A dataframe with all the delisted stocks.

        As an example:

        ```python
        from financetoolkit import Discovery

        discovery = Discovery(api_key="FINANCIAL_MODELING_PREP_KEY")

        delisted_stocks = discovery.get_delisted_stocks()

        delisted_stocks.head(10)
        ```

        Which returns:

        | Symbol   | Name                                         | Exchange   | IPO Date   | Delisted Date   |
        |:---------|:---------------------------------------------|:-----------|:-----------|:----------------|
        | AAIC     | Arlington Asset Investment Corp.             | NYSE       | 1997-12-23 | 2023-12-14      |
        | ABCM     | Abcam plc                                    | NASDAQ     | 2010-12-03 | 2023-12-12      |
        | ADZ      | DB Agriculture Short ETN                     | AMEX       | 2008-04-16 | 2023-10-27      |
        | AENZ     | Aenza S.A.A.                                 | NYSE       | 2013-07-24 | 2023-12-08      |
        | AKUMQ    | Akumin Inc                                   | NASDAQ     | 2018-03-08 | 2023-10-25      |
        | ALTMW    | Kinetik Holdings Inc - Warrants (09/11/2023) | NASDAQ     | 2017-05-01 | 2023-11-07      |
        | ARCE     | Arco Platform Limited                        | NASDAQ     | 2018-09-26 | 2023-12-07      |
        | ARTEW    | Artemis Strategic Investment Corporation     | NASDAQ     | 2021-11-22 | 2023-11-03      |
        | ASPAU    | Abri SPAC I, Inc.                            | NASDAQ     | 2021-08-10 | 2023-11-02      |
        | AVID     | Avid Technology, Inc.                        | NASDAQ     | 1993-03-12 | 2023-11-07      |
        """
        delisted_stocks = discovery_model.get_delisted_stocks(api_key=self._api_key)

        return delisted_stocks

    def get_crypto_list(self) -> pd.DataFrame:
        """
        The crypto list function returns a complete list of all crypto symbols that can be
        used in the FinanceToolkit. These are over 4.000 symbols.

        Returns:
            pd.DataFrame: A dataframe with all the symbols in the toolkit.

        As an example:

        ```python
        from financetoolkit import Discovery

        discovery = Discovery(api_key="FINANCIAL_MODELING_PREP_KEY")

        crypto_list = discovery.get_crypto_list()

        crypto_list.head(10)
        ```

        Which returns:

        | Symbol       | Name                                 | Currency   | Exchange   |
        |:-------------|:-------------------------------------|:-----------|:-----------|
        | .ALPHAUSD    | .Alpha USD                           | USD        | CCC        |
        | 00USD        | 00 Token USD                         | USD        | CCC        |
        | 0NEUSD       | Stone USD                            | USD        | CCC        |
        | 0X0USD       | 0x0.ai USD                           | USD        | CCC        |
        | 0X1USD       | 0x1.tools: AI Multi-tool Plaform USD | USD        | CCC        |
        | 0XAUSD       | 0xApe USD                            | USD        | CCC        |
        | 0XBTCUSD     | 0xBitcoin USD                        | USD        | CCC        |
        | 0XENCRYPTUSD | Encryption AI USD                    | USD        | CCC        |
        | 0XGASUSD     | 0xGasless USD                        | USD        | CCC        |
        | 0XMRUSD      | 0xMonero USD                         | USD        | CCC        |
        """
        crypto_list = discovery_model.get_crypto_list(api_key=self._api_key)

        return crypto_list

    def get_crypto_quotes(self) -> pd.DataFrame:
        """
        Returns the quotes for each crypto. This includes the symbol, the name,
        the price, the change, the change percentage, day low, day high, year high,
        year low, market cap, 50 day average, 200 day average, volume, average volume,
        open, previous close, EPS, PE, earnings announcement, shares outstanding and
        the timestamp.

        Returns:
            pd.DataFrame: A dataframe with the quotes for each crypto.

        As an example:

        ```python
        from financetoolkit import Discovery

        discovery = Discovery(api_key="FINANCIAL_MODELING_PREP_KEY")

        crypto_quotes = discovery.get_crypto_quotes()

        crypto_quotes.head(10)
        ```

        Which returns:

        | Symbol       | Name                                 |        Price |   Change % |       Change |      Day Low |    Day High |   Year High |     Year Low |       Market Cap |   50 Day Avg |   200 Day Avg |      Volume |       Avg Volume |        Open |   Previous Close |   EPS |   PE |   Earnings Announcement |   Shares Outstanding | Timestamp           |
        |:-------------|:-------------------------------------|-------------:|-----------:|-------------:|-------------:|------------:|------------:|-------------:|-----------------:|-------------:|--------------:|------------:|-----------------:|------------:|-----------------:|------:|-----:|------------------------:|---------------------:|:--------------------|
        | .ALPHAUSD    | .Alpha USD                           | 21.4023      |    0       |  0           | 21.3991      | 21.4023     |  193.252    | 21.4023      |      0           | 23.7774      |  51.0497      |     30      |    162           | 21.4023     |      21.4023     |   nan |  nan |                     nan |        nan           | 2022-10-10 23:28:00 |
        | 00USD        | 00 Token USD                         |  0.082484    |    0.67363 |  0.00055192  |  0.0808863   |  0.0857288  |    0.28559  |  0.062939    |      0           |  0.0853295   |   0.0824169   | 210396      | 235403           |  0.0819321  |       0.0819321  |   nan |  nan |                     nan |          0           | 2024-01-02 14:05:40 |
        | 0NEUSD       | Stone USD                            |  7.39e-10    |   -1.70872 | -1.3e-11     |  7.37e-10    |  7.79e-10   |    7.76e-10 |  7.52e-10    |      0           |  0           |   0           |   1110.14   |    nan           |  7.52e-10   |       7.52e-10   |   nan |  nan |                     nan |          0           | 2024-01-02 14:05:12 |
        | 0X0USD       | 0x0.ai USD                           |  0.15383     |    4.3101  |  0.00635643  |  0.14748     |  0.1551     |    0.17925  |  0.000275    |      1.33615e+08 |  0.12582     |   0.0734378   | 805257      |      1.17131e+06 |  0.14748    |       0.14748    |   nan |  nan |                     nan |          8.68563e+08 | 2024-01-02 14:05:13 |
        | 0X1USD       | 0x1.tools: AI Multi-tool Plaform USD |  0.00596268  |    2.65558 |  0.000154248 |  0.00580843  |  0.00608836 |    0.48504  |  0.005089    |      0           |  0.00587516  |   0.0448096   |     42.9976 |    216           |  0.00580843 |       0.00580843 |   nan |  nan |                     nan |          0           | 2024-01-02 14:06:00 |
        | 0XAUSD       | 0xApe USD                            |  9.86177e-06 |  -99.9921  | -0.12519     |  9.86177e-06 |  0.12527    |    0.12527  |  9.86177e-06 |      0           |  1.08846e-05 |   1.08846e-05 |    197      |    nan           |  0.1252     |       0.1252     |   nan |  nan |                     nan |        nan           | 2023-06-24 18:30:00 |
        | 0XBTCUSD     | 0xBitcoin USD                        |  0.097478    |    0.6003  |  0.00058167  |  0.0944255   |  0.10393    |    4.13419  |  0.03222     | 946195           |  0.17478     |   0.39561     |    344.45   |  97856           |  0.0968963  |       0.0968963  |   nan |  nan |                     nan |          9.70675e+06 | 2024-01-02 14:05:24 |
        | 0XENCRYPTUSD | Encryption AI USD                    |  0.0213021   |    0       |  0           |  0.0213021   |  0.0213021  |   15.4064   |  0.020326    |      0           |  1.55438     |   3.26515     |      2      | 202458           |  0.0213021  |       0.0213021  |   nan |  nan |                     nan |        nan           | 2023-07-26 18:30:00 |
        | 0XGASUSD     | 0xGasless USD                        |  0.11228     |   12.1894  |  0.0121997   |  0.10008     |  0.11228    |    0.19216  |  3.7e-05     |      0           |  0.038569    |   0.0143848   |   8700      |   9628           |  0.10008    |       0.10008    |   nan |  nan |                     nan |          0           | 2024-01-02 14:06:00 |
        | 0XMRUSD      | 0xMonero USD                         |  0.0497938   |  -38.9213  | -0.0317302   |  0.0496646   |  2.79013    |    0.18734  |  0.0418889   |      0           |  0.13616     |   0.11633     |    347.276  |     11           |  0.081524   |       0.081524   |   nan |  nan |                     nan |        nan           | 2024-01-02 14:05:07 |
        """
        crypto_quotes = discovery_model.get_crypto_quotes(api_key=self._api_key)

        return crypto_quotes

    def get_forex_list(self) -> pd.DataFrame:
        """
        The forex list function returns a complete list of all forex symbols that can be
        used in the FinanceToolkit. These are over 1.000 symbols.

        Returns:
            pd.DataFrame: A dataframe with the forex symbols.

        As an example:

        ```python
        from financetoolkit import Discovery

        discovery = Discovery(api_key="FINANCIAL_MODELING_PREP_KEY")

        forex_list = discovery.get_forex_list()

        forex_list.head(10)
        ```

        Which returns:

        | Symbol   | Name    | Currency   | Exchange   |
        |:---------|:--------|:-----------|:-----------|
        | AEDAUD   | AED/AUD | AUD        | CCY        |
        | AEDBHD   | AED/BHD | BHD        | CCY        |
        | AEDCAD   | AED/CAD | CAD        | CCY        |
        | AEDCHF   | AED/CHF | CHF        | CCY        |
        | AEDDKK   | AED/DKK | DKK        | CCY        |
        | AEDEUR   | AED/EUR | EUR        | CCY        |
        | AEDGBP   | AED/GBP | GBP        | CCY        |
        | AEDILS   | AED/ILS | ILS        | CCY        |
        | AEDINR   | AED/INR | INR        | CCY        |
        | AEDJOD   | AED/JOD | JOD        | CCY        |
        """
        forex_list = discovery_model.get_forex_list(api_key=self._api_key)

        return forex_list

    def get_forex_quotes(self) -> pd.DataFrame:
        """
        Returns the quotes for each forex. This includes the symbol, the name,
        the price, the change, the change percentage, day low, day high, year high,
        year low, market cap, 50 day average, 200 day average, volume, average volume,
        open, previous close, EPS, PE, earnings announcement, shares outstanding and
        the timestamp.

        Returns:
            pd.DataFrame: A dataframe with quotes for each forex.

        As an example:

        ```python
        from financetoolkit import Discovery

        discovery = Discovery(api_key="FINANCIAL_MODELING_PREP_KEY")

        forex_quotes = discovery.get_forex_quotes()

        forex_quotes.head(10)
        ```

        Which returns:

        | Symbol   | Name    |    Price |     Change % |       Change |   Day Low |   Day High |   Year High |   Year Low |   50 Day Avg |   200 Day Avg |   Volume |   Avg Volume |     Open |   Previous Close | Timestamp           |
        |:---------|:--------|---------:|-------------:|-------------:|----------:|-----------:|------------:|-----------:|-------------:|--------------:|---------:|-------------:|---------:|-----------------:|:--------------------|
        | AEDAUD   | AED/AUD |  0.40089 |    0.40826   |   0.00163    |   0.39766 |    0.40118 |     0.43341 |  0.38041   |      0.41514 |       0.41372 |       11 |     nan      |  0.39921 |          0.39926 | 2024-01-02 14:02:15 |
        | AEDBHD   | AED/BHD |  0.10262 |    0.0608637 |   6.2422e-05 |   0.10244 |    0.10266 |     0.10323 |  0.0991399 |      0.10264 |       0.10241 |       37 |      48.006  |  0.10256 |          0       | 2024-01-02 13:46:14 |
        | AEDCAD   | AED/CAD |  0.36177 |    0.43587   |   0.00157    |   0.35996 |    0.36295 |     0.37817 |  0.35657   |      0.3701  |       0.36716 |       14 |     nan      |  0.36002 |          0.3602  | 2024-01-02 14:02:15 |
        | AEDCHF   | AED/CHF |  0.23062 |    0.8704    |   0.00199    |   0.22847 |    0.23099 |     0.25693 |  0.2278    |      0.23976 |       0.24231 |      nan |     nan      |  0.22847 |          0.22863 | 2024-01-02 14:02:15 |
        | AEDDKK   | AED/DKK |  1.84023 |   84.023     |   0.84023    |   1.83775 |    1.84081 |     1.94068 |  1.78424   |      1.86572 |       1.87037 |       16 |      49.5329 |  1.83874 |          1       | 2024-01-02 09:37:59 |
        | AEDEUR   | AED/EUR |  0.2486  |    0.81044   |   0.00199857 |   0.24636 |    0.24871 |     0.265   |  0.2417    |      0.25271 |       0.25197 |       38 |     nan      |  0.24668 |          0.2466  | 2024-01-02 14:02:15 |
        | AEDGBP   | AED/GBP |  0.21499 |    0.75924   |   0.00162    |   0.21298 |    0.2157  |     0.23039 |  0.2073    |      0.21802 |       0.21732 |       14 |     nan      |  0.2133  |          0.21337 | 2024-01-02 14:02:15 |
        | AEDILS   | AED/ILS |  0.98746 | -100         | nan          |   0.98385 |    0.99536 |     1.1108  |  0.97828   |      1.01241 |       1.03478 |      923 |     549.264  |  0.98761 |        nan       | 2024-01-02 14:05:06 |
        | AEDINR   | AED/INR | 22.7025  |    0.14076   |   0.0319101  |  22.625   |   22.72    |    22.72    | 20.1966    |     19.8653  |      20.1966  |       14 |     nan      | 22.7082  |         22.6706  | 2024-01-02 14:02:15 |
        | AEDJOD   | AED/JOD |  0.19335 |   -3.32563   |  -0.00665126 |   0.19315 |    0.19364 |     0.19412 |  0.19185   |      0.19314 |       0.19315 |       38 |      18.8451 |  0.19331 |          0.2     | 2024-01-02 13:51:18 |

        """
        forex_quotes = discovery_model.get_forex_quotes(api_key=self._api_key)

        return forex_quotes

    def get_commodity_list(self) -> pd.DataFrame:
        """
        The commodity list function returns a complete list of all commodity symbols that can be
        used in the FinanceToolkit.

        Returns:
            pd.DataFrame: A dataframe with all the commodities available.

        As an example:

        ```python
        from financetoolkit import Discovery

        discovery = Discovery(api_key="FINANCIAL_MODELING_PREP_KEY")

        commodity_list = discovery.get_commodity_list()

        commodity_list.head(10)
        ```

        Which returns:

        | Symbol   | Name                   | Currency   | Exchange   |
        |:---------|:-----------------------|:-----------|:-----------|
        | ALIUSD   | Aluminum Futures       | USD        | COMEX      |
        | BZUSD    | Brent Crude Oil        | USD        | ICE        |
        | CCUSD    | Cocoa                  | USD        | ICE        |
        | CLUSD    | Crude Oil              | USD        | CME        |
        | CTUSX    | Cotton                 | USX        | ICE        |
        | DCUSD    | Class III Milk Futures | USD        | CME        |
        | DXUSD    | US Dollar              | USD        | ICE        |
        | ESUSD    | E-Mini S&P 500         | USD        | CME        |
        | GCUSD    | Gold Futures           | USD        | CME        |
        | GFUSX    | Feeder Cattle Futures  | USX        | CME        |
        """
        commodity_list = discovery_model.get_commodity_list(api_key=self._api_key)

        return commodity_list

    def get_commodity_quotes(self) -> pd.DataFrame:
        """
        Returns the quotes for each commodity. This includes the symbol, the name,
        the price, the change, the change percentage, day low, day high, year high,
        year low, market cap, 50 day average, 200 day average, volume, average volume,
        open, previous close, EPS, PE, earnings announcement, shares outstanding and
        the timestamp.

        Returns:
            pd.DataFrame: A dataframe with the quotes for each commodity.

        As an example:

        ```python
        from financetoolkit import Discovery

        discovery = Discovery(api_key="FINANCIAL_MODELING_PREP_KEY")

        commodity_quotes = discovery.get_commodity_quotes()

        commodity_quotes.head(10)
        ```

        Which returns:

        | Symbol   | Name                   |    Price |   Change % |   Change |   Day Low |   Day High |   Year High |   Year Low |   50 Day Avg |   200 Day Avg |   Volume |       Avg Volume |     Open |   Previous Close | Timestamp           |
        |:---------|:-----------------------|---------:|-----------:|---------:|----------:|-----------:|------------:|-----------:|-------------:|--------------:|---------:|-----------------:|---------:|-----------------:|:--------------------|
        | ALIUSD   | Aluminum Futures       | 2347     | -1.12691   |  -26.75  |  2344     |    2383.5  |     2670.75 |    2073.25 |    2200.86   |     2221.04   |     4321 |     22           | 2370.75  |         2373.75  | 2024-01-02 13:54:40 |
        | BZUSD    | Brent Crude Oil        |   78.1   |  1.37591   |    1.06  |    77.21  |      79.06 |       97.63 |      68.2  |      81.291  |       81.9377 |     2285 |  30060           |   77.21  |           77.04  | 2024-01-02 14:10:12 |
        | CCUSD    | Cocoa                  | 4249.5   |  1.27502   |   53.5   |   101.03  |    4274.5  |     4478    |    2507    |    4115.52   |     3483.99   |    18596 |  14509           | 4209     |         4196     | 2024-01-02 14:10:12 |
        | CLUSD    | Crude Oil              |   72.63  |  1.36776   |    0.98  |    71.63  |      73.65 |       95.03 |      63.64 |      76.3836 |       77.7364 |    37720 | 307715           |   71.71  |           71.65  | 2024-01-02 14:10:12 |
        | CTUSX    | Cotton                 |   80.78  | -0.2716    |   -0.22  |     3.87  |      81.75 |       90.75 |      74.77 |      79.8394 |       82.7224 |      960 |  15911           |   80.87  |           81     | 2024-01-02 14:10:00 |
        | DCUSD    | Class III Milk Futures |   16.35  |  1.5528    |    0.25  |    15.43  |      17.16 |       20.49 |      13.75 |      16.6668 |       16.7265 |       51 |    212           |   16.1   |           16.1   | 2024-01-02 13:36:35 |
        | DXUSD    | US Dollar              |  101.862 |  0.82452   |    0.833 |   101.027 |     101.88 |      107.05 |      99.22 |     103.915  |      103.24   |     2999 |  14880           |  101.065 |          101.029 | 2024-01-02 14:10:10 |
        | ESUSD    | E-Mini S&P 500         | 4783     | -0.76763   |  -37     |  4777.75  |    4828    |     4841.5  |    3808.75 |    4527.31   |     4378.91   |    75910 |      1.63378e+06 | 4818     |         4820     | 2024-01-02 14:00:13 |
        | GCUSD    | Gold Futures           | 2075     |  0.15446   |    3.2   |  2071.4   |    2094.7  |     2130.2  |    1808.1  |    2003.86   |     1960.64   |    38456 |   3511           | 2072.7   |         2071.8   | 2024-01-02 14:00:13 |
        | GFUSX    | Feeder Cattle Futures  |  223.125 |  0.0112057 |    0.025 |   222.725 |     224.45 |      257.5  |     177.55 |     226.9    |      230.114  |     4395 |   3915           |  224.4   |          223.1   | 2023-12-29 19:04:57 |
        """
        commodity_quotes = discovery_model.get_commodity_quotes(api_key=self._api_key)

        return commodity_quotes

    def get_etf_list(self) -> pd.DataFrame:
        """
        The etf list function returns a complete list of all etf symbols that can be
        used in the FinanceToolkit.

        Returns:
            pd.DataFrame: A dataframe with all the etf symbols.

        As an example:

        ```python
        from financetoolkit import Discovery

        discovery = Discovery(api_key="FINANCIAL_MODELING_PREP_KEY")

        etf_list = discovery.get_etf_list()

        etf_list.head(10)
        ```

        Which returns:

        | Symbol    | Name                                                                                            |      Price | Exchange              | Exchange Code   |
        |:----------|:------------------------------------------------------------------------------------------------|-----------:|:----------------------|:----------------|
        | 01002T.TW | Cathay No.1 REIT                                                                                |    17.29   | Taiwan                | TAI             |
        | 020Y.L    | iShares IV Public Limited Company - iShares Euro Government Bond 20yr Target Duration UCITS ETF |     3.9522 | London Stock Exchange | LSE             |
        | 069500.KS | KODEX 200                                                                                       | 36390      | KSE                   | KSC             |
        | 069660.KS | KOSEF 200                                                                                       | 36370      | KSE                   | KSC             |
        | 091160.KS | Kodex Semicon                                                                                   | 36840      | KSE                   | KSC             |
        | 091170.KS | Kodex Banks                                                                                     |  6695      | KSE                   | KSC             |
        | 091180.KS | Kodex Autos                                                                                     | 19450      | KSE                   | KSC             |
        | 091220.KS | Mirae Asset TIGER Banks ETF                                                                     |  6845      | KSE                   | KSC             |
        | 091230.KS | Mirae Asset TIGER Semicon ETF                                                                   | 38400      | KSE                   | KSC             |
        | 098560.KS | Mirae Asset TIGER Media & Telecom ETF                                                           |  7335      | KSE                   | KSC             |
        """

        etf_list = discovery_model.get_etf_list(api_key=self._api_key)

        return etf_list

    def get_index_list(self) -> pd.DataFrame:
        """
        The index list function returns a complete list of all etf symbols that can be
        used in the FinanceToolkit.

        Returns:
            pd.DataFrame: A dataframe with all the index symbols.

        As an example:

        ```python
        from financetoolkit import Discovery

        discovery = Discovery(api_key="FINANCIAL_MODELING_PREP_KEY")

        index_list = discovery.get_index_list()

        index_list.head(10)
        ```

        Which returns:

        | Symbol      | Name                          | Currency   | Exchange               |
        |:------------|:------------------------------|:-----------|:-----------------------|
        | 000001.SS   | SSE Composite Index           | CNY        | Shanghai               |
        | 399967.SZ   | CSI NATIONAL DEFENSE          | CNY        | Shenzhen               |
        | 512.HK      | CES CHINA HK MAINLAND INDEX   | HKD        | HKSE                   |
        | DX-Y.NYB    | US Dollar/USDX - Index - Cash | USD        | ICE Futures            |
        | FTSEMIB.MI  | FTSE MIB Index                | EUR        | Milan                  |
        | IAR.BA      | MERVAL ARGENTINA              | USD        | Buenos Aires           |
        | IDX30.JK    | IDX30                         | IDR        | Jakarta Stock Exchange |
        | IMOEX.ME    | MOEX Russia Index             | RUB        | MCX                    |
        | ITLMS.MI    | FTSE Italia All-Share Index   | EUR        | Milan                  |
        | KOSPI200.KS | KOSPI 200 Index               | KRW        | KSE                    |
        """
        index_list = discovery_model.get_index_list(api_key=self._api_key)

        return index_list

    def get_index_quotes(self) -> pd.DataFrame:
        """
        Returns the quotes for each index. This includes the symbol, the name,
        the price, the change, the change percentage, day low, day high, year high,
        year low, market cap, 50 day average, 200 day average, volume, average volume,
        open, previous close, EPS, PE, earnings announcement, shares outstanding and
        the timestamp.

        Returns:
            pd.DataFrame: A dataframe with all the symbols in the toolkit.

        As an example:

        ```python
        from financetoolkit import Discovery

        discovery = Discovery(api_key="FINANCIAL_MODELING_PREP_KEY")

        index_quotes = discovery.get_index_quotes()

        index_quotes.head(10)
        ```

        Which returns:

        | Symbol      | Name                          |     Price |   Change % |     Change |   Day Low |   Day High |   Year High |   Year Low |   50 Day Avg |   200 Day Avg |     Volume |   Avg Volume |      Open |   Previous Close |   Timestamp |
        |:------------|:------------------------------|----------:|-----------:|-----------:|----------:|-----------:|------------:|-----------:|-------------:|--------------:|-----------:|-------------:|----------:|-----------------:|------------:|
        | 000001.SS   | SSE Composite Index           |  2962.28  |    -0.4255 |   -12.6587 |  2962.28  |   2976.27  |    3418.95  |   2882.02  |     2999.76  |      3160.83  |  349408228 |       290686 |  2972.78  |         2974.93  |  1704178820 |
        | 399967.SZ   | CSI NATIONAL DEFENSE          |  9891.22  |     0.4875 |    47.9902 |  9834.98  |  10041.4   |   10041.4   |   9834.98  |        0     |         0     | 1115610197 |            0 |  9857.19  |         9843.23  |  1704184147 |
        | 512.HK      | CES CHINA HK MAINLAND INDEX   |  6901.25  |     0      |     0      |  6786.45  |   6912.54  |    6912.54  |   6786.45  |        0     |         0     | 2785244718 |            0 |  6862.61  |          nan     |  1434960128 |
        | DX-Y.NYB    | US Dollar/USDX - Index - Cash |   102.136 |     0.7924 |     0.803  |   101.34  |    102.167 |     107.35  |     99.58  |      104.108 |       103.421 |          0 |            0 |   101.417 |          101.333 |  1704204265 |
        | FTSEMIB.MI  | FTSE MIB Index                | 30396.8   |     0.1488 |    45.1699 | 30326.9   |  30863.6   |   30863.6   |  24111     |    29233.6   |     28164     |          0 |    473923362 | 30519.5   |        30351.6   |  1704203960 |
        | IAR.BA      | MERVAL ARGENTINA              | 33784.6   |     0      | 33784.6    | 33227.6   |  33871.5   |   33871.5   |  33227.6   |        0     |         0     |          0 |            0 | 33227.6   |          nan     |  1576872141 |
        | IDX30.JK    | IDX30                         |   498.424 |     0.6486 |     3.212  |   492.621 |    498.424 |     498.424 |    492.621 |        0     |         0     |          0 |            0 |   493.985 |          495.212 |  1704186018 |
        | IMOEX.ME    | MOEX Russia Index             |  2222.51  |    -0.1859 |    -4.1399 |  2202.52  |   2234.55  |    4292.68  |   1681.55  |     2264.41  |      3183.63  |          0 |            0 |  2225.02  |         2226.65  |  1657295461 |
        | ITLMS.MI    | FTSE Italia All-Share Index   | 32507     |     0.0859 |    27.9004 | 32434.3   |  32999.1   |   32999.1   |  23017.3   |    22902.7   |     23017.3   |          0 |            0 | 32651.2   |        32479.1   |  1704203955 |
        | KOSPI200.KS | KOSPI 200 Index               |   360.55  |     0.7151 |     2.56   |   355.96  |    361.53  |     361.53  |    355.96  |        0     |         0     |     106709 |            0 |   356.43  |          357.99  |  1704186335 |
        """
        index_quotes = discovery_model.get_index_quotes(api_key=self._api_key)

        return index_quotes
