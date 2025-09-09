"""Discovery Module"""

__docformat__ = "google"

import pandas as pd

from financetoolkit import fmp_model
from financetoolkit.discovery import discovery_model
from financetoolkit.utilities import logger_model
from financetoolkit.utilities.error_model import handle_errors

# pylint: disable=too-many-instance-attributes,too-few-public-methods,too-many-lines,
# pylint: disable=too-many-locals,line-too-long,too-many-public-methods
# ruff: noqa: E501

# Set up logger, this is meant to display useful messages, warnings or errors when
# the Finance Toolkit runs into issues or does something that might not be entirely
# logical at first
logger_model.setup_logger()
logger = logger_model.get_logger()


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

        # This tests the API key to determine the subscription plan. This is relevant for the sleep timer
        # but also for other components of the Toolkit. This prevents wait timers from occurring while
        # it wouldn't result to any other answer than a rate limit error.
        determine_plan = fmp_model.get_financial_data(
            url=f"https://financialmodelingprep.com/stable/income-statement?symbol=AAPL&apikey={api_key}&limit=10",
            sleep_timer=False,
            user_subscription="Free",
        )

        self._fmp_plan = "Premium"

        for option in [
            "PREMIUM QUERY PARAMETER",
            "EXCLUSIVE ENDPOINT",
            "NO DATA",
            "BANDWIDTH LIMIT REACH",
            "INVALID API KEY",
            "LIMIT REACH",
        ]:
            if option in determine_plan:
                self._fmp_plan = "Free"
                break
        else:
            self._fmp_plan = "Premium"

    @handle_errors
    def search_instruments(
        self, query: str | None = None, search_method: str = "name"
    ) -> pd.DataFrame:
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
        if search_method not in ["symbol", "name", "cik", "cusip", "isin"]:
            raise ValueError(
                "Please enter a valid search method. Valid options are: 'symbol', 'name', 'cik', 'cusip', 'isin'. "
            )
        if not query:
            raise ValueError(
                "Please enter a query to search for, e.g. search_instruments(query='META'). "
            )

        symbol_list = discovery_model.get_instruments(
            api_key=self._api_key,
            query=query,
            search_method=search_method,
            user_subscription=self._fmp_plan,
        )

        if symbol_list.empty and len(symbol_list.columns) == 0:
            logger.error(
                f"No results found for the given query ({query}). Please try a different query."
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
            user_subscription=self._fmp_plan,
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

        stock_list = discovery_model.get_stock_list(
            api_key=self._api_key, user_subscription=self._fmp_plan
        )

        return stock_list

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
            api_key=self._api_key, user_subscription=self._fmp_plan
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
            api_key=self._api_key, user_subscription=self._fmp_plan
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
        biggest_gainers = discovery_model.get_biggest_gainers(
            api_key=self._api_key, user_subscription=self._fmp_plan
        )

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

        biggest_losers = discovery_model.get_biggest_losers(
            api_key=self._api_key, user_subscription=self._fmp_plan
        )

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
            api_key=self._api_key, user_subscription=self._fmp_plan
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
        delisted_stocks = discovery_model.get_delisted_stocks(
            api_key=self._api_key, user_subscription=self._fmp_plan
        )

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
        crypto_list = discovery_model.get_crypto_list(
            api_key=self._api_key, user_subscription=self._fmp_plan
        )

        return crypto_list

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
        forex_list = discovery_model.get_forex_list(
            api_key=self._api_key, user_subscription=self._fmp_plan
        )

        return forex_list

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
        commodity_list = discovery_model.get_commodity_list(
            api_key=self._api_key, user_subscription=self._fmp_plan
        )

        return commodity_list

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

        etf_list = discovery_model.get_etf_list(
            api_key=self._api_key, user_subscription=self._fmp_plan
        )

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
        index_list = discovery_model.get_index_list(
            api_key=self._api_key, user_subscription=self._fmp_plan
        )

        return index_list
