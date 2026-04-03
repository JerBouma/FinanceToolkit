# Data Source Suggestion: FinancialReports.eu for Filing Access

## Overview

[FinancialReports.eu](https://financialreports.eu) provides API access to **14M+ filings** from data sources across 30+ countries. It complements FinanceToolkit's financial analysis capabilities by providing access to the **source filing documents** behind the computed metrics.

## Why This Fits FinanceToolkit

FinanceToolkit computes financial ratios, models, and indicators. FinancialReports.eu adds:

- **Source document access** — verify computed metrics against original annual reports and financial statements
- **Global company coverage** — 33,000+ companies with ISIN and LEI identifiers, matching FinanceToolkit's international scope
- **Markdown endpoint** — extract text from filings for programmatic analysis alongside computed ratios
- **11 standardized categories** — Financial Reporting, ESG, M&A, Debt/Equity, and more
- **30+ countries** including UK, EU, Japan, and more

## Integration Approaches

### 1. Filing Data Provider

Add as an optional data provider — users supply their own API key:

```python
import requests

headers = {"X-API-Key": "your-api-key"}

# Look up a company by ISIN
resp = requests.get("https://api.financialreports.eu/companies/",
    headers=headers,
    params={"isin": "NL0010273215", "page_size": 5}  # ASML
)

# Fetch financial reporting filings
resp = requests.get("https://api.financialreports.eu/filings/",
    headers=headers,
    params={
        "company_isin": "NL0010273215",
        "categories": "2",  # Financial Reporting
        "page_size": 10
    }
)

# Get filing as Markdown
filing_id = resp.json()["results"][0]["id"]
content = requests.get(
    f"https://api.financialreports.eu/filings/{filing_id}/markdown/",
    headers=headers
).text
```

### 2. MCP Server

FinancialReports.eu offers an [MCP server](https://financialreports.eu) compatible with Claude.ai and other AI platforms for interactive filing analysis.

### 3. Python SDK

```bash
pip install financial-reports-generated-client
```

```python
from financial_reports_client import Client
from financial_reports_client.api.companies import companies_list
from financial_reports_client.api.filings import filings_list

client = Client(base_url="https://api.financialreports.eu")
client = client.with_headers({"X-API-Key": "your-api-key"})

companies = companies_list.sync(client=client, isin="NL0010273215")
filings = filings_list.sync(client=client, company_isin="NL0010273215", categories="2")
```

## API Details

| Property | Value |
|---|---|
| **Base URL** | `https://api.financialreports.eu` |
| **API Docs** | [docs.financialreports.eu](https://docs.financialreports.eu/) |
| **Authentication** | API key via `X-API-Key` header |
| **Python SDK** | `pip install financial-reports-generated-client` |
| **Rate Limiting** | Burst limit + monthly quota |
| **Companies** | 33,230+ |
| **Total Filings** | 14,135,359+ |
| **Coverage** | 30+ countries |

## Complementary Value

| FinanceToolkit (current) | + FinancialReports.eu |
|---|---|
| Computed financial ratios | Source regulatory documents |
| Fundamental analysis models | Original annual report text |
| International company data | Filing documents from 30+ countries |
| Technical indicators | Filing event dates as signals |
| — | ESG disclosures, M&A announcements |
