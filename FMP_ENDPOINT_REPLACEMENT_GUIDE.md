# FMP API Endpoint Modernization Guide

## Overview
Financial Modeling Prep has systematically deprecated their legacy `/api/v3/` endpoints in favor of new `/stable/` endpoints. The FinanceToolkit library extensively uses these deprecated endpoints, causing API failures (HTTP 403 errors) as of August 31, 2025.

**This document provides a comprehensive mapping of all legacy endpoints to their modern equivalents based on validated FMP documentation analysis.**

## Critical Changes Required

### 1. URL Base Path Migration
- **Legacy Pattern**: `https://financialmodelingprep.com/api/v3/`
- **Modern Pattern**: `https://financialmodelingprep.com/stable/`

### 2. Parameter Structure Changes
- **Legacy**: Path-based parameters (e.g., `/profile/{ticker}`)
- **Modern**: Query-based parameters (e.g., `/profile?symbol={ticker}`)

## Complete Endpoint Replacement Table

| Category | Legacy Endpoint (v3) | Modern Endpoint (stable) | Parameter Changes |
|----------|---------------------|-------------------------|-------------------|
| **Company Profile** | `/api/v3/profile/{ticker}` | `/stable/profile` | `symbol={ticker}` parameter |
| **Financial Statements** | | | |
| Balance Sheet | `/api/v3/balance-sheet-statement/{ticker}` | `/stable/balance-sheet-statement` | `symbol={ticker}` parameter |
| Income Statement | `/api/v3/income-statement/{ticker}` | `/stable/income-statement` | `symbol={ticker}` parameter |
| Cash Flow Statement | `/api/v3/cash-flow-statement/{ticker}` | `/stable/cash-flow-statement` | `symbol={ticker}` parameter |
| **Historical Data** | | | |
| Historical Prices | `/api/v3/historical-price-full/{ticker}` | `/stable/historical-price-full` | `symbol={ticker}` parameter |
| Historical Chart | `/api/v3/historical-chart/{interval}/{ticker}` | `/stable/historical-chart/{interval}` | `symbol={ticker}` parameter |
| Historical Dividends | `/api/v3/historical-price-full/stock_dividend/{ticker}` | `/stable/historical-dividends` | `symbol={ticker}` parameter |
| **Market Data** | | | |
| Real-time Quote | `/api/v3/quote/{ticker}` | `/stable/quote` | `symbol={ticker}` parameter |
| Market Cap | `/api/v3/market-capitalization/{ticker}` | `/stable/market-capitalization` | `symbol={ticker}` parameter |
| **Analysis & Estimates** | | | |
| Analyst Estimates | `/api/v3/analyst-estimates/{ticker}` | `/stable/analyst-estimates` | `symbol={ticker}` parameter |
| Financial Ratios | `/api/v3/ratios/{ticker}` | `/stable/financial-ratios` | `symbol={ticker}` parameter |
| Key Metrics | `/api/v3/key-metrics/{ticker}` | `/stable/key-metrics` | `symbol={ticker}` parameter |
| **Segmentation** | | | |
| Revenue Geographic | `/api/v3/revenue-geographic-segmentation/{ticker}` | `/stable/revenue-geographic-segmentation` | `symbol={ticker}` parameter |
| Revenue Product | `/api/v3/revenue-product-segmentation/{ticker}` | `/stable/revenue-product-segmentation` | `symbol={ticker}` parameter |
| **Discovery** | | | |
| Stock Screener | `/api/v3/stock-screener` | `/stable/stock-screener` | Same parameters |
| Search Companies | `/api/v3/search` | `/stable/search` | Same parameters |
| **Enterprise Data** | | | |
| Enterprise Value | `/api/v3/enterprise-values/{ticker}` | `/stable/enterprise-values` | `symbol={ticker}` parameter |
| Financial Growth | `/api/v3/financial-growth/{ticker}` | `/stable/financial-growth` | `symbol={ticker}` parameter |

## FinanceToolkit Files Requiring Updates

### Primary Files with Legacy Endpoints
1. **`financetoolkit/fmp_model.py`** - Core FMP API functions
2. **`financetoolkit/discovery/discovery_model.py`** - Stock screening functionality
3. **`financetoolkit/fundamentals_model.py`** - Financial statement retrieval

### Specific Code Changes Required

#### 1. Financial Statement URLs (Lines 187-203 in fmp_model.py)
```python
# BEFORE (Legacy)
url = (
    f"https://financialmodelingprep.com/api/v3/{location}"
    f"/{ticker}?period={period}&apikey={api_key}&"
    f"limit={periods_to_fetch}"
)

# AFTER (Modern)
url = (
    f"https://financialmodelingprep.com/stable/{location}"
    f"?symbol={ticker}&period={period}&apikey={api_key}&"
    f"limit={periods_to_fetch}"
)
```

#### 2. Historical Price Data (Lines 307-319 in fmp_model.py)
```python
# BEFORE (Legacy)
historical_data_url = (
    f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?"
    f"apikey={api_key}&from={start_date_string}&to={end_date_string}"
)

# AFTER (Modern)
historical_data_url = (
    f"https://financialmodelingprep.com/stable/historical-price-full"
    f"?symbol={ticker}&apikey={api_key}&from={start_date_string}&to={end_date_string}"
)
```

#### 3. Historical Chart Data (Lines 471-489 in fmp_model.py)
```python
# BEFORE (Legacy)
historical_data_url = (
    f"https://financialmodelingprep.com/api/v3/historical-chart/{interval}/{ticker}?"
    f"from={start_date_string}&to={end_date_string}&apikey={api_key}"
)

# AFTER (Modern)
historical_data_url = (
    f"https://financialmodelingprep.com/stable/historical-chart/{interval}"
    f"?symbol={ticker}&from={start_date_string}&to={end_date_string}&apikey={api_key}"
)
```

#### 4. Company Profile Data (Line 1038 in fmp_model.py)
```python
# BEFORE (Legacy)
url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={api_key}"

# AFTER (Modern)
url = f"https://financialmodelingprep.com/stable/profile?symbol={ticker}&apikey={api_key}"
```

#### 5. Analyst Estimates (Lines 851-864 in fmp_model.py)
```python
# BEFORE (Legacy)
url = (
    "https://financialmodelingprep.com/api/v3/analyst-estimates/"
    f"{ticker}?period={period}&apikey={api_key}"
)

# AFTER (Modern)
url = (
    "https://financialmodelingprep.com/stable/analyst-estimates"
    f"?symbol={ticker}&period={period}&apikey={api_key}"
)
```

#### 6. Revenue Segmentation (Lines 633-650 in fmp_model.py)
```python
# BEFORE (Legacy)
url = (
    f"https://financialmodelingprep.com/api/v3/{location}"
    f"/{ticker}?period={period}&structure=flat&apikey={api_key}"
)

# AFTER (Modern)
url = (
    f"https://financialmodelingprep.com/stable/{location}"
    f"?symbol={ticker}&period={period}&structure=flat&apikey={api_key}"
)
```

#### 7. Stock Screener (discovery_model.py)
```python
# BEFORE (Legacy)
url = "https://financialmodelingprep.com/api/v3/stock-screener"

# AFTER (Modern)  
url = "https://financialmodelingprep.com/stable/stock-screener"
```

## Validation Examples

### Working Modern Endpoints (Confirmed)
- ✅ `https://financialmodelingprep.com/stable/profile?symbol=AAPL&apikey=YOUR_API_KEY`
- ✅ `https://financialmodelingprep.com/stable/income-statement?symbol=AAPL&apikey=YOUR_API_KEY`
- ✅ `https://financialmodelingprep.com/stable/balance-sheet-statement?symbol=AAPL&apikey=YOUR_API_KEY`

### Deprecated Legacy Endpoints (Returns HTTP 403)
- ❌ `https://financialmodelingprep.com/api/v3/profile/AAPL?apikey=YOUR_API_KEY`
- ❌ `https://financialmodelingprep.com/api/v3/income-statement/AAPL?apikey=YOUR_API_KEY`
- ❌ `https://financialmodelingprep.com/api/v3/balance-sheet-statement/AAPL?apikey=YOUR_API_KEY`

## Implementation Steps

### Step 1: Create FinanceToolkit Fork
```bash
git clone https://github.com/JerBouma/FinanceToolkit.git
cd FinanceToolkit
git checkout -b fmp-stable-endpoints
```

### Step 2: Apply Systematic Replacements
Use find-and-replace across the entire codebase:

1. **Base URL Replacement**:
   - Find: `https://financialmodelingprep.com/api/v3/`
   - Replace: `https://financialmodelingprep.com/stable/`

2. **Parameter Structure Updates**:
   - Find: `/{ticker}?`
   - Replace: `?symbol={ticker}&`

### Step 3: Test Critical Endpoints
```python
import requests

api_key = "YOUR_API_KEY"
test_endpoints = [
    f"https://financialmodelingprep.com/stable/profile?symbol=AAPL&apikey={api_key}",
    f"https://financialmodelingprep.com/stable/income-statement?symbol=AAPL&apikey={api_key}",
    f"https://financialmodelingprep.com/stable/balance-sheet-statement?symbol=AAPL&apikey={api_key}"
]

for url in test_endpoints:
    response = requests.get(url)
    print(f"Status: {response.status_code} - {url}")
```

### Step 4: Update Package Dependencies
```bash
# Install the updated FinanceToolkit fork
pip uninstall financetoolkit
pip install -e /path/to/your/FinanceToolkit-fork
```

## Expected Benefits

1. **Immediate API Access**: All FMP endpoints will function correctly
2. **Future-Proof**: Using current FMP API standards
3. **Performance**: Modern endpoints may have better performance characteristics
4. **Reliability**: Stable endpoint infrastructure vs deprecated legacy system

## Notes

- **Backward Compatibility**: Legacy endpoints are deprecated and may stop working completely
- **API Key Requirements**: Same API key works for both legacy and stable endpoints
- **Rate Limits**: Same rate limiting applies to modern endpoints
- **Data Format**: Response format remains consistent between legacy and stable endpoints

---

**Generated**: Based on comprehensive FMP documentation analysis and FinanceToolkit codebase review
**Status**: Ready for implementation
**Priority**: **CRITICAL** - Legacy endpoints deprecated August 31, 2025
