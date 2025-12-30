# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FinanceToolkit is a Python library (v2.0.6) providing 150+ financial ratios, indicators, and performance measurements for transparent and efficient financial analysis. It supports equities, ETFs, currencies, commodities, indices, and more.

**Python compatibility:** 3.10 - 3.13

## Development Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Run all tests
pytest

# Run specific test file
pytest tests/ratios/test_profitability_model.py

# Run tests with coverage
pytest --cov=financetoolkit tests/

# Run tests with recording for API cassettes
pytest tests --record-mode=rewrite

# Rewrite expected test outputs
pytest --rewrite-expected

# Format code
black financetoolkit

# Lint
ruff check financetoolkit --fix

# Type checking
mypy financetoolkit --ignore-missing-imports

# Run all linting via pre-commit
pre-commit run --all-files

# Run API server (requires [api] extras)
uvicorn infrastructure.api:app --reload --port 8000

# Run infrastructure API
uvicorn infrastructure.api:app --reload

# Run Streamlit app
streamlit run infrastructure/streamlit_app.py
```

## Architecture

### Core Module Structure

Each financial module follows a Model-Controller pattern:
- `*_controller.py` - User-facing API, orchestrates data flow, handles caching
- `*_model.py` - Pure calculation functions (can be used independently)
- `helpers.py` - Utility functions per module

### Main Entry Point

```python
from financetoolkit import Toolkit
toolkit = Toolkit(['AAPL', 'MSFT'], api_key="YOUR_KEY")
```

### Standalone Controllers

These can be imported and used directly without ticker data:
```python
from financetoolkit import Economics, FixedIncome, Discovery, Portfolio
```

### Sub-controllers (accessed via Toolkit)

| Controller | Access | Description |
|------------|--------|-------------|
| Ratios | `toolkit.ratios` | 50+ financial ratios (profitability, liquidity, efficiency, solvency, valuation) |
| Models | `toolkit.models` | DCF, DuPont, WACC, Altman Z-Score, Piotroski F-Score |
| Technicals | `toolkit.technicals` | Momentum, volatility, breadth, overlap indicators |
| Options | `toolkit.options` | Black-Scholes, Binomial Trees, Greeks calculations |
| Risk | `toolkit.risk` | VaR, CVaR, EVaR, GARCH models |
| Performance | `toolkit.performance` | Performance attribution metrics |
| Portfolio | `toolkit.portfolio` | Portfolio analysis and optimization |
| FixedIncome | `toolkit.fixedincome` | Bonds, derivatives, rate data (FRED, Fed, ECB) |
| Economics | `toolkit.economics` | OECD, GMDB, macroeconomic indicators |
| Discovery | `toolkit.discovery` | Stock screening and discovery |

### Data Sources

- **Financial Modeling Prep (FMP)** - Primary source for financial statements and fundamentals
- **Yahoo Finance** - Fallback for historical price data
- **FRED/Fed/ECB** - Central bank and economic data
- **OECD** - International macroeconomic statistics
- **Euribor** - European interbank rates

### Key Files

| File | Description |
|------|-------------|
| `financetoolkit/toolkit_controller.py` | Main Toolkit entry point (~3,800 lines) |
| `financetoolkit/fmp_model.py` | Financial Modeling Prep API integration |
| `financetoolkit/historical_model.py` | Historical price data handling |
| `financetoolkit/fundamentals_model.py` | Financial statement processing |
| `financetoolkit/currencies_model.py` | Currency conversion handling |
| `financetoolkit/helpers.py` | Shared utility functions |
| `financetoolkit/normalization/` | CSV files for financial statement normalization |

## Code Style

- **Line length:** 122 characters
- **Formatter:** Black (with Jupyter support)
- **Linter:** Ruff (rules: E, W, F, Q, S, UP, I, PD, SIM, PLC, PLE, PLR, PLW)
- **Docstrings:** Google-style (`__docformat__ = "google"`)
- **Type hints:** Used throughout, checked with mypy
- **Division warnings:** Runtime warnings for division by zero are intentionally ignored in financial calculations

### Pre-commit Hooks

The repository uses these pre-commit hooks:
- `check-yaml`, `end-of-file-fixer`, `trailing-whitespace`
- `check-merge-conflict`, `detect-private-key`
- `black` - Code formatting
- `ruff` - Linting
- `codespell` - Spell checking
- `mypy` - Type checking

## Testing

Tests mirror the main package structure under `tests/`.

### Test Infrastructure

- **Custom `Record` class** in `conftest.py` for API response recording/playback
- **Recorder patterns:**
  - `csv/` - DataFrame/Series test outputs
  - `json/` - Dict/list test outputs
  - `txt/` - String test outputs
- **Test datasets:** `tests/datasets/`, `tests/csv/`, `tests/json/`
- **Pytest timeout:** Default 2 minutes per test

### Recording Test Data

```bash
# Record new API responses
pytest tests --record-mode=rewrite

# Record with specific modes
# - "all": Always save
# - "once": Only save if no recording exists
# - "new_episodes": Save if changed
# - "none": Fail if no recording exists
```

### Pytest Markers

- `@pytest.mark.forecast` - Forecasting tests
- `@pytest.mark.optimization` - Optimization tests
- `@pytest.mark.session` - Session-based tests
- `@pytest.mark.record_stdout` - Capture and compare stdout

## Additional Modules

### REST API (`api/` and `infrastructure/`)

FastAPI-based REST API for financial analysis:

```bash
# Simple API (api/main.py)
uvicorn api.main:app --reload

# Production API with caching, metrics, security (infrastructure/api.py)
uvicorn infrastructure.api:app --reload
```

**Key endpoints:**
- `/api/statements/{type}/{ticker}` - Financial statements
- `/api/ratios/{category}/{ticker}` - Financial ratios
- `/api/health-score/{ticker}` - Altman Z-Score, Piotroski F-Score
- `/api/risk/{ticker}` - VaR, max drawdown
- `/api/compare` - Multi-ticker comparison
- `/health`, `/health/ready` - Health probes
- `/metrics` - Prometheus metrics

**Environment variables:**
- `FMP_API_KEY` - Financial Modeling Prep API key
- `REDIS_URL` - Redis cache URL
- `ALLOWED_ORIGINS` - CORS origins
- `API_CACHE_TTL` - Cache TTL in seconds
- `ENVIRONMENT` - development/production

### Backtesting Engine (`backtesting/`)

Event-driven backtesting framework:

```python
from backtesting.engine import BacktestEngine, Order, Side

def my_strategy(date, prices, portfolio, engine):
    if date == engine.dates[0]:
        return [Order('AAPL', Side.BUY, 100)]
    return []

engine = BacktestEngine(price_data, initial_cash=100000)
results = engine.run(my_strategy)
print(results.summary())
```

**Features:** Transaction costs, portfolio tracking, performance metrics (CAGR, Sharpe, max drawdown)

### Infrastructure (`infrastructure/`)

Production-ready components:
- `api.py` - Full-featured FastAPI with Redis caching, Prometheus metrics
- `database.py` - SQLite-based caching layer
- `streamlit_app.py` - Interactive web dashboard
- `telegram_bot.py` - Telegram bot interface
- `security/` - Rate limiting, API key validation, audit logging

### Spreadsheet Integration (`excel/`, `sheets/`)

- `excel/financetoolkit.py` - Excel add-in functions
- `sheets/` - Google Sheets integration
- See `SPREADSHEET_INTEGRATION.md` for setup

## Deployment

### Docker

```bash
# Build image
docker build -t financetoolkit-api .

# Run container
docker run -p 8080:8080 -e FMP_API_KEY=your_key financetoolkit-api
```

### Google Cloud Run

The Dockerfile is configured for Cloud Run deployment:
- Uses Python 3.11-slim base
- Gunicorn + Uvicorn workers
- Health checks on `/health`
- Exposes port 8080

## Optional Dependencies

Install extras for additional functionality:

```bash
# API server
uv sync --extra api

# Web dashboard
uv sync --extra web

# Telegram bot
uv sync --extra bot

# All infrastructure
uv sync --extra infra
```

## Important Notes

- Never commit API keys or credentials
- Test API cassettes are stored in version control for reproducible tests
- The FMP API has rate limits; use caching in production
- Financial calculations may produce NaN/Inf values; these are intentionally not filtered
