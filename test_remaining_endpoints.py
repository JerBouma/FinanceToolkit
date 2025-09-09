#!/usr/bin/env python3
"""
Complete FMP Endpoint Validation - Remaining Endpoints
Tests all the other endpoints we fixed, excluding the 4 that already passed
"""

import sys
import os
import requests

# Add current directory to path for config import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from config import FMP_API_KEY
except ImportError:
    print("ERROR: config.py not found")
    sys.exit(1)

# Test all remaining fixed endpoints (excluding the 4 that already passed)
REMAINING_TESTS = [
    # FMP Model endpoints (excluding profile and income statement)
    {
        "name": "Historical Dividends",
        "url": f"https://financialmodelingprep.com/stable/dividends?symbol=AAPL&from=2024-01-01&to=2024-12-31&apikey={FMP_API_KEY}",
        "category": "FMP Model - Historical Data"
    },
    {
        "name": "Historical Chart 1min",
        "url": f"https://financialmodelingprep.com/stable/historical-chart/1min?symbol=AAPL&from=2024-08-28&to=2024-08-29&apikey={FMP_API_KEY}",
        "category": "FMP Model - Chart Data"
    },
    {
        "name": "Analyst Estimates",
        "url": f"https://financialmodelingprep.com/stable/analyst-estimates?symbol=AAPL&period=annual&apikey={FMP_API_KEY}",
        "category": "FMP Model - Analyst Data"
    },
    {
        "name": "Historical Rating",
        "url": f"https://financialmodelingprep.com/stable/ratings-historical?symbol=AAPL&apikey={FMP_API_KEY}",
        "category": "FMP Model - Ratings"
    },
    {
        "name": "Earnings Calendar",
        "url": f"https://financialmodelingprep.com/stable/earnings?symbol=AAPL&apikey={FMP_API_KEY}",
        "category": "FMP Model - Earnings"
    },
    
    # Discovery Model endpoints (excluding stock screener)
    {
        "name": "Search Companies",
        "url": f"https://financialmodelingprep.com/stable/search-symbol?query=AAPL&limit=5&apikey={FMP_API_KEY}",
        "category": "Discovery - Search"
    },
    {
        "name": "Stock List",
        "url": f"https://financialmodelingprep.com/stable/stock-list?apikey={FMP_API_KEY}",
        "category": "Discovery - Lists"
    },
    {
        "name": "Market Gainers",
        "url": f"https://financialmodelingprep.com/stable/biggest-gainers?apikey={FMP_API_KEY}",
        "category": "Discovery - Market Performance"
    },
    {
        "name": "Market Losers", 
        "url": f"https://financialmodelingprep.com/stable/biggest-losers?apikey={FMP_API_KEY}",
        "category": "Discovery - Market Performance"
    },
    {
        "name": "Most Active",
        "url": f"https://financialmodelingprep.com/stable/most-actives?apikey={FMP_API_KEY}",
        "category": "Discovery - Market Performance"
    },
    {
        "name": "Cryptocurrency List",
        "url": f"https://financialmodelingprep.com/stable/cryptocurrency-list?apikey={FMP_API_KEY}",
        "category": "Discovery - Crypto"
    },
    {
        "name": "Crypto Quotes",
        "url": f"https://financialmodelingprep.com/stable/batch-crypto-quotes?apikey={FMP_API_KEY}",
        "category": "Discovery - Crypto Quotes"
    },
    {
        "name": "Forex List",
        "url": f"https://financialmodelingprep.com/stable/forex-list?apikey={FMP_API_KEY}",
        "category": "Discovery - Forex"
    },
    {
        "name": "Forex Quotes",
        "url": f"https://financialmodelingprep.com/stable/batch-forex-quotes?apikey={FMP_API_KEY}",
        "category": "Discovery - Forex Quotes"
    },
    {
        "name": "Commodities List",
        "url": f"https://financialmodelingprep.com/stable/commodities-list?apikey={FMP_API_KEY}",
        "category": "Discovery - Commodities"
    },
    {
        "name": "Commodity Quotes",
        "url": f"https://financialmodelingprep.com/stable/batch-commodity-quotes?apikey={FMP_API_KEY}",
        "category": "Discovery - Commodity Quotes"
    },
    {
        "name": "ETF List",
        "url": f"https://financialmodelingprep.com/stable/etf-list?apikey={FMP_API_KEY}",
        "category": "Discovery - ETF"
    },
    {
        "name": "Index List",
        "url": f"https://financialmodelingprep.com/stable/index-list?apikey={FMP_API_KEY}",
        "category": "Discovery - Indexes"
    },
    {
        "name": "Index Quotes",
        "url": f"https://financialmodelingprep.com/stable/batch-index-quotes?apikey={FMP_API_KEY}",
        "category": "Discovery - Index Quotes"
    }
]

def test_endpoint(test):
    """Test a single endpoint"""
    try:
        print(f"\n🧪 Testing: {test['name']} ({test['category']})")
        print(f"   URL: {test['url'][:80]}...")
        
        response = requests.get(test['url'], timeout=15)
        
        if response.status_code == 200:
            try:
                data = response.json()
                if data and len(data) > 0:
                    print(f"   ✅ SUCCESS - {len(data)} records returned")
                    return True
                else:
                    print(f"   ⚠️  WARNING - Empty response")
                    return False
            except:
                print(f"   ❌ ERROR - Invalid JSON response")
                return False
        elif response.status_code == 402:
            print(f"   💰 SUBSCRIPTION - Requires higher tier (HTTP 402)")
            return None  # Don't count as failure
        elif response.status_code == 403:
            print(f"   🔐 FORBIDDEN - Check API key or endpoint (HTTP 403)")
            print(f"      Response: {response.text[:200]}")
            return False
        else:
            print(f"   ❌ FAILED - HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ ERROR - {str(e)}")
        return False

def main():
    print("🔍 FMP Remaining Endpoints Validation")
    print(f"API Key: {FMP_API_KEY[:8]}...")
    print(f"Total remaining tests: {len(REMAINING_TESTS)}")
    print("\n⚠️  Note: Some endpoints may require higher subscription tiers")
    
    results = []
    subscription_limited = []
    
    for test in REMAINING_TESTS:
        result = test_endpoint(test)
        if result is None:  # Subscription limited
            subscription_limited.append(test['name'])
        else:
            results.append(result)
    
    success_count = sum(results)
    total_tested = len(results)
    success_rate = success_count / total_tested * 100 if total_tested > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {success_count}/{total_tested} endpoints working ({success_rate:.1f}%)")
    
    if subscription_limited:
        print(f"SUBSCRIPTION LIMITED: {len(subscription_limited)} endpoints require higher tier:")
        for endpoint in subscription_limited:
            print(f"  - {endpoint}")
    
    print(f"\nCOMBINED WITH PREVIOUS VALIDATION:")
    total_all = 4 + total_tested  # 4 from previous successful validation
    success_all = 4 + success_count
    combined_rate = success_all / total_all * 100
    print(f"OVERALL: {success_all}/{total_all} endpoints working ({combined_rate:.1f}%)")
    
    if combined_rate >= 70:
        print("\n🎉 Your FMP endpoint migration is highly successful!")
        print("✅ The majority of endpoints are working correctly!")
    else:
        print("\n⚠️  Some endpoints may need attention")
    
    return combined_rate >= 70

if __name__ == "__main__":
    main()
