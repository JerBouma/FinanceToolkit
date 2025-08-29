#!/usr/bin/env python3
"""
Minimal FMP Endpoint Validation - Budget Conscious
Tests just ONE call per major endpoint type to validate fixes
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

# Test only the most critical endpoints with exactly the URLs from FMP docs
CRITICAL_TESTS = [
    {
        "name": "Company Profile",
        "url": f"https://financialmodelingprep.com/stable/profile?symbol=AAPL&apikey={FMP_API_KEY}",
        "category": "Company Data"
    },
    {
        "name": "Income Statement",
        "url": f"https://financialmodelingprep.com/stable/income-statement?symbol=AAPL&period=annual&limit=1&apikey={FMP_API_KEY}",
        "category": "Financial Statements"
    },
    {
        "name": "Stock Quote", 
        "url": f"https://financialmodelingprep.com/stable/quote?symbol=AAPL&apikey={FMP_API_KEY}",
        "category": "Market Data"
    },
    {
        "name": "Historical Price",
        "url": f"https://financialmodelingprep.com/stable/historical-price-eod/light?symbol=AAPL&apikey={FMP_API_KEY}",
        "category": "Chart Data"
    },
    {
        "name": "Stock Screener", 
        "url": f"https://financialmodelingprep.com/stable/company-screener?marketCapMoreThan=1000000000&limit=1&apikey={FMP_API_KEY}",
        "category": "Discovery"
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
        else:
            print(f"   ❌ FAILED - HTTP {response.status_code}")
            if response.status_code == 403:
                print(f"      Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"   ❌ ERROR - {str(e)}")
        return False

def main():
    print("🔍 FMP Endpoint Validation (Budget-Conscious)")
    print(f"API Key: {FMP_API_KEY[:8]}...")
    print(f"Total tests: {len(CRITICAL_TESTS)}")
    
    results = []
    for test in CRITICAL_TESTS:
        result = test_endpoint(test)
        results.append(result)
    
    success_count = sum(results)
    success_rate = success_count / len(results) * 100
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {success_count}/{len(results)} passed ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 Endpoints are working correctly!")
        print("✅ Your FinanceToolkit patches are successful!")
    else:
        print("⚠️  Some endpoints need attention")
        print("❌ Check the failing endpoints above")
    
    return success_rate >= 80

if __name__ == "__main__":
    main()
