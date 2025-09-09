#!/usr/bin/env python3
"""
Test script to validate FMP endpoint fixes
Run this script to test if the updated endpoints are working correctly.
"""

import sys
import os

# Add the current directory to Python path so we can import config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from config import FMP_API_KEY
except ImportError:
    print("ERROR: config.py not found or FMP_API_KEY not set")
    print("Please create config.py and set your FMP_API_KEY")
    sys.exit(1)

import requests
import pandas as pd
from financetoolkit.fmp_model import get_financial_data

def test_endpoints():
    """Test key endpoints to verify they're working"""
    
    if FMP_API_KEY == "your_fmp_api_key_here":
        print("ERROR: Please replace 'your_fmp_api_key_here' with your actual FMP API key in config.py")
        return False
    
    print("Testing FMP endpoint updates...")
    print(f"Using API key: {FMP_API_KEY[:10]}...")
    
    # Test endpoints
    test_cases = [
        {
            "name": "Company Profile",
            "url": f"https://financialmodelingprep.com/stable/profile?symbol=AAPL&apikey={FMP_API_KEY}"
        },
        {
            "name": "Income Statement",
            "url": f"https://financialmodelingprep.com/stable/income-statement?symbol=AAPL&period=annual&apikey={FMP_API_KEY}&limit=1"
        },
        {
            "name": "Historical Price",
            "url": f"https://financialmodelingprep.com/stable/historical-price-full?symbol=AAPL&apikey={FMP_API_KEY}&from=2024-01-01&to=2024-01-05"
        },
        {
            "name": "Quote",
            "url": f"https://financialmodelingprep.com/stable/quote?symbol=AAPL&apikey={FMP_API_KEY}"
        },
        {
            "name": "Stock Screener",
            "url": f"https://financialmodelingprep.com/stable/stock-screener?apikey={FMP_API_KEY}&marketCapMoreThan=1000000000&limit=5"
        }
    ]
    
    results = []
    for test in test_cases:
        try:
            print(f"\nTesting: {test['name']}")
            response = requests.get(test['url'], timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data:  # Check if data is not empty
                    print(f"✅ {test['name']}: SUCCESS - Got {len(data)} records")
                    results.append(True)
                else:
                    print(f"⚠️  {test['name']}: WARNING - No data returned")
                    results.append(False)
            else:
                print(f"❌ {test['name']}: FAILED - HTTP {response.status_code}")
                if response.status_code == 403:
                    print("   This might be an API key issue or rate limit")
                results.append(False)
                
        except Exception as e:
            print(f"❌ {test['name']}: ERROR - {str(e)}")
            results.append(False)
    
    success_rate = sum(results) / len(results) * 100
    print(f"\n{'='*50}")
    print(f"Test Results: {sum(results)}/{len(results)} passed ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 Endpoints appear to be working correctly!")
        return True
    else:
        print("⚠️  Some endpoints may need attention")
        return False

if __name__ == "__main__":
    test_endpoints()
