#!/usr/bin/env python3
"""
Minimal FMP endpoint validation - BUDGET CONSCIOUS
Tests only 1 endpoint from each major category to validate fixes.
Total API calls: ~6-7 maximum
"""

import sys
import os
import requests
import json

# Add the current directory to Python path so we can import config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from config import FMP_API_KEY
except ImportError:
    print("ERROR: config.py not found or FMP_API_KEY not set")
    sys.exit(1)

def test_minimal_endpoints():
    """Test only the most critical endpoints from each category"""
    
    print("🔍 Testing FMP endpoint fixes (MINIMAL - Budget Conscious)")
    print(f"Using API key: {FMP_API_KEY[:10]}...")
    print(f"Estimated API calls: ~6-7 total\n")
    
    # Minimal test set - one endpoint from each major category
    test_cases = [
        {
            "category": "Company Profile",
            "endpoint": "profile", 
            "url": f"https://financialmodelingprep.com/stable/profile?symbol=AAPL&apikey={FMP_API_KEY}",
            "expected_field": "companyName"
        },
        {
            "category": "Financial Statements", 
            "endpoint": "income-statement",
            "url": f"https://financialmodelingprep.com/stable/income-statement?symbol=AAPL&period=annual&apikey={FMP_API_KEY}&limit=1",
            "expected_field": "revenue"
        },
        {
            "category": "Historical Data",
            "endpoint": "historical-price-full", 
            "url": f"https://financialmodelingprep.com/stable/historical-price-full?symbol=AAPL&apikey={FMP_API_KEY}&from=2024-01-01&to=2024-01-02",
            "expected_field": "historical"
        },
        {
            "category": "Market Data",
            "endpoint": "quote",
            "url": f"https://financialmodelingprep.com/stable/quote?symbol=AAPL&apikey={FMP_API_KEY}",
            "expected_field": "price"
        },
        {
            "category": "Discovery - Stock Screener",
            "endpoint": "stock-screener",
            "url": f"https://financialmodelingprep.com/stable/stock-screener?apikey={FMP_API_KEY}&marketCapMoreThan=100000000000&limit=3",
            "expected_field": "symbol"
        },
        {
            "category": "Discovery - Search",
            "endpoint": "search",
            "url": f"https://financialmodelingprep.com/stable/search?query=Apple&apikey={FMP_API_KEY}",
            "expected_field": "symbol"
        }
    ]
    
    results = []
    api_call_count = 0
    
    for i, test in enumerate(test_cases, 1):
        try:
            print(f"[{i}/{len(test_cases)}] Testing: {test['category']}")
            print(f"   Endpoint: {test['endpoint']}")
            
            response = requests.get(test['url'], timeout=15)
            api_call_count += 1
            
            if response.status_code == 200:
                data = response.json()
                
                # Check if we got expected data structure
                has_expected_field = False
                if isinstance(data, list) and len(data) > 0:
                    has_expected_field = test['expected_field'] in data[0]
                elif isinstance(data, dict):
                    has_expected_field = test['expected_field'] in data or (
                        'historical' in data and len(data.get('historical', [])) > 0
                    )
                
                if has_expected_field:
                    print(f"   ✅ SUCCESS - Data structure looks correct")
                    results.append(True)
                else:
                    print(f"   ⚠️  SUCCESS (HTTP 200) but unexpected data structure")
                    print(f"   📋 Data preview: {str(data)[:200]}...")
                    results.append(False)
                    
            elif response.status_code == 403:
                print(f"   ❌ FAILED - HTTP 403 (Likely still using old endpoint or API key issue)")
                results.append(False)
            elif response.status_code == 429:
                print(f"   ⚠️  Rate limited - Try again later")
                results.append(False)
            else:
                print(f"   ❌ FAILED - HTTP {response.status_code}")
                print(f"   📋 Response: {response.text[:200]}...")
                results.append(False)
                
        except Exception as e:
            print(f"   ❌ ERROR - {str(e)}")
            results.append(False)
        
        print()  # Empty line for readability
    
    # Summary
    success_count = sum(results)
    success_rate = success_count / len(results) * 100
    
    print(f"{'='*60}")
    print(f"📊 RESULTS SUMMARY:")
    print(f"   ✅ Successful: {success_count}/{len(results)} ({success_rate:.1f}%)")
    print(f"   🔢 API calls used: {api_call_count}")
    print(f"   💰 Remaining budget: ~{50-api_call_count} calls")
    
    if success_rate >= 80:
        print(f"\n🎉 GREAT NEWS! The endpoint updates appear to be working correctly!")
        print(f"   Your FinanceToolkit fork should now work with FMP's stable endpoints.")
    elif success_rate >= 50:
        print(f"\n⚠️  MIXED RESULTS: Some endpoints working, others may need attention.")
    else:
        print(f"\n❌ ISSUES DETECTED: Multiple endpoints failed - may need further investigation.")
    
    print(f"\n💡 Next steps:")
    if success_rate >= 80:
        print(f"   • Your FinanceToolkit is ready to use!")
        print(f"   • Try running some actual financial analysis")
    else:
        print(f"   • Check API key permissions")
        print(f"   • Verify endpoint URLs in the error responses")
        print(f"   • Some endpoints might need different parameter formats")
    
    return success_rate >= 80

if __name__ == "__main__":
    test_minimal_endpoints()
