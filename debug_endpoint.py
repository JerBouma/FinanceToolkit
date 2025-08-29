import requests

# Test one specific failing endpoint
api_key = "UBuyCEYKTUM77pS4Z5ITGX38DUziRbzF"
url = f"https://financialmodelingprep.com/stable/historical-price-full?symbol=AAPL&apikey={api_key}&from=2024-01-01&to=2024-01-02"

print(f"Testing URL: {url}")
response = requests.get(url)
print(f"Status Code: {response.status_code}")
print(f"Response: {response.text[:500]}")
