import requests

test_invoices = [
    {"historical_client_delay": 75, "relative_amount": 2.0, "is_weekend": 1, "days_until_due": 10, "avg_client_amount": 50000},
    {"historical_client_delay": 25, "relative_amount": 0.8, "is_weekend": 0, "days_until_due": 45, "avg_client_amount": 15000},
    {"historical_client_delay": 5, "relative_amount": 0.5, "is_weekend": 0, "days_until_due": 60, "avg_client_amount": 8000},
]

for i, invoice in enumerate(test_invoices):
    response = requests.post("http://127.0.0.1:8000/predict", json=invoice)
    result = response.json()
    print(f"Invoice {i+1}: {result['nudge']} ({result['predicted_days_to_pay']:.1f} days)")