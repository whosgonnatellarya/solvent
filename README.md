# Solvent: Cash Flow Decision Engine (ML-powered)

Solvent predicts invoice payment risk and returns a prescriptive action plan - not just a score. Built with 2026-standard Python tooling.

## How it works

1. **Data**: 1,000 synthetic invoices generated with realistic client payment behavior
2. **Features**: Engineered with Polars - `historical_client_delay`, `relative_amount`, `is_weekend`, `days_until_due`, `avg_client_amount`
3. **Model**: XGBoost Survival Model (`survival:aft`) predicting days-to-payment over a 60-day horizon. MAE: 11.45 days
4. **API**: FastAPI endpoint returning a specific nudge action based on predicted risk

## Example

**Request:**
```json
POST /predict
{
  "historical_client_delay": 45,
  "relative_amount": 1.5,
  "is_weekend": 0,
  "days_until_due": 30,
  "avg_client_amount": 25000
}
```

**Response:**
```json
{
  "predicted_days_to_pay": 34.97,
  "nudge": "Medium risk: send a polite email in 3 days"
}
```

## Run it yourself

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Generate data + features
uv run generate_data.py
uv run features.py

# Train model
uv run train.py

# Start API
uv run uvicorn main:app --reload
```

## Tech stack
- **Polars** - fast feature engineering
- **XGBoost** - survival:aft model
- **FastAPI** - inference API
- **uv** - package management