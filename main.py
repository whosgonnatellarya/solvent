from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import numpy as np

app = FastAPI()
model = xgb.Booster()
model.load_model("model.json")

class InvoiceFeatures(BaseModel):
    historical_client_delay: float
    relative_amount: float
    is_weekend: int
    days_until_due: float
    avg_client_amount: float

@app.post("/predict")
def predict(invoice: InvoiceFeatures):
    features = np.array([[invoice.historical_client_delay, invoice.relative_amount, invoice.is_weekend, invoice.days_until_due, invoice.avg_client_amount]])
    dmatrix = xgb.DMatrix(features)
    prediction = model.predict(dmatrix)[0]
    
    if prediction > 60:
        nudge = "high risk: send a firm WhatsApp reminder tomorrow at 10 AM"
    elif prediction > 30:
        nudge = "medium risk: send a polite email in 3 days"
    else:
        nudge = "low risk: no action needed"

    return {"predicted_days_to_pay": float(prediction), "nudge": nudge}