from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import xgboost as xgb
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = xgb.Booster()
model.load_model("model.json")

class InvoiceFeatures(BaseModel):
    historical_client_delay: float
    relative_amount: float
    is_weekend: int
    days_until_due: float
    avg_client_amount: float

def _score(invoice: InvoiceFeatures):
    features = np.array([[
        invoice.historical_client_delay,
        invoice.relative_amount,
        invoice.is_weekend,
        invoice.days_until_due,
        invoice.avg_client_amount,
    ]])
    prediction = float(model.predict(xgb.DMatrix(features))[0])
    if prediction > 60:
        nudge = "high risk: send a firm WhatsApp reminder tomorrow at 10 AM"
    elif prediction > 30:
        nudge = "medium risk: send a polite email in 3 days"
    else:
        nudge = "low risk: no action needed"
    return {"predicted_days_to_pay": prediction, "nudge": nudge, "invoice": invoice.model_dump()}

class BatchRequest(BaseModel):
    invoices: List[InvoiceFeatures]

@app.post("/predict")
def predict(invoice: InvoiceFeatures):
    result = _score(invoice)
    return {"predicted_days_to_pay": result["predicted_days_to_pay"], "nudge": result["nudge"]}

@app.post("/predict_batch")
def predict_batch(request: BatchRequest):
    results = [_score(inv) for inv in request.invoices]
    results.sort(key=lambda x: x["predicted_days_to_pay"], reverse=True)
    return {"predictions": results}

@app.get("/health")
def health():
    return {"status": "ok"}

