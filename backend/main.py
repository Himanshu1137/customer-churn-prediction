from pathlib import Path
from typing import Literal
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
MODEL_PATH = BASE_DIR / "model" / "churn_model.pkl"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"

app = FastAPI(title="Customer Churn Prediction", version="1.0.0")
model = joblib.load(MODEL_PATH)
metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}

class CustomerInput(BaseModel):
    gender: Literal["Male", "Female"]
    senior_citizen: int = Field(ge=0, le=1)
    partner: Literal["Yes", "No"]
    dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=100)
    phone_service: Literal["Yes", "No"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    monthly_charges: float = Field(ge=0)
    total_charges: float = Field(ge=0)
    contract: Literal["Month-to-month", "One year", "Two year"]
    payment_method: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]


def explain_factors(d: CustomerInput):
    factors = []
    if d.contract == "Month-to-month": factors.append(("Contract", 92, "Month-to-month plans usually have less switching friction."))
    elif d.contract == "One year": factors.append(("Contract", 45, "Annual contract gives moderate retention stability."))
    else: factors.append(("Contract", 18, "Two-year contract is a strong retention signal."))

    tenure_score = 90 if d.tenure < 6 else 72 if d.tenure < 12 else 50 if d.tenure < 24 else 28 if d.tenure < 48 else 14
    factors.append(("Tenure", tenure_score, f"Customer tenure is {d.tenure} months."))

    billing = 28
    if d.payment_method == "Electronic check": billing += 35
    if d.monthly_charges > 85: billing += 25
    elif d.monthly_charges < 45: billing -= 8
    factors.append(("Billing", max(5, min(100, billing)), "Payment method and monthly cost influence billing risk."))

    service = 76 if d.internet_service == "Fiber optic" else 42 if d.internet_service == "DSL" else 20
    if d.senior_citizen: service = min(100, service + 8)
    factors.append(("Service", service, "Service mix is used as a supporting risk signal."))
    return [{"name":n,"score":int(s),"detail":t} for n,s,t in factors]

@app.get("/api/health")
def health():
    return {"status":"ok","model_loaded":True}

@app.get("/api/model-info")
def model_info():
    return metrics

@app.post("/api/predict")
def predict(customer: CustomerInput):
    try:
        frame = pd.DataFrame([customer.model_dump()])
        probability = float(model.predict_proba(frame)[0, 1])
        prediction = int(probability >= 0.5)
        pct = round(probability * 100, 1)
        risk = "High" if pct >= 70 else "Medium" if pct >= 40 else "Low"
        priority = "Immediate retention action" if risk == "High" else "Review within 7 days" if risk == "Medium" else "Routine monitoring"
        segment = "New / early-stage" if customer.tenure < 12 else "Developing" if customer.tenure < 36 else "Established"
        return {
            "prediction": prediction,
            "churn_probability": pct,
            "risk_level": risk,
            "retention_priority": priority,
            "customer_segment": segment,
            "factors": explain_factors(customer),
            "model_note": metrics.get("note", "")
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")

@app.get("/")
def frontend():
    return FileResponse(FRONTEND_DIR / "index.html")
