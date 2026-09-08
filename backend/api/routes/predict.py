import sys
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from prediction.predictor import predict_fuel_consumption_with_uncertainty, explain_fuel_prediction, calculate_wtw_emissions, calculate_voyage_cost, calculate_cii_rating, calculate_compliance_forecast
from prediction.data_preprocessing import load_and_preprocess_data

router = APIRouter(prefix="/api", tags=["Prediction"])

class FuelPredictionRequest(BaseModel):
    vessel_type: str = Field(..., example="Container Ship")
    capacity: float = Field(..., example=10000)
    engine_power_kw: float = Field(default=35000, example=35000)
    distance_nmi: float = Field(..., example=3600)
    speed_knots: float = Field(..., example=18.0)
    sea_state: float = Field(default=3.0, example=3.5)
    payload_pct: float = Field(default=0.85, example=0.85)
    fuel_type: str = Field(default="HFO", example="Methanol")
    shore_power_used: bool = Field(default=False)
    carbon_tax_usd_tco2e: float = Field(default=0.0)

@router.post("/predict-fuel")
def predict_fuel_endpoint(req: FuelPredictionRequest):
    try:
        prediction = predict_fuel_consumption_with_uncertainty(
            vessel_type=req.vessel_type,
            capacity=req.capacity,
            engine_power_kw=req.engine_power_kw,
            distance_nmi=req.distance_nmi,
            speed_knots=req.speed_knots,
            sea_state=req.sea_state,
            payload_pct=req.payload_pct,
            fuel_type=req.fuel_type
        )
        fc = prediction["fuel_consumption_tonnes"]
        emissions = calculate_wtw_emissions(fc, req.fuel_type)
        cost = calculate_voyage_cost(
            fuel_consumption_tonnes=fc,
            fuel_type=req.fuel_type,
            distance_nmi=req.distance_nmi,
            speed_knots=req.speed_knots,
            shore_power_used=req.shore_power_used,
            carbon_tax_usd_tco2e=req.carbon_tax_usd_tco2e
        )
        cii = calculate_cii_rating(emissions, req.capacity, req.distance_nmi)
        compliance = calculate_compliance_forecast(emissions, req.capacity, req.distance_nmi, req.speed_knots)
        
        return {
            "status": "success",
            **prediction,
            "wtw_emissions_tco2e": emissions,
            "total_cost_usd": cost,
            "cii_rating": cii,
            "compliance": compliance,
            "explainability": explain_fuel_prediction(
                vessel_type=req.vessel_type, capacity=req.capacity, engine_power_kw=req.engine_power_kw,
                distance_nmi=req.distance_nmi, speed_knots=req.speed_knots, sea_state=req.sea_state,
                payload_pct=req.payload_pct, fuel_type=req.fuel_type
            )
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/prediction-metrics")
def prediction_metrics():
    """Return reproducible holdout metrics and residual diagnostics from the saved model."""
    artifact = __import__("prediction.predictor", fromlist=["get_model_artifact"]).get_model_artifact()
    if artifact is None:
        raise HTTPException(status_code=503, detail="Train the model artifact first")
    X, y, _ = load_and_preprocess_data()
    model = artifact["model"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    predictions = model.predict(X_test)
    residuals = np.asarray(y_test) - predictions
    return {
        "model": "GradientBoostingRegressor",
        "dataset_rows": int(len(y)),
        "evaluation_split": "deterministic 80/20 holdout (random_state=42)",
        "r2_score": round(float(r2_score(y_test, predictions)), 4),
        "rmse_tonnes": round(float(np.sqrt(mean_squared_error(y_test, predictions))), 4),
        "mae_tonnes": round(float(mean_absolute_error(y_test, predictions)), 4),
        "residual_mean_tonnes": round(float(np.mean(residuals)), 4),
        "residual_std_tonnes": round(float(np.std(residuals)), 4),
        "residual_percentiles_tonnes": {str(percentile): round(float(np.percentile(residuals, percentile)), 4) for percentile in [5, 50, 95]},
        "feature_count": int(X.shape[1]),
    }
