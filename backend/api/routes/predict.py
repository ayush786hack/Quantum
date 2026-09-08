import sys
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from prediction.predictor import predict_fuel_consumption_with_uncertainty, explain_fuel_prediction, calculate_wtw_emissions, calculate_voyage_cost, calculate_cii_rating, calculate_compliance_forecast

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
