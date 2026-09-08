import sys
import os
import json
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from optimization.optimizer import run_fleet_optimization

router = APIRouter(prefix="/api", tags=["Optimization"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

class OptimizationRequest(BaseModel):
    algorithm: str = Field(default="QIGA", example="QIGA")
    fleet_size: int = Field(default=10, example=10)
    generations: int = Field(default=25, example=25)
    pop_size: int = Field(default=35, example=35)
    fuel_prices: dict = Field(default=None)
    carbon_tax: float = Field(default=0.0)
    demand_mult: float = Field(default=1.0)
    allowed_fuels: list = Field(default=None)
    shore_power_enabled: bool = Field(default=True)
    cargo_demand: float = Field(default=None)

@router.post("/optimize-fleet")
def optimize_fleet_endpoint(req: OptimizationRequest):
    try:
        vessels_path = os.path.join(RAW_DIR, "vessels.json")
        routes_path = os.path.join(RAW_DIR, "routes.json")
        
        if not os.path.exists(vessels_path) or not os.path.exists(routes_path):
            from data.synthetic_generator import run as run_gen
            run_gen()
            
        with open(vessels_path) as f:
            vessels = json.load(f)
        with open(routes_path) as f:
            routes = json.load(f)
            
        selected_vessels = vessels[:min(req.fleet_size, len(vessels))]
        
        result = run_fleet_optimization(
            vessels=selected_vessels,
            routes=routes,
            algorithm=req.algorithm,
            pop_size=req.pop_size,
            generations=req.generations,
            fuel_prices=req.fuel_prices,
            carbon_tax=req.carbon_tax,
            demand_mult=req.demand_mult,
            allowed_fuels=req.allowed_fuels,
            shore_power_enabled=req.shore_power_enabled,
            cargo_demand=req.cargo_demand
        )
        return sanitize_for_json(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
