import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "prediction", "fuel_predictor_model.joblib")

FUEL_SPECS = {
    "HFO": {"energy_density_mj_kg": 40.5, "wtw_gco2e_mj": 92.0, "base_price_usd_t": 600},
    "LNG": {"energy_density_mj_kg": 50.0, "wtw_gco2e_mj": 76.0, "base_price_usd_t": 850},
    "Methanol": {"energy_density_mj_kg": 19.9, "wtw_gco2e_mj": 65.0, "base_price_usd_t": 750},
    "Hydrogen": {"energy_density_mj_kg": 120.0, "wtw_gco2e_mj": 10.0, "base_price_usd_t": 2200},
    "Ammonia": {"energy_density_mj_kg": 18.6, "wtw_gco2e_mj": 5.0, "base_price_usd_t": 1100}
}

_MODEL_CACHE = None

def _prediction_features(
    vessel_type: str,
    capacity: float,
    engine_power_kw: float,
    distance_nmi: float,
    speed_knots: float,
    sea_state: float,
    payload_pct: float,
    fuel_type: str
) -> pd.DataFrame:
    artifact = get_model_artifact()
    if artifact is None:
        return None

    row = {col: 0.0 for col in artifact["feature_cols"]}
    row.update({
        "capacity": float(capacity),
        "engine_power_kw": float(engine_power_kw),
        "distance_nmi": float(distance_nmi),
        "speed_knots": float(speed_knots),
        "speed_cubed": float(speed_knots ** 3),
        "sea_state": float(sea_state),
        "sea_state_penalty": float((sea_state - 1.0) ** 1.3),
        "payload_pct": float(payload_pct),
        "energy_density_ratio": 40.5 / FUEL_SPECS.get(fuel_type, FUEL_SPECS["HFO"])["energy_density_mj_kg"]
    })
    if f"vtype_{vessel_type}" in row:
        row[f"vtype_{vessel_type}"] = 1.0
    if f"fuel_{fuel_type}" in row:
        row[f"fuel_{fuel_type}"] = 1.0
    return pd.DataFrame([row])[artifact["feature_cols"]]

def get_model_artifact():
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
        
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            _MODEL_CACHE = joblib.load(MODEL_SAVE_PATH)
            return _MODEL_CACHE
        except Exception as e:
            print(f"[Predictor Warning] Failed to load model: {e}")
            
    return None

def predict_fuel_consumption(
    vessel_type: str,
    capacity: float,
    engine_power_kw: float,
    distance_nmi: float,
    speed_knots: float,
    sea_state: float = 3.0,
    payload_pct: float = 0.85,
    fuel_type: str = "HFO"
) -> float:
    """Predicts fuel consumption in tonnes for a given voyage."""
    artifact = get_model_artifact()
    
    if artifact is not None:
        model = artifact["model"]
        feature_cols = artifact["feature_cols"]
        
        df_input = _prediction_features(vessel_type, capacity, engine_power_kw, distance_nmi, speed_knots, sea_state, payload_pct, fuel_type)
        predicted_tonnes = float(model.predict(df_input)[0])
        return max(0.1, round(predicted_tonnes, 2))
    else:
        # Fallback physics calculation FC ~ D * s^3 / (eta * LHV)
        lhv = FUEL_SPECS.get(fuel_type, FUEL_SPECS["HFO"])["energy_density_mj_kg"]
        cap_factor = (capacity / 10000.0) ** 0.65
        weather_penalty = 1.0 + 0.04 * (sea_state - 1.0) ** 1.3
        load_factor = 0.85 + 0.25 * payload_pct
        hours = distance_nmi / max(1.0, speed_knots)
        
        daily_fc_hfo = 25.0 * cap_factor * ((speed_knots / 15.0) ** 3) * weather_penalty * load_factor
        energy_req_mj = daily_fc_hfo * 1000.0 * 40.5
        daily_fc_chosen = (energy_req_mj / lhv) / 1000.0
        
        fc_tonnes = daily_fc_chosen * (hours / 24.0)
        return max(0.1, round(fc_tonnes, 2))

def predict_fuel_consumption_with_uncertainty(**kwargs) -> dict:
    """Return a point estimate and a practical confidence band for demo decisions."""
    point = predict_fuel_consumption(**kwargs)
    artifact = get_model_artifact()
    if artifact is not None:
        model = artifact["model"]
        features = _prediction_features(**kwargs)
        tree_predictions = np.asarray([tree.predict(features)[0] for tree in model.estimators_.ravel()])
        spread = float(np.std(tree_predictions)) if len(tree_predictions) else point * 0.08
        confidence = 0.90
    else:
        spread = point * 0.08
        confidence = 0.80
    margin = max(0.05, 1.645 * spread)
    return {
        "fuel_consumption_tonnes": round(point, 2),
        "lower_bound_tonnes": round(max(0.1, point - margin), 2),
        "upper_bound_tonnes": round(point + margin, 2),
        "confidence_level": confidence,
        "uncertainty_method": "gradient_boosting_tree_spread" if artifact is not None else "physics_fallback_margin"
    }

def calculate_wtw_emissions(fuel_consumption_tonnes: float, fuel_type: str) -> float:
    """Calculates Well-to-Wake (WtW) GHG emissions in tonnes CO2e."""
    spec = FUEL_SPECS.get(fuel_type, FUEL_SPECS["HFO"])
    lhv_mj_kg = spec["energy_density_mj_kg"]
    wtw_factor_g_mj = spec["wtw_gco2e_mj"]
    
    total_energy_mj = fuel_consumption_tonnes * 1000.0 * lhv_mj_kg
    emissions_tco2e = (total_energy_mj * wtw_factor_g_mj) / 1e6
    return max(0.0, round(emissions_tco2e, 2))

def calculate_voyage_cost(
    fuel_consumption_tonnes: float,
    fuel_type: str,
    distance_nmi: float,
    speed_knots: float,
    shore_power_used: bool,
    custom_fuel_prices: dict = None,
    carbon_tax_usd_tco2e: float = 0.0
) -> float:
    """Calculates total voyage operational cost ($)."""
    prices = custom_fuel_prices or {}
    fuel_price = prices.get(fuel_type, FUEL_SPECS.get(fuel_type, FUEL_SPECS["HFO"])["base_price_usd_t"])
    
    fuel_cost = fuel_consumption_tonnes * fuel_price
    
    hours_in_port = 24.0
    if shore_power_used:
        shore_power_cost = hours_in_port * 120.0 * 0.18 # Grid tariff
    else:
        # Auxiliary engine fuel consumption at berth (0.5 tonnes HFO @ fuel_price)
        shore_power_cost = 0.5 * prices.get("HFO", FUEL_SPECS["HFO"]["base_price_usd_t"])
        
    wtw_emissions = calculate_wtw_emissions(fuel_consumption_tonnes, fuel_type)
    carbon_tax_cost = wtw_emissions * carbon_tax_usd_tco2e
    
    total_cost = fuel_cost + shore_power_cost + carbon_tax_cost
    return round(total_cost, 2)

def calculate_cii_rating(emissions_tco2e: float, capacity: float, distance_nmi: float) -> str:
    """Calculates IMO Carbon Intensity Indicator (CII) Rating (A to E)."""
    if capacity <= 0 or distance_nmi <= 0:
        return "C"
        
    actual_cii_gco2_t_nmi = (emissions_tco2e * 1e6) / (capacity * distance_nmi)
    
    # Reference baseline CII curve for container/bulk
    ref_cii = 12.5 * (capacity ** (-0.15))
    ratio = actual_cii_gco2_t_nmi / max(0.1, ref_cii)
    
    if ratio <= 0.75:
        return "A"
    elif ratio <= 0.90:
        return "B"
    elif ratio <= 1.05:
        return "C"
    elif ratio <= 1.25:
        return "D"
    else:
        return "E"

if __name__ == "__main__":
    fc = predict_fuel_consumption("Container Ship", 10000, 35000, 3600, 18.0, sea_state=3.5, fuel_type="Methanol")
    emissions = calculate_wtw_emissions(fc, "Methanol")
    cost = calculate_voyage_cost(fc, "Methanol", 3600, 18.0, shore_power_used=True, carbon_tax_usd_tco2e=50)
    cii = calculate_cii_rating(emissions, 10000, 3600)
    print(f"[Predictor Test] FC: {fc}t, Emissions: {emissions} tCO2e, Cost: ${cost}, CII Rating: {cii}")
