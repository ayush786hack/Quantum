import json
import os
import random
import numpy as np
import pandas as pd

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Vessel Types and Characteristics
VESSEL_TYPES = {
    "Container Ship": {"min_cap": 2000, "max_cap": 18000, "base_speed": 18.0, "eta_engine": 0.45, "unit": "TEU"},
    "Bulk Carrier": {"min_cap": 20000, "max_cap": 180000, "base_speed": 13.0, "eta_engine": 0.42, "unit": "DWT"},
    "Oil Tanker": {"min_cap": 30000, "max_cap": 200000, "base_speed": 14.0, "eta_engine": 0.43, "unit": "DWT"},
    "Ro-Ro Vessel": {"min_cap": 1500, "max_cap": 7000, "base_speed": 16.0, "eta_engine": 0.44, "unit": "CEU"}
}

# Maritime Routes
ROUTES = [
    {"id": "R01", "origin": "Shanghai", "destination": "Rotterdam", "distance_nmi": 10500, "avg_sea_state": 3.2, "shore_power_origin": True, "shore_power_dest": True},
    {"id": "R02", "origin": "Singapore", "destination": "Jawaharlal Nehru (Mumbai)", "distance_nmi": 2400, "avg_sea_state": 2.5, "shore_power_origin": True, "shore_power_dest": False},
    {"id": "R03", "origin": "Hamburg", "destination": "New York", "distance_nmi": 3600, "avg_sea_state": 4.1, "shore_power_origin": True, "shore_power_dest": True},
    {"id": "R04", "origin": "Jebel Ali (Dubai)", "destination": "Colombo", "distance_nmi": 1900, "avg_sea_state": 2.8, "shore_power_origin": False, "shore_power_dest": False},
    {"id": "R05", "origin": "Tokyo", "destination": "Los Angeles", "distance_nmi": 4800, "avg_sea_state": 3.8, "shore_power_origin": True, "shore_power_dest": True},
    {"id": "R06", "origin": "Ningbo-Zhoushan", "destination": "Singapore", "distance_nmi": 2200, "avg_sea_state": 2.9, "shore_power_origin": True, "shore_power_dest": True},
    {"id": "R07", "origin": "Santos", "destination": "Antwerp", "distance_nmi": 5500, "avg_sea_state": 3.5, "shore_power_origin": False, "shore_power_dest": True},
    {"id": "R08", "origin": "Busan", "destination": "Vancouver", "distance_nmi": 4200, "avg_sea_state": 4.0, "shore_power_origin": True, "shore_power_dest": True}
]

# Fuel Specifications (Energy Density MJ/kg, WtW gCO2e/MJ, Cost $/tonne baseline)
FUEL_SPECS = {
    "HFO": {"energy_density_mj_kg": 40.5, "wtw_gco2e_mj": 92.0, "base_price_usd_t": 600},
    "LNG": {"energy_density_mj_kg": 50.0, "wtw_gco2e_mj": 76.0, "base_price_usd_t": 850},
    "Methanol": {"energy_density_mj_kg": 19.9, "wtw_gco2e_mj": 65.0, "base_price_usd_t": 750},
    "Hydrogen": {"energy_density_mj_kg": 120.0, "wtw_gco2e_mj": 10.0, "base_price_usd_t": 2200},
    "Ammonia": {"energy_density_mj_kg": 18.6, "wtw_gco2e_mj": 5.0, "base_price_usd_t": 1100}
}

def generate_vessels(num_vessels=50):
    vessels = []
    np.random.seed(42)
    random.seed(42)
    types_list = list(VESSEL_TYPES.keys())
    
    for i in range(1, num_vessels + 1):
        v_type = types_list[i % len(types_list)]
        meta = VESSEL_TYPES[v_type]
        cap = int(np.random.uniform(meta["min_cap"], meta["max_cap"]))
        built = int(np.random.randint(2005, 2024))
        
        # Engine power kW roughly scaled to capacity
        if v_type == "Container Ship":
            power_kw = 10000 + (cap / 18000.0) * 65000
        elif v_type == "Bulk Carrier":
            power_kw = 8000 + (cap / 180000.0) * 25000
        elif v_type == "Oil Tanker":
            power_kw = 9000 + (cap / 200000.0) * 30000
        else:
            power_kw = 7000 + (cap / 7000.0) * 22000
            
        vessels.append({
            "vessel_id": f"V{i:03d}",
            "name": f"Green Fleet Vessel {i}",
            "vessel_type": v_type,
            "capacity": cap,
            "capacity_unit": meta["unit"],
            "base_speed_knots": meta["base_speed"],
            "engine_power_kw": round(power_kw, 1),
            "built_year": built,
            "supported_fuels": ["HFO", "LNG"] if built < 2018 else ["HFO", "LNG", "Methanol", "Hydrogen", "Ammonia"]
        })
    return vessels

def generate_voyage_dataset(vessels, num_records=2500):
    np.random.seed(42)
    random.seed(42)
    records = []
    
    fuel_types = list(FUEL_SPECS.keys())
    
    for r_idx in range(num_records):
        v = random.choice(vessels)
        route = random.choice(ROUTES)
        fuel = random.choice(v["supported_fuels"])
        
        # Speed deviation around base speed
        speed = round(float(np.random.normal(v["base_speed_knots"], 1.8)), 1)
        speed = max(9.0, min(24.0, speed))
        
        # Weather / Sea State
        sea_state = round(float(np.clip(np.random.normal(route["avg_sea_state"], 1.0), 1.0, 6.0)), 1)
        wind_speed = round(sea_state * 6.5 + np.random.uniform(-3, 3), 1)
        
        # Payload fraction
        payload_pct = round(float(np.random.uniform(0.5, 1.0)), 2)
        
        # Shore power at berth option
        shore_power = route["shore_power_origin"] and route["shore_power_dest"] and random.choice([True, False])
        
        # Physical baseline fuel consumption approximation: FC ~ D * s^3 / (eta * LHV)
        lhv = FUEL_SPECS[fuel]["energy_density_mj_kg"]
        # Base factor based on capacity displacement
        cap_factor = (v["capacity"] / 10000.0) ** 0.65
        
        # Weather penalty: sea state increases resistance quadratic effect
        weather_penalty = 1.0 + 0.04 * (sea_state - 1) ** 1.3
        
        # Payload factor
        load_factor = 0.85 + 0.25 * payload_pct
        
        # Hours sailing
        hours_sailing = route["distance_nmi"] / speed
        
        # Daily fuel consumption (tonnes/day) baseline proportional to (speed/15)^3
        daily_fc_hfo_equiv = 25.0 * cap_factor * ((speed / 15.0) ** 3) * weather_penalty * load_factor
        
        # Convert HFO equivalent energy to chosen fuel tonnes
        # Energy required in MJ = daily_fc_hfo_equiv * 1000 kg * 40.5 MJ/kg
        energy_req_mj_day = daily_fc_hfo_equiv * 1000.0 * FUEL_SPECS["HFO"]["energy_density_mj_kg"]
        
        daily_fc_chosen_tonnes = (energy_req_mj_day / FUEL_SPECS[fuel]["energy_density_mj_kg"]) / 1000.0
        
        total_fuel_tonnes = round(daily_fc_chosen_tonnes * (hours_sailing / 24.0), 2)
        
        # Add random sensor/environmental noise +/- 4%
        noise = np.random.uniform(0.96, 1.04)
        total_fuel_tonnes = round(total_fuel_tonnes * noise, 2)
        
        # Compute lifecycle WtW emissions (tonnes CO2e)
        # Total MJ = total_fuel_tonnes * 1000 kg * LHV_MJ_kg
        total_energy_mj = total_fuel_tonnes * 1000.0 * lhv
        wtw_emissions_tco2e = round((total_energy_mj * FUEL_SPECS[fuel]["wtw_gco2e_mj"]) / 1e6, 2)
        
        # Cost USD
        fuel_cost = total_fuel_tonnes * FUEL_SPECS[fuel]["base_price_usd_t"]
        berth_hours = 24.0
        shore_power_cost = (berth_hours * 150.0 * 0.18) if shore_power else (berth_hours * 0.8 * 600 / 24)
        total_cost_usd = round(fuel_cost + shore_power_cost, 2)
        
        records.append({
            "record_id": f"REC_{r_idx+1:05d}",
            "vessel_id": v["vessel_id"],
            "vessel_type": v["vessel_type"],
            "capacity": v["capacity"],
            "engine_power_kw": v["engine_power_kw"],
            "route_id": route["id"],
            "origin": route["origin"],
            "destination": route["destination"],
            "distance_nmi": route["distance_nmi"],
            "speed_knots": speed,
            "sea_state": sea_state,
            "wind_speed_knots": wind_speed,
            "payload_pct": payload_pct,
            "fuel_type": fuel,
            "shore_power_used": shore_power,
            "hours_sailing": round(hours_sailing, 1),
            "fuel_consumption_tonnes": total_fuel_tonnes,
            "wtw_emissions_tco2e": wtw_emissions_tco2e,
            "total_cost_usd": total_cost_usd
        })
        
    return pd.DataFrame(records)

def run():
    print("[Synthetic Generator] Generating vessel fleet specifications...")
    vessels = generate_vessels(50)
    
    with open(os.path.join(RAW_DIR, "vessels.json"), "w") as f:
        json.dump(vessels, f, indent=2)
        
    with open(os.path.join(RAW_DIR, "routes.json"), "w") as f:
        json.dump(ROUTES, f, indent=2)
        
    with open(os.path.join(RAW_DIR, "fuel_specs.json"), "w") as f:
        json.dump(FUEL_SPECS, f, indent=2)

    print("[Synthetic Generator] Generating 2,500 historical voyage records...")
    df = generate_voyage_dataset(vessels, 2500)
    df.to_csv(os.path.join(PROCESSED_DIR, "voyage_dataset.csv"), index=False)
    print(f"[Synthetic Generator] Done! Dataset saved to {PROCESSED_DIR}/voyage_dataset.csv ({len(df)} rows)")

if __name__ == "__main__":
    run()
