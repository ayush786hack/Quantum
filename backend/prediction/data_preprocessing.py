import os
import pandas as pd
import numpy as np

FUEL_ENERGY_DENSITY = {
    "HFO": 40.5,
    "LNG": 50.0,
    "Methanol": 19.9,
    "Hydrogen": 120.0,
    "Ammonia": 18.6
}

def load_and_preprocess_data(csv_path=None):
    if csv_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, "data", "processed", "voyage_dataset.csv")
        
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Voyage dataset not found at {csv_path}. Run synthetic_generator.py first.")
        
    df = pd.read_csv(csv_path)
    
    # Feature Engineering
    df["speed_cubed"] = df["speed_knots"] ** 3
    df["sea_state_penalty"] = (df["sea_state"] - 1.0) ** 1.3
    df["energy_density"] = df["fuel_type"].map(FUEL_ENERGY_DENSITY)
    df["energy_density_ratio"] = 40.5 / df["energy_density"]
    
    # One-hot encoding
    vessel_type_dummies = pd.get_dummies(df["vessel_type"], prefix="vtype", dtype=float)
    fuel_type_dummies = pd.get_dummies(df["fuel_type"], prefix="fuel", dtype=float)
    
    feature_cols = [
        "capacity", "engine_power_kw", "distance_nmi", "speed_knots", "speed_cubed",
        "sea_state", "sea_state_penalty", "payload_pct", "energy_density_ratio"
    ]
    
    X = pd.concat([df[feature_cols], vessel_type_dummies, fuel_type_dummies], axis=1)
    y = df["fuel_consumption_tonnes"]
    
    return X, y, list(X.columns)

if __name__ == "__main__":
    X, y, cols = load_and_preprocess_data()
    print(f"[Data Preprocessing] Features shape: {X.shape}, Target shape: {y.shape}")
    print(f"[Data Preprocessing] Columns: {cols}")
