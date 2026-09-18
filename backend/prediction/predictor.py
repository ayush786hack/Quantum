import os
import joblib
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_SAVE_PATH = os.path.join(
    BASE_DIR,
    "prediction",
    "fuel_predictor_model.joblib"
)


# ============================================================
# MODEL FEATURES
# ============================================================

PLANNING_FEATURES = [
    "Ship_SpeedOverGround",
    "Ship_DraftAft",
    "Ship_DraftFore",
    "Weather_WindSpeed10M",
    "Weather_WindGusts10M",
    "Weather_WaveHeight",
    "Weather_WavePeriod",
    "Weather_SwellWaveHeight",
    "Weather_SwellWavePeriod",
    "Weather_OceanCurrentVelocity",
    "Weather_Temperature2M"
]


# ============================================================
# FUEL SPECIFICATIONS
# ============================================================

# NOTE:
# These values are currently project/demo assumptions.
# Replace with validated/cited values before final deployment.

FUEL_SPECS = {

    "HFO": {
        "energy_density_mj_kg": 40.5,
        "wtw_gco2e_mj": 92.0,
        "base_price_usd_t": 600
    },

    "LNG": {
        "energy_density_mj_kg": 50.0,
        "wtw_gco2e_mj": 76.0,
        "base_price_usd_t": 850
    },

    "Methanol": {
        "energy_density_mj_kg": 19.9,
        "wtw_gco2e_mj": 65.0,
        "base_price_usd_t": 750
    },

    "Hydrogen": {
        "energy_density_mj_kg": 120.0,
        "wtw_gco2e_mj": 10.0,
        "base_price_usd_t": 2200
    },

    "Ammonia": {
        "energy_density_mj_kg": 18.6,
        "wtw_gco2e_mj": 5.0,
        "base_price_usd_t": 1100
    }
}


# ============================================================
# MODEL CACHE
# ============================================================

_MODEL_CACHE = None


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def get_model_artifact():

    global _MODEL_CACHE

    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    if not os.path.exists(MODEL_SAVE_PATH):

        print(
            f"[Predictor Warning] Model not found at:\n"
            f"{MODEL_SAVE_PATH}"
        )

        return None

    try:

        _MODEL_CACHE = joblib.load(
            MODEL_SAVE_PATH
        )

        print(
            "[Predictor] Model loaded successfully."
        )

        return _MODEL_CACHE

    except Exception as e:

        print(
            f"[Predictor Warning] "
            f"Failed to load model: {e}"
        )

        return None


# ============================================================
# CREATE MODEL INPUT
# ============================================================

def _prediction_features(
    speed_knots: float,
    draft_aft: float,
    draft_fore: float,
    wind_speed: float,
    wind_gusts: float,
    wave_height: float,
    wave_period: float,
    swell_height: float,
    swell_period: float,
    current_velocity: float,
    temperature: float
) -> pd.DataFrame:

    """
    Convert user-friendly inputs into the exact
    feature format expected by the trained XGBoost model.
    """

    input_data = {

        # FuelCast stores speed in m/s.
        # User provides speed in knots.
        "Ship_SpeedOverGround":
            speed_knots * 0.514444,

        "Ship_DraftAft":
            draft_aft,

        "Ship_DraftFore":
            draft_fore,

        "Weather_WindSpeed10M":
            wind_speed,

        "Weather_WindGusts10M":
            wind_gusts,

        "Weather_WaveHeight":
            wave_height,

        "Weather_WavePeriod":
            wave_period,

        "Weather_SwellWaveHeight":
            swell_height,

        "Weather_SwellWavePeriod":
            swell_period,

        "Weather_OceanCurrentVelocity":
            current_velocity,

        "Weather_Temperature2M":
            temperature
    }

    input_df = pd.DataFrame(
        [input_data],
        columns=PLANNING_FEATURES
    )

    return input_df


# ============================================================
# PREDICT FUEL RATE
# ============================================================

def predict_fuel_rate(
    speed_knots: float,
    draft_aft: float,
    draft_fore: float,
    wind_speed: float,
    wind_gusts: float,
    wave_height: float,
    wave_period: float,
    swell_height: float,
    swell_period: float,
    current_velocity: float,
    temperature: float
) -> float:

    """
    Predict main-engine fuel consumption rate.

    Returns:
        Fuel rate in kg/s
    """

    artifact = get_model_artifact()

    if artifact is None:

        raise RuntimeError(
            "Fuel prediction model is not available."
        )

    model = artifact["model"]

    input_df = _prediction_features(

        speed_knots=speed_knots,

        draft_aft=draft_aft,

        draft_fore=draft_fore,

        wind_speed=wind_speed,

        wind_gusts=wind_gusts,

        wave_height=wave_height,

        wave_period=wave_period,

        swell_height=swell_height,

        swell_period=swell_period,

        current_velocity=current_velocity,

        temperature=temperature
    )

    prediction = model.predict(
        input_df
    )

    fuel_rate = float(
        prediction[0]
    )

    # Fuel rate cannot physically be negative.
    fuel_rate = max(
        0.0,
        fuel_rate
    )

    return fuel_rate


# ============================================================
# CALCULATE VOYAGE TIME
# ============================================================

def calculate_voyage_time(
    distance_nmi: float,
    speed_knots: float
) -> float:

    """
    Calculate voyage duration.

    nautical miles / knots = hours
    """

    if speed_knots <= 0:

        raise ValueError(
            "Speed must be greater than 0."
        )

    return (
        distance_nmi / speed_knots
    )


# ============================================================
# CONVERT FUEL RATE TO VOYAGE FUEL
# ============================================================

def calculate_voyage_fuel(
    fuel_rate_kg_per_s: float,
    distance_nmi: float,
    speed_knots: float
) -> float:

    """
    Convert fuel rate (kg/s) into
    total voyage fuel (tonnes).
    """

    voyage_time_hours = (
        calculate_voyage_time(
            distance_nmi,
            speed_knots
        )
    )

    # kg/s × hours × 3600 sec/hour
    # / 1000 kg/tonne
    #
    # = kg/s × hours × 3.6

    fuel_tonnes = (
        fuel_rate_kg_per_s
        * voyage_time_hours
        * 3.6
    )

    return max(
        0.0,
        fuel_tonnes
    )


# ============================================================
# COMPLETE VOYAGE FUEL PREDICTION
# ============================================================

def predict_voyage_fuel(
    distance_nmi: float,
    speed_knots: float,
    draft_aft: float,
    draft_fore: float,
    wind_speed: float,
    wind_gusts: float,
    wave_height: float,
    wave_period: float,
    swell_height: float,
    swell_period: float,
    current_velocity: float,
    temperature: float
) -> dict:

    """
    Predict both fuel rate and total voyage fuel.
    """

    fuel_rate = predict_fuel_rate(

        speed_knots=speed_knots,

        draft_aft=draft_aft,

        draft_fore=draft_fore,

        wind_speed=wind_speed,

        wind_gusts=wind_gusts,

        wave_height=wave_height,

        wave_period=wave_period,

        swell_height=swell_height,

        swell_period=swell_period,

        current_velocity=current_velocity,

        temperature=temperature
    )

    voyage_time_hours = (
        calculate_voyage_time(
            distance_nmi,
            speed_knots
        )
    )

    fuel_tonnes = (
        calculate_voyage_fuel(
            fuel_rate,
            distance_nmi,
            speed_knots
        )
    )

    return {

        "fuel_rate_kg_per_s":
            round(fuel_rate, 6),

        "voyage_time_hours":
            round(voyage_time_hours, 2),

        "fuel_consumption_tonnes":
            round(fuel_tonnes, 3)
    }


# ============================================================
# UNCERTAINTY
# ============================================================

def predict_fuel_consumption_with_uncertainty(
    **kwargs
) -> dict:

    """
    Return fuel prediction with an approximate
    uncertainty band based on tree prediction spread.

    This is a practical demo uncertainty estimate,
    not a statistically calibrated confidence interval.
    """

    artifact = get_model_artifact()

    if artifact is None:

        raise RuntimeError(
            "Fuel prediction model is not available."
        )

    model = artifact["model"]

    input_df = _prediction_features(
        **kwargs
    )

    point = float(
        model.predict(input_df)[0]
    )

    point = max(
        0.0,
        point
    )

    # Pipeline contains:
    # imputer -> XGBRegressor

    xgb_model = model.named_steps["model"]

    tree_predictions = np.asarray([

        tree.predict(
            model.named_steps["imputer"].transform(
                input_df
            )
        )[0]

        for tree in xgb_model.estimators_.ravel()

    ])

    spread = float(
        np.std(tree_predictions)
    )

    # Approximate 90% interval.
    margin = max(
        0.001,
        1.645 * spread
    )

    lower = max(
        0.0,
        point - margin
    )

    upper = (
        point + margin
    )

    return {

        "fuel_rate_kg_per_s":
            round(point, 6),

        "lower_bound_kg_per_s":
            round(lower, 6),

        "upper_bound_kg_per_s":
            round(upper, 6),

        "confidence_level":
            0.90,

        "uncertainty_method":
            "XGBoost tree prediction spread"
    }


# ============================================================
# WTW EMISSIONS
# ============================================================

def calculate_wtw_emissions(
    fuel_consumption_tonnes: float,
    fuel_type: str
) -> float:

    """
    Calculate Well-to-Wake GHG emissions.

    Returns:
        tonnes CO2e
    """

    spec = FUEL_SPECS.get(
        fuel_type,
        FUEL_SPECS["HFO"]
    )

    energy_density = (
        spec["energy_density_mj_kg"]
    )

    wtw_factor = (
        spec["wtw_gco2e_mj"]
    )

    total_energy_mj = (
        fuel_consumption_tonnes
        * 1000.0
        * energy_density
    )

    emissions_tco2e = (
        total_energy_mj
        * wtw_factor
        / 1_000_000
    )

    return max(
        0.0,
        round(
            emissions_tco2e,
            2
        )
    )


# ============================================================
# VOYAGE COST
# ============================================================

def calculate_voyage_cost(
    fuel_consumption_tonnes: float,
    fuel_type: str,
    distance_nmi: float,
    speed_knots: float,
    shore_power_used: bool,
    custom_fuel_prices: dict = None,
    carbon_tax_usd_tco2e: float = 0.0
) -> float:

    """
    Calculate total voyage operational cost.
    """

    prices = (
        custom_fuel_prices
        or {}
    )

    fuel_price = prices.get(

        fuel_type,

        FUEL_SPECS.get(
            fuel_type,
            FUEL_SPECS["HFO"]
        )["base_price_usd_t"]
    )

    fuel_cost = (
        fuel_consumption_tonnes
        * fuel_price
    )

    # Simplified berth/shore-power assumption
    hours_in_port = 24.0

    if shore_power_used:

        shore_power_cost = (
            hours_in_port
            * 120.0
            * 0.18
        )

    else:

        shore_power_cost = (
            0.5
            * prices.get(
                "HFO",
                FUEL_SPECS["HFO"][
                    "base_price_usd_t"
                ]
            )
        )

    wtw_emissions = calculate_wtw_emissions(

        fuel_consumption_tonnes,

        fuel_type
    )

    carbon_tax_cost = (
        wtw_emissions
        * carbon_tax_usd_tco2e
    )

    total_cost = (
        fuel_cost
        + shore_power_cost
        + carbon_tax_cost
    )

    return round(
        total_cost,
        2
    )


# ============================================================
# CII RATING
# ============================================================

def calculate_cii_rating(
    emissions_tco2e: float,
    capacity: float,
    distance_nmi: float
) -> str:

    """
    Simplified CII rating for project demonstration.
    """

    if (
        capacity <= 0
        or distance_nmi <= 0
    ):

        return "C"

    actual_cii = (
        emissions_tco2e
        * 1e6
        / (
            capacity
            * distance_nmi
        )
    )

    # Project/demo reference curve
    reference_cii = (
        12.5
        * (
            capacity ** -0.15
        )
    )

    ratio = (
        actual_cii
        / max(
            0.1,
            reference_cii
        )
    )

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


# ============================================================
# COMPLIANCE FORECAST
# ============================================================

def calculate_compliance_forecast(
    emissions_tco2e: float,
    capacity: float,
    distance_nmi: float,
    speed_knots: float = 16.0
) -> dict:

    """
    Simplified pre-voyage compliance indicator.
    """

    grade = calculate_cii_rating(

        emissions_tco2e,

        capacity,

        distance_nmi
    )

    actual = (
        emissions_tco2e
        * 1e6
        / max(
            1.0,
            capacity * distance_nmi
        )
    )

    reference = (
        12.5
        * (
            max(
                1.0,
                capacity
            ) ** -0.15
        )
    )

    cii_ratio = (
        actual
        / max(
            0.1,
            reference
        )
    )

    eexi_ratio = (
        cii_ratio
        * (
            1.0
            + max(
                0.0,
                speed_knots - 14.0
            )
            * 0.025
        )
    )

    if (
        grade in ["A", "B"]
        and eexi_ratio <= 1.0
    ):

        status = "green"

        message = (
            "Compliant before sailing"
        )

    elif (
        grade == "C"
        and eexi_ratio <= 1.25
    ):

        status = "amber"

        message = (
            "Review speed or fuel before sailing"
        )

    else:

        status = "red"

        message = (
            "Plan is forecast non-compliant"
        )

    return {

        "status":
            status,

        "cii_grade":
            grade,

        "cii_ratio":
            round(
                cii_ratio,
                2
            ),

        "eexi_ratio":
            round(
                eexi_ratio,
                2
            ),

        "message":
            message
    }


# ============================================================
# COMPLETE DECISION FUNCTION
# ============================================================

def evaluate_voyage(
    distance_nmi: float,
    speed_knots: float,
    draft_aft: float,
    draft_fore: float,
    wind_speed: float,
    wind_gusts: float,
    wave_height: float,
    wave_period: float,
    swell_height: float,
    swell_period: float,
    current_velocity: float,
    temperature: float,
    fuel_type: str,
    capacity: float,
    shore_power_used: bool = False,
    carbon_tax_usd_tco2e: float = 0.0
) -> dict:

    """
    Complete voyage evaluation.

    This connects:

        ML prediction
             ↓
        Voyage fuel
             ↓
        Cost
             ↓
        WTW emissions
             ↓
        CII
    """

    prediction = predict_voyage_fuel(

        distance_nmi=distance_nmi,

        speed_knots=speed_knots,

        draft_aft=draft_aft,

        draft_fore=draft_fore,

        wind_speed=wind_speed,

        wind_gusts=wind_gusts,

        wave_height=wave_height,

        wave_period=wave_period,

        swell_height=swell_height,

        swell_period=swell_period,

        current_velocity=current_velocity,

        temperature=temperature
    )

    fuel_tonnes = (
        prediction[
            "fuel_consumption_tonnes"
        ]
    )

    wtw_emissions = (
        calculate_wtw_emissions(

            fuel_tonnes,

            fuel_type
        )
    )

    voyage_cost = (
        calculate_voyage_cost(

            fuel_consumption_tonnes=
                fuel_tonnes,

            fuel_type=
                fuel_type,

            distance_nmi=
                distance_nmi,

            speed_knots=
                speed_knots,

            shore_power_used=
                shore_power_used,

            carbon_tax_usd_tco2e=
                carbon_tax_usd_tco2e
        )
    )

    cii = calculate_compliance_forecast(

        emissions_tco2e=
            wtw_emissions,

        capacity=
            capacity,

        distance_nmi=
            distance_nmi,

        speed_knots=
            speed_knots
    )

    return {

        "fuel_rate_kg_per_s":
            prediction[
                "fuel_rate_kg_per_s"
            ],

        "voyage_time_hours":
            prediction[
                "voyage_time_hours"
            ],

        "fuel_consumption_tonnes":
            fuel_tonnes,

        "fuel_type":
            fuel_type,

        "fuel_cost_and_operations_usd":
            voyage_cost,

        "wtw_emissions_tco2e":
            wtw_emissions,

        "cii":
            cii
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n===================================")
    print("       FUEL PREDICTOR TEST")
    print("===================================\n")


    # Example voyage:
    #
    # Vessel: V001
    # Route: Mumbai -> Dubai
    # Distance: 1050 nmi
    # Speed: 17 knots
    # Draft: 6 m
    # Weather: representative conditions
    # Fuel: LNG

    result = evaluate_voyage(

        distance_nmi=1050,

        speed_knots=17,

        draft_aft=6.0,

        draft_fore=6.0,

        wind_speed=6.0,

        wind_gusts=8.0,

        wave_height=1.0,

        wave_period=5.0,

        swell_height=0.5,

        swell_period=6.0,

        current_velocity=0.5,

        temperature=20.0,

        fuel_type="LNG",

        capacity=12000,

        shore_power_used=False,

        carbon_tax_usd_tco2e=50
    )


    print(
        "\n========== PREDICTION =========="
    )

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )

    print(
        "\n===================================\n"
    )