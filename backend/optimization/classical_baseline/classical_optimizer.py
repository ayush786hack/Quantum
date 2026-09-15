import os
import sys
import numpy as np
import pandas as pd


# ============================================================
# PATH SETUP
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

BACKEND_DIR = os.path.dirname(
    os.path.dirname(CURRENT_DIR)
)

sys.path.append(BACKEND_DIR)


# ============================================================
# IMPORT PREDICTOR
# ============================================================

from prediction.predictor import (
    predict_fuel_rate,
    calculate_voyage_time,
    calculate_voyage_fuel
)


# ============================================================
# DATA PATH
# ============================================================

DATA_DIR = os.path.join(
    BACKEND_DIR,
    "data",
    "processed"
)


# ============================================================
# LOAD DATA
# ============================================================

vessels = pd.read_csv(
    os.path.join(DATA_DIR, "vessels.csv")
)

routes = pd.read_csv(
    os.path.join(DATA_DIR, "routes.csv")
)

voyages = pd.read_csv(
    os.path.join(DATA_DIR, "voyages.csv")
)

fuels = pd.read_csv(
    os.path.join(DATA_DIR, "fuels.csv")
)


# ============================================================
# SYNTHETIC VESSEL CALIBRATION
# ============================================================

REFERENCE_ENGINE_POWER = 22000

VESSEL_POWER_FACTORS = {
    "V001": 22000 / REFERENCE_ENGINE_POWER,
    "V002": 18000 / REFERENCE_ENGINE_POWER,
    "V003": 25000 / REFERENCE_ENGINE_POWER,
    "V004": 15000 / REFERENCE_ENGINE_POWER,
    "V005": 12000 / REFERENCE_ENGINE_POWER,
}


# ============================================================
# FUEL ENERGY FACTORS
# ============================================================

REFERENCE_ENERGY_DENSITY = 40.5

FUEL_ENERGY_FACTORS = {
    "HFO": 40.5 / REFERENCE_ENERGY_DENSITY,
    "LNG": 40.5 / 50.0,
    "Methanol": 40.5 / 19.9,
    "Hydrogen": 40.5 / 120.0,
    "Ammonia": 40.5 / 18.6,
}


# ============================================================
# DEFAULT WEATHER
# ============================================================

DEFAULT_WEATHER = {
    "wind_speed_10m": 8.0,
    "wind_gusts_10m": 12.0,
    "wave_height": 1.0,
    "wave_period": 8.0,
    "swell_wave_height": 0.5,
    "swell_wave_period": 10.0,
    "ocean_current_velocity": 0.5,
    "temperature_2m": 25.0,
}


# ============================================================
# FEASIBILITY CHECK
# ============================================================

def check_feasibility(
    vessel,
    voyage,
    route,
    fuel_type,
    speed
):

    # --------------------------------------------------------
    # Capacity
    # --------------------------------------------------------

    capacity_ok = (
        float(vessel["capacity_tonnes"])
        >= float(voyage["cargo_demand_tonnes"])
    )


    # --------------------------------------------------------
    # Vessel speed
    # --------------------------------------------------------

    vessel_speed_ok = (
        float(vessel["min_speed_knots"])
        <= speed
        <= float(vessel["max_speed_knots"])
    )


    # --------------------------------------------------------
    # Route speed limit
    # --------------------------------------------------------

    route_speed_ok = (
        speed
        <= float(route["speed_limit_knots"])
    )


    # --------------------------------------------------------
    # Voyage time
    # --------------------------------------------------------

    voyage_time = (
        float(route["distance_nmi"])
        / speed
    )


    # --------------------------------------------------------
    # Deadline
    # --------------------------------------------------------

    deadline_ok = (
        voyage_time
        <= float(voyage["deadline_hours"])
    )


    # --------------------------------------------------------
    # Fuel compatibility
    # --------------------------------------------------------

    compatible_fuels = [
        fuel.strip()
        for fuel in str(
            vessel["fuel_compatible"]
        ).split(",")
    ]

    fuel_ok = (
        fuel_type.strip()
        in compatible_fuels
    )


    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    feasible = (
        capacity_ok
        and vessel_speed_ok
        and route_speed_ok
        and deadline_ok
        and fuel_ok
    )


    return {
        "feasible": feasible,
        "capacity_ok": capacity_ok,
        "speed_ok": (
            vessel_speed_ok
            and route_speed_ok
        ),
        "deadline_ok": deadline_ok,
        "fuel_ok": fuel_ok,
        "voyage_time": voyage_time
    }


# ============================================================
# EVALUATE ONE CANDIDATE
# ============================================================

def evaluate_candidate(
    vessel,
    voyage,
    route,
    fuel,
    speed
):

    fuel_type = str(
        fuel["fuel_type"]
    ).strip()


    # --------------------------------------------------------
    # Check feasibility
    # --------------------------------------------------------

    feasibility = check_feasibility(
        vessel,
        voyage,
        route,
        fuel_type,
        speed
    )


    if not feasibility["feasible"]:
        return None


    # --------------------------------------------------------
    # ML FUEL PREDICTION
    # --------------------------------------------------------

    ml_fuel_rate = predict_fuel_rate(
        speed_knots=float(speed),
        draft_aft=8.0,
        draft_fore=8.0,

        wind_speed=DEFAULT_WEATHER[
            "wind_speed_10m"
        ],

        wind_gusts=DEFAULT_WEATHER[
            "wind_gusts_10m"
        ],

        wave_height=DEFAULT_WEATHER[
            "wave_height"
        ],

        wave_period=DEFAULT_WEATHER[
            "wave_period"
        ],

        swell_height=DEFAULT_WEATHER[
            "swell_wave_height"
        ],

        swell_period=DEFAULT_WEATHER[
            "swell_wave_period"
        ],

        current_velocity=DEFAULT_WEATHER[
            "ocean_current_velocity"
        ],

        temperature=DEFAULT_WEATHER[
            "temperature_2m"
        ]
    )


    # --------------------------------------------------------
    # Vessel calibration
    # --------------------------------------------------------

    vessel_id = str(
        vessel["vessel_id"]
    )

    vessel_factor = VESSEL_POWER_FACTORS.get(
        vessel_id,
        1.0
    )


    # --------------------------------------------------------
    # Fuel calibration
    # --------------------------------------------------------

    fuel_factor = FUEL_ENERGY_FACTORS.get(
        fuel_type,
        1.0
    )


    # --------------------------------------------------------
    # Calibrated fuel rate
    # --------------------------------------------------------

    calibrated_fuel_rate = (
        ml_fuel_rate
        * vessel_factor
        * fuel_factor
    )


    # --------------------------------------------------------
    # Voyage time
    # --------------------------------------------------------

    voyage_time = calculate_voyage_time(
        distance_nmi=float(
            route["distance_nmi"]
        ),
        speed_knots=float(speed)
    )


    # --------------------------------------------------------
    # Fuel consumption
    # --------------------------------------------------------

    fuel_tonnes = calculate_voyage_fuel(
        fuel_rate_kg_per_s=float(
            calibrated_fuel_rate
        ),
        distance_nmi=float(
            route["distance_nmi"]
        ),
        speed_knots=float(speed)
    )


    # --------------------------------------------------------
    # Fuel cost
    # --------------------------------------------------------

    fuel_cost = (
        fuel_tonnes
        * float(
            fuel["price_per_tonne"]
        )
    )


    # --------------------------------------------------------
    # Well-to-Wake GHG
    # --------------------------------------------------------

    wtw_ghg_tonnes = (
        fuel_tonnes
        * float(
            fuel["wtw_ghg_kg_per_kg_fuel"]
        )
    )


    # --------------------------------------------------------
    # Return candidate
    # --------------------------------------------------------

    return {

        "vessel_id": vessel_id,

        "vessel_type": vessel[
            "vessel_type"
        ],

        "voyage_id": voyage[
            "voyage_id"
        ],

        "route_id": route[
            "route_id"
        ],

        "fuel_type": fuel_type,

        "speed_knots": float(speed),

        "fuel_rate_kg_per_s": float(
            calibrated_fuel_rate
        ),

        "voyage_time_hours": float(
            voyage_time
        ),

        "fuel_tonnes": float(
            fuel_tonnes
        ),

        "fuel_cost_usd": float(
            fuel_cost
        ),

        "wtw_ghg_tonnes": float(
            wtw_ghg_tonnes
        )
    }



# ============================================================
# VESSEL SCHEDULING
# ============================================================

def schedules_overlap(start_a, end_a, start_b, end_b):
    """Return True when two time intervals overlap."""
    return max(start_a, start_b) < min(end_a, end_b)


def fleet_has_conflict(plan):
    """
    Check whether the same vessel is assigned to overlapping voyages.
    Each plan row must contain start_time_hours and voyage_time_hours.
    """
    if plan.empty:
        return False

    schedules = {}
    for _, row in plan.iterrows():
        vessel_id = str(row["vessel_id"])
        start = float(row["start_time_hours"])
        end = start + float(row["voyage_time_hours"])

        for old_start, old_end in schedules.get(vessel_id, []):
            if schedules_overlap(start, end, old_start, old_end):
                return True

        schedules.setdefault(vessel_id, []).append((start, end))

    return False


def add_schedule_fields(result, voyage):
    """Add start/end schedule information to a candidate result."""
    result = result.copy()
    start = float(voyage.get("start_time_hours", 0.0))
    result["start_time_hours"] = start
    result["end_time_hours"] = start + float(result["voyage_time_hours"])
    return result


# ============================================================
# CLASSICAL OPTIMIZER
# ============================================================

def classical_optimize(top_candidates_per_voyage=40):
    """
    Classical fleet baseline with vessel-availability scheduling.

    The optimizer first performs the deterministic grid search used by the
    original baseline, then combines candidate decisions across voyages and
    rejects fleets where the same vessel is used simultaneously.

    To keep the exhaustive fleet search tractable, the strongest candidates
    by cost and by WTW GHG are retained for each voyage.
    """

    candidates_by_voyage = {}

    # --------------------------------------------------------
    # Generate all feasible per-voyage candidates
    # --------------------------------------------------------
    for _, voyage in voyages.iterrows():

        route_matches = routes[
            routes["route_id"] == voyage["route_id"]
        ]

        if route_matches.empty:
            continue

        route = route_matches.iloc[0]
        voyage_candidates = []

        for _, vessel in vessels.iterrows():

            if (
                float(vessel["capacity_tonnes"])
                < float(voyage["cargo_demand_tonnes"])
            ):
                continue

            compatible_fuels = [
                fuel.strip()
                for fuel in str(vessel["fuel_compatible"]).split(",")
            ]

            for _, fuel in fuels.iterrows():

                fuel_type = str(fuel["fuel_type"]).strip()

                if fuel_type not in compatible_fuels:
                    continue

                min_speed = float(vessel["min_speed_knots"])
                max_speed = min(
                    float(vessel["max_speed_knots"]),
                    float(route["speed_limit_knots"])
                )

                speeds = np.arange(
                    min_speed,
                    max_speed + 0.001,
                    0.5
                )

                for speed in speeds:

                    result = evaluate_candidate(
                        vessel=vessel,
                        voyage=voyage,
                        route=route,
                        fuel=fuel,
                        speed=float(speed)
                    )

                    if result is not None:
                        result = add_schedule_fields(result, voyage)
                        voyage_candidates.append(result)

        if voyage_candidates:
            candidate_df = pd.DataFrame(voyage_candidates)

            # Retain strong candidates from BOTH objectives.
            n = min(top_candidates_per_voyage, len(candidate_df))

            cost_part = candidate_df.nsmallest(
                n, "fuel_cost_usd"
            )

            ghg_part = candidate_df.nsmallest(
                n, "wtw_ghg_tonnes"
            )

            candidate_df = pd.concat(
                [cost_part, ghg_part],
                ignore_index=True
            ).drop_duplicates(
                subset=[
                    "voyage_id",
                    "vessel_id",
                    "fuel_type",
                    "speed_knots"
                ]
            )

            candidates_by_voyage[str(voyage["voyage_id"])] = (
                candidate_df
            )

    if len(candidates_by_voyage) != len(voyages):
        return pd.DataFrame()

    # --------------------------------------------------------
    # Fleet-level search with scheduling constraint
    # --------------------------------------------------------
    voyage_order = [
        str(v["voyage_id"])
        for _, v in voyages.iterrows()
    ]

    feasible_fleets = []

    def backtrack(index, selected):
        if index == len(voyage_order):
            fleet = pd.DataFrame(selected).copy()

            if not fleet_has_conflict(fleet):
                feasible_fleets.append({
                    "plan": fleet,
                    "cost": float(fleet["fuel_cost_usd"].sum()),
                    "ghg": float(fleet["wtw_ghg_tonnes"].sum())
                })
            return

        voyage_id = voyage_order[index]
        current_candidates = candidates_by_voyage[voyage_id]

        for _, candidate in current_candidates.iterrows():

            vessel_id = str(candidate["vessel_id"])
            start = float(candidate["start_time_hours"])
            end = float(candidate["end_time_hours"])

            conflict = False

            for old in selected:
                if str(old["vessel_id"]) != vessel_id:
                    continue

                old_start = float(old["start_time_hours"])
                old_end = float(old["end_time_hours"])

                if schedules_overlap(
                    start, end, old_start, old_end
                ):
                    conflict = True
                    break

            if not conflict:
                selected.append(candidate.to_dict())
                backtrack(index + 1, selected)
                selected.pop()

    backtrack(0, [])

    if not feasible_fleets:
        return pd.DataFrame()

    # --------------------------------------------------------
    # Select the classical baseline plan by minimum total cost
    # --------------------------------------------------------
    best_cost_fleet = min(
        feasible_fleets,
        key=lambda x: x["cost"]
    )

    return best_cost_fleet["plan"].reset_index(drop=True)


# ============================================================
# FLEET-LEVEL PARETO SUMMARY
# ============================================================

def get_fleet_objectives(plan):
    """Return total cost and total WTW GHG for a fleet plan."""
    return (
        float(plan["fuel_cost_usd"].sum()),
        float(plan["wtw_ghg_tonnes"].sum())
    )



# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("      CLASSICAL FLEET OPTIMIZER")
    print("=" * 60)
    print("Scheduling constraints: ENABLED")
    print("Objective: Minimum total fleet fuel cost")
    print("=" * 60)

    results = classical_optimize()

    if results.empty:
        print("\nNo feasible fleet schedule found.")
        print(
            "Check vessel availability, start_time_hours, "
            "capacity, speed limits and deadlines."
        )

    else:
        total_cost = float(results["fuel_cost_usd"].sum())
        total_ghg = float(results["wtw_ghg_tonnes"].sum())

        print("\n" + "-" * 60)
        print("BEST COST FEASIBLE FLEET PLAN")
        print("-" * 60)

        print(f"Total Cost : ${total_cost:,.2f}")
        print(f"Total WTW GHG : {total_ghg:,.3f} tonnes CO2e")

        display_columns = [
            "voyage_id",
            "vessel_id",
            "vessel_type",
            "fuel_type",
            "speed_knots",
            "start_time_hours",
            "end_time_hours",
            "voyage_time_hours",
            "fuel_tonnes",
            "fuel_cost_usd",
            "wtw_ghg_tonnes"
        ]

        print("\nVoyage-wise plan:")
        print(
            results[display_columns].to_string(index=False)
        )

        # ----------------------------------------------------
        # Verify schedule one final time
        # ----------------------------------------------------
        conflict = fleet_has_conflict(results)

        print("\n" + "-" * 60)
        print("SCHEDULE VALIDATION")
        print("-" * 60)

        if conflict:
            print("WARNING: Vessel scheduling conflict detected.")
        else:
            print("✓ No overlapping vessel assignments detected.")

        # ----------------------------------------------------
        # Save the selected fleet plan
        # ----------------------------------------------------
        output_path = os.path.join(
            CURRENT_DIR,
            "classical_results.csv"
        )

        results.to_csv(
            output_path,
            index=False
        )

        print("\nResults saved to:")
        print(output_path)
