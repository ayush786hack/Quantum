import os
import sys
import numpy as np
import pandas as pd


# ============================================================
# PATH SETUP
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

BACKEND_DIR = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..")
)

if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)


# ============================================================
# IMPORTS
# ============================================================

from prediction.predictor import (
    predict_fuel_rate,
    calculate_voyage_time,
    calculate_voyage_fuel
)

from optimization.quantum_inspired.encoding2 import (
    encode_decision,
    decode_particle
)


# ============================================================
# DATA PATHS
# ============================================================

DATA_DIR = os.path.join(
    BACKEND_DIR,
    "data",
    "processed"
)

VESSELS_FILE = os.path.join(
    DATA_DIR,
    "vessels.csv"
)

ROUTES_FILE = os.path.join(
    DATA_DIR,
    "routes.csv"
)

VOYAGES_FILE = os.path.join(
    DATA_DIR,
    "voyages.csv"
)

FUELS_FILE = os.path.join(
    DATA_DIR,
    "fuels.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

vessels_df = pd.read_csv(VESSELS_FILE)
routes_df = pd.read_csv(ROUTES_FILE)
voyages_df = pd.read_csv(VOYAGES_FILE)
fuels_df = pd.read_csv(FUELS_FILE)


# ============================================================
# SYNTHETIC CALIBRATION
# ============================================================

VESSEL_POWER_FACTORS = {
    "V001": 1.0,
    "V002": 0.8181818182,
    "V003": 1.1363636364,
    "V004": 0.6818181818,
    "V005": 0.5454545455
}


FUEL_ENERGY_FACTORS = {
    "HFO": 1.0,
    "LNG": 0.81,
    "Methanol": 2.0351758794,
    "Hydrogen": 0.3375,
    "Ammonia": 2.1774193548
}


# ============================================================
# MOQPSO PARAMETERS
# ============================================================

N_PARTICLES = 20

N_ITERATIONS = 50

BETA_START = 1.0

BETA_END = 0.5

ARCHIVE_SIZE = 30

RANDOM_SEED = 42


# ============================================================
# GET ROUTE
# ============================================================

def get_route(voyage):

    matches = routes_df[
        routes_df["route_id"]
        == voyage["route_id"]
    ]

    if matches.empty:

        raise ValueError(
            f"Route not found: "
            f"{voyage['route_id']}"
        )

    return matches.iloc[0]


# ============================================================
# GET COMPATIBLE FUELS
# ============================================================

def get_compatible_fuels(vessel):

    fuels = str(
        vessel["fuel_compatible"]
    ).split(",")

    return [
        fuel.strip()
        for fuel in fuels
    ]
    
# ============================================================
# VESSEL AVAILABILITY / SCHEDULING CONSTRAINT
# ============================================================

def get_voyage_start_hours(voyage):
    """Return voyage start time from voyages.csv."""
    if "start_time_hours" not in voyages_df.columns:
        raise ValueError(
            "voyages.csv must contain 'start_time_hours' "
            "for vessel availability constraints."
        )

    return float(voyage["start_time_hours"])


def get_voyage_end_hours(voyage, speed):
    """Calculate voyage end time from start time and speed."""
    route = get_route(voyage)
    distance_nmi = float(route["distance_nmi"])

    start_time = get_voyage_start_hours(voyage)
    voyage_time = calculate_voyage_time(distance_nmi, speed)

    return start_time + voyage_time


def schedules_overlap(start_a, end_a, start_b, end_b):
    """Return True when two time intervals overlap."""
    return (
        start_a < end_b
        and start_b < end_a
    )


def decision_conflicts_with_schedules(
    voyage,
    vessel_id,
    speed,
    schedules
):
    """
    Check whether a proposed vessel assignment overlaps with
    already scheduled voyages using the same vessel.
    """
    start = get_voyage_start_hours(voyage)
    end = get_voyage_end_hours(voyage, speed)

    for schedule in schedules:
        if schedule["vessel_id"] != vessel_id:
            continue

        if schedules_overlap(
            start,
            end,
            schedule["start"],
            schedule["end"]
        ):
            return True

    return False


def has_vessel_conflict(results):
    """
    Final safeguard: check whether the same vessel is assigned
    to overlapping voyages in a complete evaluated particle.
    """
    schedules = []

    for result in results:
        voyage_id = result["voyage_id"]
        vessel_id = result["vessel_id"]

        voyage_matches = voyages_df[
            voyages_df["voyage_id"] == voyage_id
        ]

        if voyage_matches.empty:
            return True

        voyage = voyage_matches.iloc[0]

        start_time = get_voyage_start_hours(voyage)

        # evaluate_decision stores this as 'voyage_time'
        duration = float(result["voyage_time"])

        end_time = start_time + duration

        schedules.append({
            "voyage_id": voyage_id,
            "vessel_id": vessel_id,
            "start": start_time,
            "end": end_time
        })

    for i in range(len(schedules)):
        for j in range(i + 1, len(schedules)):
            a = schedules[i]
            b = schedules[j]

            if a["vessel_id"] != b["vessel_id"]:
                continue

            if schedules_overlap(
                a["start"],
                a["end"],
                b["start"],
                b["end"]
            ):
                return True

    return False


def create_non_conflicting_decision(
    voyage,
    rng,
    occupied_schedules
):
    """
    Create a feasible decision that does not conflict with
    already assigned vessels in the current particle.
    """
    candidates = []

    for _, vessel in vessels_df.iterrows():

        if (
            float(vessel["capacity_tonnes"])
            < float(voyage["cargo_demand_tonnes"])
        ):
            continue

        speed_range = get_feasible_speed_range(
            voyage,
            vessel
        )

        if speed_range is None:
            continue

        min_speed, max_speed = speed_range

        for fuel in get_compatible_fuels(vessel):

            if fuel not in FUEL_ENERGY_FACTORS:
                continue

            # Try several deterministic speed points plus random
            # sampling so the candidate has a chance to fit the
            # vessel availability window.
            speed_candidates = [
                min_speed,
                max_speed,
                (min_speed + max_speed) / 2.0
            ]

            for speed in speed_candidates:
                if not decision_conflicts_with_schedules(
                    voyage,
                    vessel["vessel_id"],
                    speed,
                    occupied_schedules
                ):
                    candidates.append(
                        (
                            vessel["vessel_id"],
                            fuel,
                            min_speed,
                            max_speed
                        )
                    )
                    break

    if not candidates:
        raise ValueError(
            f"No schedule-feasible decision for "
            f"{voyage['voyage_id']}. "
            f"Check voyage start times and vessel availability."
        )

    candidate = candidates[
        rng.integers(len(candidates))
    ]

    vessel_id = candidate[0]
    fuel = candidate[1]
    min_speed = candidate[2]
    max_speed = candidate[3]

    # Sample speed until it does not create a conflict.
    for _ in range(20):
        speed = rng.uniform(
            min_speed,
            max_speed
        )

        if not decision_conflicts_with_schedules(
            voyage,
            vessel_id,
            speed,
            occupied_schedules
        ):
            return encode_decision(
                vessel_id,
                fuel,
                speed
            )

    # Fallback to midpoint if random samples conflict.
    speed = (min_speed + max_speed) / 2.0

    return encode_decision(
        vessel_id,
        fuel,
        speed
    )

# ============================================================
# FEASIBILITY CHECK

# ============================================================
# FEASIBILITY CHECK
# ============================================================

def is_feasible(
    voyage,
    vessel,
    fuel,
    speed
):

    route = get_route(voyage)

    # Capacity
    if (
        float(vessel["capacity_tonnes"])
        <
        float(voyage["cargo_demand_tonnes"])
    ):

        return False


    # Vessel minimum speed
    if speed < float(
        vessel["min_speed_knots"]
    ):

        return False


    # Vessel maximum speed
    if speed > float(
        vessel["max_speed_knots"]
    ):

        return False


    # Route speed limit
    if speed > float(
        route["speed_limit_knots"]
    ):

        return False


    # Fuel compatibility
    if fuel not in get_compatible_fuels(
        vessel
    ):

        return False


    # Deadline
    voyage_time = calculate_voyage_time(
        float(route["distance_nmi"]),
        speed
    )

    if (
        voyage_time
        >
        float(voyage["deadline_hours"])
    ):

        return False


    return True


# ============================================================
# FEASIBLE SPEED RANGE
# ============================================================

def get_feasible_speed_range(
    voyage,
    vessel
):

    route = get_route(voyage)

    vessel_min = float(
        vessel["min_speed_knots"]
    )

    vessel_max = float(
        vessel["max_speed_knots"]
    )

    route_max = float(
        route["speed_limit_knots"]
    )

    max_speed = min(
        vessel_max,
        route_max
    )

    required_speed = (
        float(route["distance_nmi"])
        /
        float(voyage["deadline_hours"])
    )

    min_speed = max(
        vessel_min,
        required_speed
    )

    if min_speed > max_speed:

        return None

    return (
        min_speed,
        max_speed
    )


# ============================================================
# CREATE FEASIBLE DECISION
# ============================================================

def create_feasible_decision(
    voyage,
    rng
):

    candidates = []

    for _, vessel in vessels_df.iterrows():

        # Capacity
        if (
            float(vessel["capacity_tonnes"])
            <
            float(voyage["cargo_demand_tonnes"])
        ):

            continue


        # Speed
        speed_range = (
            get_feasible_speed_range(
                voyage,
                vessel
            )
        )

        if speed_range is None:

            continue


        min_speed, max_speed = (
            speed_range
        )


        # Compatible fuels
        compatible_fuels = (
            get_compatible_fuels(
                vessel
            )
        )

        for fuel in compatible_fuels:

            if fuel not in (
                FUEL_ENERGY_FACTORS
            ):

                continue

            candidates.append(
                (
                    vessel["vessel_id"],
                    fuel,
                    min_speed,
                    max_speed
                )
            )


    if not candidates:

        raise ValueError(
            f"No feasible solution for "
            f"{voyage['voyage_id']}"
        )


    candidate = candidates[
        rng.integers(
            len(candidates)
        )
    ]


    vessel_id = candidate[0]

    fuel = candidate[1]

    min_speed = candidate[2]

    max_speed = candidate[3]


    speed = rng.uniform(
        min_speed,
        max_speed
    )


    return encode_decision(
        vessel_id,
        fuel,
        speed
    )


# ============================================================
# CREATE FEASIBLE PARTICLE
# ============================================================

def create_feasible_particle(rng):

    particle = []
    occupied_schedules = []

    for _, voyage in voyages_df.iterrows():

        decision = create_non_conflicting_decision(
            voyage,
            rng,
            occupied_schedules
        )

        particle.append(decision)

        vessel_id = int(round(decision[0]))
        vessel = vessels_df.iloc[vessel_id]

        speed = float(decision[2])

        start_time = get_voyage_start_hours(voyage)
        end_time = get_voyage_end_hours(
            voyage,
            speed
        )

        occupied_schedules.append({
            "voyage_id": voyage["voyage_id"],
            "vessel_id": vessel["vessel_id"],
            "start": start_time,
            "end": end_time
        })

    return np.array(
        particle,
        dtype=float
    )


# ============================================================
# EVALUATE ONE DECISION

# ============================================================
# EVALUATE ONE DECISION
# ============================================================

def evaluate_decision(
    voyage,
    vessel,
    fuel,
    speed
):

    if not is_feasible(
        voyage,
        vessel,
        fuel,
        speed
    ):

        return None


    # Route
    route = get_route(
        voyage
    )

    distance_nmi = float(
        route["distance_nmi"]
    )


    # --------------------------------------------------------
    # TEMPORARY DRAFT ASSUMPTION
    # --------------------------------------------------------

    draft_aft = 8.0

    draft_fore = 8.0


    # --------------------------------------------------------
    # WEATHER SCENARIO
    # --------------------------------------------------------

    wind_speed = 8.0

    wind_gusts = 12.0

    wave_height = 1.0

    wave_period = 6.0

    swell_height = 0.5

    swell_period = 8.0

    current_velocity = 0.2

    temperature = 25.0


    # --------------------------------------------------------
    # ML FUEL PREDICTION
    # --------------------------------------------------------

    fuel_rate = predict_fuel_rate(

        speed_knots=speed,

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


    # --------------------------------------------------------
    # VESSEL CALIBRATION
    # --------------------------------------------------------

    vessel_id = vessel[
        "vessel_id"
    ]

    vessel_factor = (
        VESSEL_POWER_FACTORS.get(
            vessel_id,
            1.0
        )
    )


    # --------------------------------------------------------
    # FUEL ENERGY ADJUSTMENT
    # --------------------------------------------------------

    fuel_factor = (
        FUEL_ENERGY_FACTORS.get(
            fuel,
            1.0
        )
    )


    fuel_rate *= (
        vessel_factor
        *
        fuel_factor
    )


    # --------------------------------------------------------
    # VOYAGE TIME
    # --------------------------------------------------------

    voyage_time = calculate_voyage_time(
        distance_nmi,
        speed
    )


    # --------------------------------------------------------
    # FUEL CONSUMPTION
    # --------------------------------------------------------

    fuel_tonnes = calculate_voyage_fuel(
        fuel_rate,
        distance_nmi,
        speed
    )


    # --------------------------------------------------------
    # FUEL INFORMATION
    # --------------------------------------------------------

    fuel_matches = fuels_df[
        fuels_df["fuel_type"]
        == fuel
    ]

    if fuel_matches.empty:

        raise ValueError(
            f"Fuel not found: {fuel}"
        )


    fuel_row = fuel_matches.iloc[0]


    fuel_price = float(
        fuel_row[
            "price_per_tonne"
        ]
    )


    wtw_factor = float(
        fuel_row[
            "wtw_ghg_kg_per_kg_fuel"
        ]
    )


    # --------------------------------------------------------
    # COST
    # --------------------------------------------------------

    fuel_cost = (
        fuel_tonnes
        *
        fuel_price
    )


    # --------------------------------------------------------
    # WTW GHG
    # --------------------------------------------------------

    wtw_ghg = (
        fuel_tonnes
        *
        wtw_factor
    )


    start_time = get_voyage_start_hours(voyage)
    end_time = start_time + voyage_time

    return {

        "fuel_rate":
            fuel_rate,

        "voyage_time":
            voyage_time,

        "start_time_hours":
            start_time,

        "end_time_hours":
            end_time,

        "fuel_tonnes":
            fuel_tonnes,

        "fuel_cost":
            fuel_cost,

        "wtw_ghg":
            wtw_ghg
    }


# ============================================================
# EVALUATE PARTICLE
# ============================================================

def evaluate_particle(
    particle
):

    decisions = decode_particle(
        particle
    )

    total_cost = 0.0

    total_ghg = 0.0

    decoded_results = []


    for i, decision in enumerate(
        decisions
    ):

        voyage = voyages_df.iloc[i]


        vessel_id = decision[
            "vessel_id"
        ]

        fuel = decision[
            "fuel_type"
        ]

        speed = decision[
            "speed_knots"
        ]


        vessel_matches = vessels_df[
            vessels_df["vessel_id"]
            == vessel_id
        ]


        if vessel_matches.empty:

            return (
                np.inf,
                np.inf,
                None
            )


        vessel = (
            vessel_matches.iloc[0]
        )


        result = evaluate_decision(
            voyage,
            vessel,
            fuel,
            speed
        )


        if result is None:

            return (
                np.inf,
                np.inf,
                None
            )


        total_cost += (
            result["fuel_cost"]
        )


        total_ghg += (
            result["wtw_ghg"]
        )


        decoded_results.append({

            "voyage_id":
                voyage["voyage_id"],

            "vessel_id":
                vessel_id,

            "fuel_type":
                fuel,

            "speed":
                speed,

            **result
        })


    # Final fleet-level scheduling safeguard
    if has_vessel_conflict(decoded_results):
        return (
            np.inf,
            np.inf,
            None
        )


    return (
        total_cost,
        total_ghg,
        decoded_results
    )


# ============================================================
# PARETO DOMINANCE
# ============================================================

def dominates(
    objective_a,
    objective_b
):

    """
    Both objectives are minimized.

    A dominates B when:
    - A is no worse in either objective
    - A is strictly better in at least one
    """

    return (

        objective_a[0]
        <=
        objective_b[0]

        and

        objective_a[1]
        <=
        objective_b[1]

        and

        (
            objective_a[0]
            <
            objective_b[0]

            or

            objective_a[1]
            <
            objective_b[1]
        )
    )


# ============================================================
# UPDATE PARETO ARCHIVE
# ============================================================

def update_archive(
    archive,
    particle,
    cost,
    ghg,
    results
):

    if (
        not np.isfinite(cost)
        or
        not np.isfinite(ghg)
    ):

        return archive


    candidate_objective = (
        cost,
        ghg
    )


    # --------------------------------------------------------
    # Check whether candidate is dominated
    # --------------------------------------------------------

    for entry in archive:

        existing_objective = (
            entry["cost"],
            entry["ghg"]
        )

        if dominates(
            existing_objective,
            candidate_objective
        ):

            return archive


    # --------------------------------------------------------
    # Remove solutions dominated by candidate
    # --------------------------------------------------------

    new_archive = []

    for entry in archive:

        existing_objective = (
            entry["cost"],
            entry["ghg"]
        )

        if not dominates(
            candidate_objective,
            existing_objective
        ):

            new_archive.append(
                entry
            )


    # --------------------------------------------------------
    # Add candidate
    # --------------------------------------------------------

    new_archive.append({

        "particle":
            particle.copy(),

        "cost":
            cost,

        "ghg":
            ghg,

        "results":
            results
    })


    return new_archive


# ============================================================
# CROWDING DISTANCE
# ============================================================

def crowding_distance(
    archive
):

    n = len(archive)

    if n == 0:

        return []


    if n <= 2:

        return [
            np.inf
            for _ in range(n)
        ]


    costs = np.array([
        entry["cost"]
        for entry in archive
    ])


    ghgs = np.array([
        entry["ghg"]
        for entry in archive
    ])


    distances = np.zeros(n)


    for values in (
        costs,
        ghgs
    ):

        order = np.argsort(
            values
        )


        distances[
            order[0]
        ] = np.inf


        distances[
            order[-1]
        ] = np.inf


        value_range = (
            values[order[-1]]
            -
            values[order[0]]
        )


        if value_range == 0:

            continue


        for j in range(
            1,
            n - 1
        ):

            if np.isinf(
                distances[order[j]]
            ):

                continue


            distances[
                order[j]
            ] += (

                values[
                    order[j + 1]
                ]

                -

                values[
                    order[j - 1]
                ]

            ) / value_range


    return distances.tolist()


# ============================================================
# TRUNCATE ARCHIVE
# ============================================================

def truncate_archive(
    archive,
    max_size
):

    if len(archive) <= max_size:

        return archive


    distances = crowding_distance(
        archive
    )


    order = np.argsort(
        np.array(
            distances
        )
    )[::-1]


    return [
        archive[i]
        for i in order[
            :max_size
        ]
    ]


# ============================================================
# SELECT PARETO LEADER
# ============================================================

def select_leader(
    archive,
    rng
):

    if not archive:

        raise ValueError(
            "Pareto archive is empty."
        )


    distances = np.array(
        crowding_distance(
            archive
        ),
        dtype=float
    )


    # Random leader if all are boundary points
    if np.all(
        np.isinf(distances)
    ):

        index = rng.integers(
            len(archive)
        )

        return archive[index]


    finite = np.isfinite(
        distances
    )


    probabilities = np.zeros(
        len(archive)
    )


    if np.any(finite):

        probabilities[finite] = (
            distances[finite]
            + 1e-12
        )


        max_finite = np.max(
            probabilities[finite]
        )


        probabilities[
            ~finite
        ] = max_finite * 2.0


        total = probabilities.sum()


        if total > 0:

            probabilities /= total


            index = rng.choice(
                len(archive),
                p=probabilities
            )


            return archive[index]


    index = rng.integers(
        len(archive)
    )

    return archive[index]


# ============================================================
# QUANTUM-INSPIRED UPDATE
# ============================================================

def quantum_update(
    particle,
    personal_best,
    global_best,
    beta
):

    phi = np.random.rand(
        *particle.shape
    )


    local_attractor = (

        phi
        *
        personal_best

        +

        (1.0 - phi)
        *
        global_best
    )


    direction = np.where(

        np.random.rand(
            *particle.shape
        ) < 0.5,

        -1.0,

        1.0
    )


    u = np.random.uniform(

        1e-10,

        1.0,

        size=particle.shape
    )


    new_particle = (

        local_attractor

        +

        direction

        *

        beta

        *

        np.abs(
            local_attractor
            -
            particle
        )

        *

        np.log(
            1.0 / u
        )
    )


    return new_particle


# ============================================================
# REPAIR PARTICLE
# ============================================================

def repair_particle(
    particle,
    rng
):

    particle = np.asarray(
        particle,
        dtype=float
    ).copy()

    repaired = []
    occupied_schedules = []

    for i, voyage in enumerate(
        voyages_df.itertuples(index=False)
    ):

        voyage_series = voyages_df.iloc[i]
        candidate = particle[i]

        vessel_index = int(
            np.round(candidate[0])
        )

        fuel_index = int(
            np.round(candidate[1])
        )

        speed = float(candidate[2])

        vessel_index = int(
            np.clip(
                vessel_index,
                0,
                len(vessels_df) - 1
            )
        )

        fuel_index = int(
            np.clip(
                fuel_index,
                0,
                len(fuels_df) - 1
            )
        )

        vessel = vessels_df.iloc[vessel_index]
        fuel = fuels_df.iloc[fuel_index]["fuel_type"]

        speed_range = get_feasible_speed_range(
            voyage_series,
            vessel
        )

        compatible_fuels = get_compatible_fuels(
            vessel
        )

        current_valid = (
            speed_range is not None
            and fuel in compatible_fuels
            and fuel in FUEL_ENERGY_FACTORS
        )

        if current_valid:
            min_speed, max_speed = speed_range
            speed = np.clip(
                speed,
                min_speed,
                max_speed
            )

            conflict = decision_conflicts_with_schedules(
                voyage_series,
                vessel["vessel_id"],
                speed,
                occupied_schedules
            )
        else:
            conflict = True

        if not current_valid or conflict:

            decision = create_non_conflicting_decision(
                voyage_series,
                rng,
                occupied_schedules
            )

        else:
            decision = [
                float(vessel_index),
                float(fuel_index),
                float(speed)
            ]

        repaired.append(decision)

        selected_vessel_index = int(round(decision[0]))
        selected_vessel = vessels_df.iloc[
            selected_vessel_index
        ]

        selected_speed = float(decision[2])

        start_time = get_voyage_start_hours(
            voyage_series
        )

        end_time = get_voyage_end_hours(
            voyage_series,
            selected_speed
        )

        occupied_schedules.append({
            "voyage_id": voyage_series["voyage_id"],
            "vessel_id": selected_vessel["vessel_id"],
            "start": start_time,
            "end": end_time
        })

    return np.array(
        repaired,
        dtype=float
    )


# ============================================================
# PERSONAL BEST

# ============================================================
# PERSONAL BEST
# ============================================================

def choose_personal_best(
    current_particle,
    current_objective,
    best_particle,
    best_objective,
    rng
):

    # Current dominates old best
    if dominates(
        current_objective,
        best_objective
    ):

        return (
            current_particle.copy(),
            current_objective
        )


    # Old best dominates current
    if dominates(
        best_objective,
        current_objective
    ):

        return (
            best_particle.copy(),
            best_objective
        )


    # Both are non-dominated
    # Randomly keep one for diversity

    if rng.random() < 0.5:

        return (
            current_particle.copy(),
            current_objective
        )


    return (
        best_particle.copy(),
        best_objective
    )


# ============================================================
# SAVE PARETO FRONT
# ============================================================

def save_pareto_front(
    archive
):

    output_file = os.path.join(

        CURRENT_DIR,

        "moqpso_pareto_front.csv"
    )


    rows = []


    for rank, entry in enumerate(
        archive,
        start=1
    ):

        row = {

            "pareto_rank":
                rank,

            "total_cost":
                entry["cost"],

            "total_wtw_ghg":
                entry["ghg"]
        }


        for result in (
            entry["results"]
        ):

            voyage_id = (
                result["voyage_id"]
            )


            row[
                f"{voyage_id}_vessel"
            ] = result[
                "vessel_id"
            ]


            row[
                f"{voyage_id}_fuel"
            ] = result[
                "fuel_type"
            ]


            row[
                f"{voyage_id}_speed"
            ] = result[
                "speed"
            ]


            row[
                f"{voyage_id}_fuel_tonnes"
            ] = result[
                "fuel_tonnes"
            ]


        rows.append(row)


    pd.DataFrame(
        rows
    ).to_csv(

        output_file,

        index=False
    )


    return output_file


# ============================================================
# MAIN MOQPSO
# ============================================================

def run_moqpso():

    rng = np.random.default_rng(
        RANDOM_SEED
    )


    np.random.seed(
        RANDOM_SEED
    )


    print("=" * 65)

    print(
        "MULTI-OBJECTIVE "
        "QUANTUM-INSPIRED PSO"
    )

    print("=" * 65)


    # --------------------------------------------------------
    # INITIALIZE POPULATION
    # --------------------------------------------------------

    particles = [

        create_feasible_particle(
            rng
        )

        for _ in range(
            N_PARTICLES
        )
    ]


    # --------------------------------------------------------
    # PERSONAL BEST STORAGE
    # --------------------------------------------------------

    personal_best = []

    personal_objectives = []


    # --------------------------------------------------------
    # PARETO ARCHIVE
    # --------------------------------------------------------

    archive = []


    # --------------------------------------------------------
    # INITIAL EVALUATION
    # --------------------------------------------------------

    for particle in particles:

        cost, ghg, results = (
            evaluate_particle(
                particle
            )
        )


        if results is None:

            continue


        objective = (
            cost,
            ghg
        )


        personal_best.append(
            particle.copy()
        )


        personal_objectives.append(
            objective
        )


        archive = update_archive(

            archive,

            particle,

            cost,

            ghg,

            results
        )


    archive = truncate_archive(

        archive,

        ARCHIVE_SIZE
    )


    if not archive:

        raise RuntimeError(
            "No feasible solution found."
        )


    print(
        f"Initial Pareto archive size: "
        f"{len(archive)}"
    )


    # ========================================================
    # ITERATIONS
    # ========================================================

    for iteration in range(
        N_ITERATIONS
    ):


        # Decreasing beta

        beta = (

            BETA_START

            -

            (
                BETA_START
                -
                BETA_END
            )

            *

            iteration

            /

            max(
                N_ITERATIONS - 1,
                1
            )
        )


        # ----------------------------------------------------
        # UPDATE EACH PARTICLE
        # ----------------------------------------------------

        for i in range(
            len(personal_best)
        ):


            # Select leader from Pareto archive

            leader = select_leader(

                archive,

                rng
            )


            # Quantum-inspired movement

            particles[i] = (
                quantum_update(

                    particles[i],

                    personal_best[i],

                    leader["particle"],

                    beta
                )
            )


            # Repair invalid values

            particles[i] = (
                repair_particle(

                    particles[i],

                    rng
                )
            )


            # Evaluate

            cost, ghg, results = (
                evaluate_particle(
                    particles[i]
                )
            )


            if results is None:

                continue


            current_objective = (
                cost,
                ghg
            )


            # ------------------------------------------------
            # PERSONAL BEST
            # ------------------------------------------------

            (
                personal_best[i],

                personal_objectives[i]

            ) = choose_personal_best(

                particles[i],

                current_objective,

                personal_best[i],

                personal_objectives[i],

                rng
            )


            # ------------------------------------------------
            # PARETO ARCHIVE
            # ------------------------------------------------

            archive = update_archive(

                archive,

                particles[i],

                cost,

                ghg,

                results
            )


            archive = truncate_archive(

                archive,

                ARCHIVE_SIZE
            )


        print(

            f"Iteration "
            f"{iteration + 1:02d}/"
            f"{N_ITERATIONS}"

            f" | Pareto solutions: "
            f"{len(archive):02d}"
        )


    # ========================================================
    # FINAL PARETO FRONT
    # ========================================================

    archive = truncate_archive(

        archive,

        ARCHIVE_SIZE
    )


    archive = sorted(

        archive,

        key=lambda x: x["cost"]
    )


    print("\n" + "=" * 65)

    print(
        "FINAL MOQPSO PARETO FRONT"
    )

    print("=" * 65)


    for i, entry in enumerate(

        archive,

        start=1
    ):

        print(

            f"Plan {i:02d} | "

            f"Cost: "
            f"${entry['cost']:.2f} | "

            f"WTW GHG: "
            f"{entry['ghg']:.3f} "
            f"tCO2e"
        )


    # ========================================================
    # SAVE
    # ========================================================

    output_file = (
        save_pareto_front(
            archive
        )
    )


    print(
        "\nSaved Pareto front:"
    )

    print(
        output_file
    )


    # ========================================================
    # REPRESENTATIVE PLAN
    # ========================================================

    if archive:

        representative = archive[
            len(archive) // 2
        ]


        print(
            "\nRepresentative "
            "Pareto Plan"
        )

        print("-" * 65)


        print(

            f"Total Cost : "
            f"${representative['cost']:.2f}"
        )


        print(

            f"Total WTW GHG : "
            f"{representative['ghg']:.3f} "
            f"tonnes CO2e"
        )


        print(
            "\nVoyage-wise plan:"
        )


        for result in (
            representative["results"]
        ):

            print(

                f"{result['voyage_id']} | "

                f"{result['vessel_id']} | "

                f"{result['fuel_type']} | "

                f"{result['speed']:.2f} kn | "

                f"Fuel: "
                f"{result['fuel_tonnes']:.3f} t | "

                f"Cost: "
                f"${result['fuel_cost']:.2f} | "

                f"GHG: "
                f"{result['wtw_ghg']:.3f}"
            )


    return archive


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    run_moqpso()