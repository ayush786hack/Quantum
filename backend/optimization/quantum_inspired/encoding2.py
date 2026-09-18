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

BACKEND_DIR = os.path.dirname(
    os.path.dirname(CURRENT_DIR)
)

sys.path.append(BACKEND_DIR)


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
    os.path.join(
        DATA_DIR,
        "vessels.csv"
    )
)

routes = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "routes.csv"
    )
)

voyages = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "voyages.csv"
    )
)

fuels = pd.read_csv(
    os.path.join(
        DATA_DIR,
        "fuels.csv"
    )
)


# ============================================================
# VESSEL ENCODING
# ============================================================

VESSEL_IDS = (
    vessels["vessel_id"]
    .astype(str)
    .tolist()
)

VESSEL_TO_INDEX = {
    vessel_id: index
    for index, vessel_id
    in enumerate(VESSEL_IDS)
}

INDEX_TO_VESSEL = {
    index: vessel_id
    for vessel_id, index
    in VESSEL_TO_INDEX.items()
}


# ============================================================
# FUEL ENCODING
# ============================================================

FUEL_TYPES = (
    fuels["fuel_type"]
    .astype(str)
    .tolist()
)

FUEL_TO_INDEX = {
    fuel_type: index
    for index, fuel_type
    in enumerate(FUEL_TYPES)
}

INDEX_TO_FUEL = {
    index: fuel_type
    for fuel_type, index
    in FUEL_TO_INDEX.items()
}


# ============================================================
# VOYAGE ORDER
# ============================================================

VOYAGE_IDS = (
    voyages["voyage_id"]
    .astype(str)
    .tolist()
)


# ============================================================
# GET VOYAGE
# ============================================================

def get_voyage(voyage_id):

    matches = voyages[
        voyages["voyage_id"]
        == voyage_id
    ]

    if matches.empty:
        raise ValueError(
            f"Unknown voyage_id: {voyage_id}"
        )

    return matches.iloc[0]


# ============================================================
# GET ROUTE FOR VOYAGE
# ============================================================

def get_route_for_voyage(voyage_id):

    voyage = get_voyage(
        voyage_id
    )

    route_id = voyage[
        "route_id"
    ]

    matches = routes[
        routes["route_id"]
        == route_id
    ]

    if matches.empty:
        raise ValueError(
            f"Unknown route_id: {route_id}"
        )

    return matches.iloc[0]


# ============================================================
# GET COMPATIBLE FUELS
# ============================================================

def get_compatible_fuels(
    vessel_id
):

    matches = vessels[
        vessels["vessel_id"]
        == vessel_id
    ]

    if matches.empty:
        raise ValueError(
            f"Unknown vessel_id: {vessel_id}"
        )

    vessel = matches.iloc[0]

    compatible = [
        fuel.strip()
        for fuel in str(
            vessel["fuel_compatible"]
        ).split(",")
    ]

    return compatible


# ============================================================
# SPEED BOUNDS
# ============================================================

def get_speed_bounds(
    vessel_id,
    voyage_id
):

    vessel_matches = vessels[
        vessels["vessel_id"]
        == vessel_id
    ]

    if vessel_matches.empty:
        raise ValueError(
            f"Unknown vessel_id: {vessel_id}"
        )

    vessel = vessel_matches.iloc[0]

    route = get_route_for_voyage(
        voyage_id
    )

    min_speed = float(
        vessel["min_speed_knots"]
    )

    max_speed = min(
        float(
            vessel["max_speed_knots"]
        ),
        float(
            route["speed_limit_knots"]
        )
    )

    return min_speed, max_speed


# ============================================================
# ENCODE ONE DECISION
# ============================================================

def encode_decision(
    vessel_id,
    fuel_type,
    speed_knots
):

    if vessel_id not in VESSEL_TO_INDEX:
        raise ValueError(
            f"Unknown vessel: {vessel_id}"
        )

    if fuel_type not in FUEL_TO_INDEX:
        raise ValueError(
            f"Unknown fuel: {fuel_type}"
        )

    vessel_index = (
        VESSEL_TO_INDEX[
            vessel_id
        ]
    )

    fuel_index = (
        FUEL_TO_INDEX[
            fuel_type
        ]
    )

    return np.array(
        [
            float(vessel_index),
            float(fuel_index),
            float(speed_knots)
        ]
    )


# ============================================================
# DECODE ONE DECISION
# ============================================================

def decode_decision(
    particle_position,
    voyage_id
):

    position = np.asarray(
        particle_position,
        dtype=float
    )

    if position.shape[0] != 3:
        raise ValueError(
            "A decision must contain "
            "[vessel, fuel, speed]"
        )


    # --------------------------------------------------------
    # Vessel
    # --------------------------------------------------------

    vessel_index = int(
        np.round(
            position[0]
        )
    )

    vessel_index = int(
        np.clip(
            vessel_index,
            0,
            len(VESSEL_IDS) - 1
        )
    )

    vessel_id = INDEX_TO_VESSEL[
        vessel_index
    ]


    # --------------------------------------------------------
    # Fuel
    # --------------------------------------------------------

    fuel_index = int(
        np.round(
            position[1]
        )
    )

    fuel_index = int(
        np.clip(
            fuel_index,
            0,
            len(FUEL_TYPES) - 1
        )
    )

    fuel_type = INDEX_TO_FUEL[
        fuel_index
    ]


    # --------------------------------------------------------
    # Speed
    # --------------------------------------------------------

    min_speed, max_speed = (
        get_speed_bounds(
            vessel_id,
            voyage_id
        )
    )

    speed_knots = float(
        np.clip(
            position[2],
            min_speed,
            max_speed
        )
    )


    return {
        "voyage_id": voyage_id,
        "vessel_id": vessel_id,
        "fuel_type": fuel_type,
        "speed_knots": speed_knots
    }


# ============================================================
# RANDOM VALID DECISION
# ============================================================

def random_decision(
    voyage_id,
    rng=None
):

    if rng is None:
        rng = np.random.default_rng()


    # --------------------------------------------------------
    # Random vessel
    # --------------------------------------------------------

    vessel_id = rng.choice(
        VESSEL_IDS
    )


    # --------------------------------------------------------
    # Random compatible fuel
    # --------------------------------------------------------

    compatible_fuels = (
        get_compatible_fuels(
            vessel_id
        )
    )

    fuel_type = rng.choice(
        compatible_fuels
    )


    # --------------------------------------------------------
    # Random speed
    # --------------------------------------------------------

    min_speed, max_speed = (
        get_speed_bounds(
            vessel_id,
            voyage_id
        )
    )

    speed_knots = rng.uniform(
        min_speed,
        max_speed
    )


    return encode_decision(
        vessel_id,
        fuel_type,
        speed_knots
    )


# ============================================================
# CREATE RANDOM PARTICLE
# ============================================================

def create_random_particle(
    rng=None
):

    if rng is None:
        rng = np.random.default_rng()

    particle = []


    for voyage_id in VOYAGE_IDS:

        decision = random_decision(
            voyage_id,
            rng
        )

        particle.append(
            decision
        )


    return np.array(
        particle,
        dtype=float
    )


# ============================================================
# DECODE COMPLETE PARTICLE
# ============================================================

def decode_particle(
    particle
):

    particle = np.asarray(
        particle,
        dtype=float
    )

    if particle.shape != (
        len(VOYAGE_IDS),
        3
    ):

        raise ValueError(
            "Particle shape must be "
            f"({len(VOYAGE_IDS)}, 3)"
        )


    fleet_plan = []


    for i, voyage_id in enumerate(
        VOYAGE_IDS
    ):

        decision = decode_decision(
            particle[i],
            voyage_id
        )

        fleet_plan.append(
            decision
        )


    return fleet_plan


# ============================================================
# CREATE PARTICLE BOUNDS
# ============================================================

def get_particle_bounds():

    lower_bounds = []
    upper_bounds = []


    for voyage_id in VOYAGE_IDS:

        # -----------------------------------------------
        # Vessel index
        # -----------------------------------------------

        lower_bounds.append(
            0.0
        )

        upper_bounds.append(
            float(
                len(VESSEL_IDS) - 1
            )
        )


        # -----------------------------------------------
        # Fuel index
        # -----------------------------------------------

        lower_bounds.append(
            0.0
        )

        upper_bounds.append(
            float(
                len(FUEL_TYPES) - 1
            )
        )


        # -----------------------------------------------
        # Speed
        # -----------------------------------------------

        # Global speed bounds are initially used by QPSO.
        # Exact vessel/route limits are enforced while
        # decoding the particle.

        lower_bounds.append(
            5.0
        )

        upper_bounds.append(
            22.0
        )


    return (
        np.array(
            lower_bounds,
            dtype=float
        ),
        np.array(
            upper_bounds,
            dtype=float
        )
    )


# ============================================================
# PARTICLE TO FLAT VECTOR
# ============================================================

def particle_to_vector(
    particle
):

    particle = np.asarray(
        particle,
        dtype=float
    )

    return particle.flatten()


# ============================================================
# FLAT VECTOR TO PARTICLE
# ============================================================

def vector_to_particle(
    vector
):

    vector = np.asarray(
        vector,
        dtype=float
    )

    expected_size = (
        len(VOYAGE_IDS) * 3
    )

    if vector.size != expected_size:

        raise ValueError(
            f"Expected vector of size "
            f"{expected_size}, "
            f"got {vector.size}"
        )

    return vector.reshape(
        len(VOYAGE_IDS),
        3
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("       QUANTUM-INSPIRED PARTICLE ENCODING")
    print("=" * 60)


    print("\nVessels:")
    print(VESSEL_IDS)


    print("\nFuels:")
    print(FUEL_TYPES)


    print("\nVoyages:")
    print(VOYAGE_IDS)


    # --------------------------------------------------------
    # Create random particle
    # --------------------------------------------------------

    rng = np.random.default_rng(
        42
    )

    particle = create_random_particle(
        rng
    )


    print("\nRandom particle:")
    print(particle)


    # --------------------------------------------------------
    # Decode particle
    # --------------------------------------------------------

    fleet_plan = decode_particle(
        particle
    )


    print("\nDecoded fleet plan:")

    for decision in fleet_plan:

        print(
            f"  {decision['voyage_id']} → "
            f"{decision['vessel_id']} + "
            f"{decision['fuel_type']} + "
            f"{decision['speed_knots']:.2f} kn"
        )


    # --------------------------------------------------------
    # Flatten / reconstruct test
    # --------------------------------------------------------

    vector = particle_to_vector(
        particle
    )

    reconstructed = vector_to_particle(
        vector
    )


    print(
        "\nFlatten/reconstruct test:",
        np.allclose(
            particle,
            reconstructed
        )
    )


    print("\nEncoding test completed.")