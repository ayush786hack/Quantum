import os
import pandas as pd
import numpy as np


# Features used by the planning fuel prediction model
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

TARGET_COLUMN = "ME_Total_MomentaryFuel"


def load_and_preprocess_data(csv_path=None):

    # --------------------------------------------------
    # 1. Locate FuelCast dataset
    # --------------------------------------------------

    if csv_path is None:
        base_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        csv_path = os.path.join(
            base_dir,
            "data",
            "processed",
            "fuelcast_combined.csv"
        )

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"FuelCast dataset not found at: {csv_path}"
        )

    print(f"[Data] Loading dataset from: {csv_path}")

    df = pd.read_csv(csv_path)

    print(f"[Data] Raw shape: {df.shape}")


    # --------------------------------------------------
    # 2. Construct Main Engine fuel target
    # --------------------------------------------------

    df[TARGET_COLUMN] = (
        df["Consumer_MainEnginePort_MomentaryFuel"]
        + df["Consumer_MainEngineStarboard_MomentaryFuel"]
    )


    # --------------------------------------------------
    # 3. Keep only rows with valid target
    # --------------------------------------------------

    # --------------------------------------------------
# 3. Keep only CPS Triton rows with valid target
# --------------------------------------------------

    df = df[
        (df["vessel_name"] == "cps_triton") &
        (df[TARGET_COLUMN].notna())
    ].copy()
    print(
        f"[Data] Rows with valid target: {len(df)}"
    )


    # --------------------------------------------------
    # 4. Remove physically invalid speed rows
    # --------------------------------------------------

    df = df[
        df["Ship_SpeedOverGround"].notna()
    ].copy()

    df = df[
        df["Ship_SpeedOverGround"] >= 0
    ].copy()


    # --------------------------------------------------
    # 5. Select features
    # --------------------------------------------------

    X = df[PLANNING_FEATURES].copy()

    y = df[TARGET_COLUMN].copy()


    # --------------------------------------------------
    # 6. Basic numeric cleaning
    # --------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    y = y.replace(
        [np.inf, -np.inf],
        np.nan
    )

    valid_rows = y.notna()

    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].copy()


    # --------------------------------------------------
    # 7. Report missing values
    # --------------------------------------------------

    missing = X.isna().sum()

    print("\n[Data] Missing values:")
    print(
        missing[missing > 0]
    )


    # --------------------------------------------------
    # 8. Return raw X and y
    #    Imputation will happen inside the ML pipeline
    # --------------------------------------------------

    print("\n[Data] Final shapes:")
    print(f"X: {X.shape}")
    print(f"y: {y.shape}")

    print("\n[Data] Features:")
    print(list(X.columns))

    print("\n[Data] Target statistics:")
    print(y.describe())

    return X, y, PLANNING_FEATURES


if __name__ == "__main__":

    X, y, features = load_and_preprocess_data()

    print("\n[Data] Preprocessing successful!")