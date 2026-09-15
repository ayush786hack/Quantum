import os
import pandas as pd
from datasets import load_dataset


print("[FuelCast] Loading dataset configs...")

configs = [
    "cps_poseidon",
    "cps_triton",
    "oss_ceto"
]

dataframes = []

for config in configs:

    print(f"\n[FuelCast] Loading: {config}")

    dataset = load_dataset(
        "krohnedigital/FuelCast",
        config
    )

    print(dataset)

    for split_name in dataset.keys():

        split_df = dataset[split_name].to_pandas()

        print(
            f"[FuelCast] {config} / {split_name}: "
            f"{split_df.shape}"
        )

        # Add vessel identifier
        split_df["vessel_name"] = config

        dataframes.append(split_df)


# ---------------------------------------
# Combine all configs
# ---------------------------------------

df = pd.concat(
    dataframes,
    ignore_index=True
)

print("\n[FuelCast] Combined dataset shape:")
print(df.shape)


# ---------------------------------------
# Save
# ---------------------------------------

base_dir = os.path.dirname(
    os.path.abspath(__file__)
)

output_dir = os.path.join(
    base_dir,
    "data",
    "processed"
)

os.makedirs(
    output_dir,
    exist_ok=True
)

output_path = os.path.join(
    output_dir,
    "fuelcast_combined.csv"
)

df.to_csv(
    output_path,
    index=False
)

print(
    f"\n[FuelCast] Successfully saved to:\n"
    f"{output_path}"
)