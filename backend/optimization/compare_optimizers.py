import pandas as pd
import os
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLASSICAL_FILE = os.path.join(
    BASE_DIR, "classical_baseline", "classical_results.csv"
)

QPSO_FILE = os.path.join(
    BASE_DIR, "quantum_inspired", "qpso_result.csv"
)

MOQPSO_FILE = os.path.join(
    BASE_DIR, "quantum_inspired", "moqpso_pareto_front.csv"
)


# ============================================================
# LOAD RESULTS
# ============================================================

classical = pd.read_csv(CLASSICAL_FILE)
qpso = pd.read_csv(QPSO_FILE)
moqpso = pd.read_csv(MOQPSO_FILE)


# ============================================================
# CLASSICAL
# ============================================================

classical_cost = classical["fuel_cost_usd"].sum()
classical_ghg = classical["wtw_ghg_tonnes"].sum()


# ============================================================
# QPSO
# ============================================================

qpso_cost = qpso["fuel_cost"].sum()
qpso_ghg = qpso["wtw_ghg"].sum()


# ============================================================
# MOQPSO
# ============================================================

# Support common column names
if "total_cost" in moqpso.columns:
    cost_col = "total_cost"
elif "fuel_cost_usd" in moqpso.columns:
    cost_col = "fuel_cost_usd"
else:
    raise ValueError("Cannot find MOQPSO cost column")

if "total_wtw_ghg" in moqpso.columns:
    ghg_col = "total_wtw_ghg"
elif "wtw_ghg_tonnes" in moqpso.columns:
    ghg_col = "wtw_ghg_tonnes"
else:
    raise ValueError("Cannot find MOQPSO GHG column")


moqpso_best_cost = moqpso[cost_col].min()
moqpso_best_ghg = moqpso[ghg_col].min()


# ============================================================
# COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame({
    "Optimizer": [
        "Classical",
        "QPSO",
        "MOQPSO - Best Cost",
        "MOQPSO - Best GHG"
    ],
    "Total Cost (USD)": [
        classical_cost,
        qpso_cost,
        moqpso_best_cost,
        moqpso.loc[moqpso[ghg_col].idxmin(), cost_col]
    ],
    "WTW GHG (tonnes CO2e)": [
        classical_ghg,
        qpso_ghg,
        moqpso.loc[moqpso[cost_col].idxmin(), ghg_col],
        moqpso_best_ghg
    ]
})


print("\n" + "=" * 70)
print("              OPTIMIZER COMPARISON")
print("=" * 70)

print(comparison.to_string(index=False))

# ============================================================
# IMPROVEMENTS
# ============================================================

qpso_cost_change = (
    (classical_cost - qpso_cost) / classical_cost * 100
)

qpso_ghg_change = (
    (classical_ghg - qpso_ghg) / classical_ghg * 100
)

print("\n" + "-" * 70)
print("QPSO vs CLASSICAL")
print("-" * 70)

print(f"Cost change : {qpso_cost_change:+.2f}%")
print(f"GHG change  : {qpso_ghg_change:+.2f}%")


# ============================================================
# SAVE TABLE
# ============================================================

output_file = os.path.join(BASE_DIR, "optimizer_comparison.csv")

comparison.to_csv(output_file, index=False)

print(f"\nComparison saved to:")
print(output_file)


# ============================================================
# COST GRAPH
# ============================================================

plt.figure(figsize=(9, 5))

plt.bar(
    comparison["Optimizer"],
    comparison["Total Cost (USD)"]
)

plt.ylabel("Total Cost (USD)")
plt.title("Optimizer Cost Comparison")
plt.xticks(rotation=20)
plt.tight_layout()

cost_plot = os.path.join(BASE_DIR, "optimizer_cost_comparison.png")
plt.savefig(cost_plot, dpi=200)
plt.show()


# ============================================================
# GHG GRAPH
# ============================================================

plt.figure(figsize=(9, 5))

plt.bar(
    comparison["Optimizer"],
    comparison["WTW GHG (tonnes CO2e)"]
)

plt.ylabel("WTW GHG (tonnes CO2e)")
plt.title("Optimizer WTW GHG Comparison")
plt.xticks(rotation=20)
plt.tight_layout()

ghg_plot = os.path.join(BASE_DIR, "optimizer_ghg_comparison.png")
plt.savefig(ghg_plot, dpi=200)
plt.show()


# ============================================================
# MOQPSO PARETO FRONT
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    moqpso[cost_col],
    moqpso[ghg_col],
    s=70
)

plt.xlabel("Total Cost (USD)")
plt.ylabel("WTW GHG (tonnes CO2e)")
plt.title("MOQPSO Pareto Front")
plt.grid(True)

pareto_plot = os.path.join(BASE_DIR, "moqpso_pareto_front.png")
plt.savefig(pareto_plot, dpi=200)
plt.show()

print("\nAll comparison plots generated successfully.")