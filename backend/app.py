import os
import sys
import pandas as pd
import streamlit as st

# ============================================================
# PATH SETUP
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# app.py is inside backend/
BACKEND_DIR = CURRENT_DIR

if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

# ============================================================
# IMPORT PREDICTOR
# ============================================================

from prediction.predictor import (
    predict_fuel_rate,
    calculate_voyage_time,
    calculate_voyage_fuel,
)

# ============================================================
# FILE PATHS
# ============================================================

DATA_DIR = os.path.join(BACKEND_DIR, "data", "processed")
OPT_DIR = os.path.join(BACKEND_DIR, "optimization")

CLASSICAL_FILE = os.path.join(
    OPT_DIR, "classical_baseline", "classical_results.csv"
)

QPSO_FILE = os.path.join(
    OPT_DIR, "quantum_inspired", "qpso_result.csv"
)

MOQPSO_FILE = os.path.join(
    OPT_DIR, "quantum_inspired", "moqpso_pareto_front.csv"
)

VESSELS_FILE = os.path.join(DATA_DIR, "vessels.csv")
ROUTES_FILE = os.path.join(DATA_DIR, "routes.csv")
VOYAGES_FILE = os.path.join(DATA_DIR, "voyages.csv")
FUELS_FILE = os.path.join(DATA_DIR, "fuels.csv")

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Green Fleet Optimizer",
    page_icon="🚢",
    layout="wide",
)

# ============================================================
# HELPERS
# ============================================================

@st.cache_data
def load_data():
    vessels = pd.read_csv(VESSELS_FILE)
    routes = pd.read_csv(ROUTES_FILE)
    voyages = pd.read_csv(VOYAGES_FILE)
    fuels = pd.read_csv(FUELS_FILE)
    return vessels, routes, voyages, fuels


@st.cache_data
def load_result(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def money(value):
    return f"${value:,.2f}"


# ============================================================
# LOAD DATA
# ============================================================

try:
    vessels_df, routes_df, voyages_df, fuels_df = load_data()
except Exception as e:
    st.error(f"Could not load project data: {e}")
    st.stop()

# ============================================================
# HEADER
# ============================================================

st.title("🚢 Green Fleet Optimizer")
st.caption(
    "Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization"
)

st.markdown("---")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select module",
    [
        "Dashboard",
        "Fuel Prediction",
        "Voyage Analysis",
        "Optimizer Comparison",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Prototype dashboard for SIH 2026. "
    "Prediction uses the trained FuelCast XGBoost model."
)

# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.header("📊 Fleet Optimization Dashboard")

    classical = load_result(CLASSICAL_FILE)
    qpso = load_result(QPSO_FILE)
    moqpso = load_result(MOQPSO_FILE)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Voyages", len(voyages_df))
    col2.metric("Fleet Vessels", len(vessels_df))
    col3.metric("Fuel Options", len(fuels_df))
    col4.metric("Routes", len(routes_df))

    st.markdown("### Current Optimizer Results")

    rows = []

    if classical is not None:
        rows.append({
            "Optimizer": "Classical",
            "Cost (USD)": classical["fuel_cost_usd"].sum()
            if "fuel_cost_usd" in classical.columns
            else classical["fuel_cost"].sum(),
            "WTW GHG (tCO2e)": classical["wtw_ghg_tonnes"].sum()
            if "wtw_ghg_tonnes" in classical.columns
            else classical["wtw_ghg"].sum(),
        })

    if qpso is not None:
        rows.append({
            "Optimizer": "QPSO",
            "Cost (USD)": qpso["fuel_cost"].sum(),
            "WTW GHG (tCO2e)": qpso["wtw_ghg"].sum(),
        })

    if moqpso is not None:
        cost_col = (
            "total_cost"
            if "total_cost" in moqpso.columns
            else "fuel_cost_usd"
            if "fuel_cost_usd" in moqpso.columns
            else None
        )

        ghg_col = (
            "total_wtw_ghg"
            if "total_wtw_ghg" in moqpso.columns
            else "wtw_ghg_tonnes"
            if "wtw_ghg_tonnes" in moqpso.columns
            else None
        )

        if cost_col and ghg_col:
            rows.append({
                "Optimizer": "MOQPSO - Best Cost",
                "Cost (USD)": moqpso[cost_col].min(),
                "WTW GHG (tCO2e)": moqpso.loc[
                    moqpso[cost_col].idxmin(), ghg_col
                ],
            })

            rows.append({
                "Optimizer": "MOQPSO - Best GHG",
                "Cost (USD)": moqpso.loc[
                    moqpso[ghg_col].idxmin(), cost_col
                ],
                "WTW GHG (tCO2e)": moqpso[ghg_col].min(),
            })

    if rows:
        comparison_df = pd.DataFrame(rows)

        st.dataframe(
            comparison_df.style.format({
                "Cost (USD)": "${:,.2f}",
                "WTW GHG (tCO2e)": "{:,.3f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Cost Comparison")
        st.bar_chart(
            comparison_df.set_index("Optimizer")["Cost (USD)"]
        )

        st.markdown("### WTW GHG Comparison")
        st.bar_chart(
            comparison_df.set_index("Optimizer")["WTW GHG (tCO2e)"]
        )

    else:
        st.warning("Optimizer result files are not available yet.")

# ============================================================
# FUEL PREDICTION
# ============================================================

elif page == "Fuel Prediction":

    st.header("⛽ Fuel Consumption Prediction")

    st.write(
        "Select a voyage and operating conditions. "
        "The trained XGBoost model predicts fuel consumption "
        "for the actual voyage distance."
    )

    # ------------------------------------------------------------
    # ROUTE / ORIGIN / DESTINATION SELECTION
    # ------------------------------------------------------------

    st.subheader("🗺️ Route Selection")

    route_col1, route_col2 = st.columns(2)

    origins = sorted(routes_df["origin"].dropna().unique().tolist())

    with route_col1:
        selected_origin = st.selectbox(
            "Origin Port",
            origins,
        )

    destination_options = sorted(
        routes_df[
            routes_df["origin"] == selected_origin
        ]["destination"].dropna().unique().tolist()
    )

    with route_col2:
        selected_destination = st.selectbox(
            "Destination Port",
            destination_options,
        )

    matching_routes = routes_df[
        (routes_df["origin"] == selected_origin)
        & (routes_df["destination"] == selected_destination)
    ]

    if matching_routes.empty:
        st.warning("No route is available for this origin and destination.")
        st.stop()

    selected_route = matching_routes.iloc[0]

    matching_voyages = voyages_df[
        voyages_df["route_id"] == selected_route["route_id"]
    ]

    if matching_voyages.empty:
        st.warning("No voyage is configured for this route.")
        st.stop()

    voyage_options = matching_voyages["voyage_id"].tolist()

    selected_voyage_id = st.selectbox(
        "Select Voyage",
        voyage_options,
    )

    selected_voyage = matching_voyages[
        matching_voyages["voyage_id"] == selected_voyage_id
    ].iloc[0]

    distance_nmi = float(selected_route["distance_nmi"])

    st.info(
        f'🚢 **{selected_origin} → {selected_destination}** | '
        f'Route: **{selected_route["route_id"]}** | '
        f'Distance: **{distance_nmi:.0f} nautical miles**'
    )

    # ------------------------------------------------------------
    # VESSEL + FUEL
    # ------------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        vessel_id = st.selectbox(
            "Select Vessel",
            vessels_df["vessel_id"].tolist(),
        )

        vessel = vessels_df[
            vessels_df["vessel_id"] == vessel_id
        ].iloc[0]

    with col2:
        compatible_fuels = [
            x.strip()
            for x in str(vessel["fuel_compatible"]).split(",")
        ]

        fuel = st.selectbox(
            "Select Fuel",
            compatible_fuels,
        )

    # ------------------------------------------------------------
    # OPERATING + ENVIRONMENTAL INPUTS
    # ------------------------------------------------------------

    st.subheader("Operating & Environmental Conditions")

    col1, col2, col3 = st.columns(3)

    with col1:
        speed = st.number_input(
            "Ship speed (knots)",
            min_value=0.1,
            max_value=30.0,
            value=float(
                min(
                    vessel["max_speed_knots"],
                    selected_route["speed_limit_knots"],
                )
            ),
            step=0.1,
        )

        draft_aft = st.number_input(
            "Draft aft (m)",
            min_value=0.0,
            max_value=20.0,
            value=8.0,
            step=0.1,
        )

        draft_fore = st.number_input(
            "Draft fore (m)",
            min_value=0.0,
            max_value=20.0,
            value=8.0,
            step=0.1,
        )

        wind_speed = st.number_input(
            "Wind speed (m/s)",
            min_value=0.0,
            max_value=50.0,
            value=8.0,
            step=0.1,
        )

    with col2:
        wind_gusts = st.number_input(
            "Wind gusts (m/s)",
            min_value=0.0,
            max_value=60.0,
            value=12.0,
            step=0.1,
        )

        wave_height = st.number_input(
            "Wave height (m)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
        )

        wave_period = st.number_input(
            "Wave period (s)",
            min_value=0.0,
            max_value=30.0,
            value=6.0,
            step=0.1,
        )

        swell_height = st.number_input(
            "Swell wave height (m)",
            min_value=0.0,
            max_value=10.0,
            value=0.5,
            step=0.1,
        )

    with col3:
        swell_period = st.number_input(
            "Swell wave period (s)",
            min_value=0.0,
            max_value=30.0,
            value=8.0,
            step=0.1,
        )

        current_velocity = st.number_input(
            "Ocean current velocity",
            min_value=-10.0,
            max_value=10.0,
            value=0.2,
            step=0.1,
        )

        temperature = st.number_input(
            "Temperature 2m (°C)",
            min_value=-50.0,
            max_value=60.0,
            value=25.0,
            step=0.5,
        )

    st.markdown("---")

    if st.button(
        "🔮 Predict Fuel for Actual Voyage",
        type="primary",
        use_container_width=True,
    ):

        try:
            # ----------------------------------------------------
            # ML FUEL PREDICTION
            # ----------------------------------------------------

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
                temperature=temperature,
            )

            # ----------------------------------------------------
            # ACTUAL VOYAGE CALCULATION
            # ----------------------------------------------------

            voyage_time = calculate_voyage_time(
                distance_nmi,
                speed,
            )

            fuel_tonnes = calculate_voyage_fuel(
                fuel_rate,
                distance_nmi,
                speed,
            )

            fuel_row = fuels_df[
                fuels_df["fuel_type"] == fuel
            ].iloc[0]

            fuel_price = float(
                fuel_row["price_per_tonne"]
            )

            wtw_factor = float(
                fuel_row["wtw_ghg_kg_per_kg_fuel"]
            )

            fuel_cost = (
                fuel_tonnes * fuel_price
            )

            wtw_ghg = (
                fuel_tonnes * wtw_factor
            )

            # ----------------------------------------------------
            # RESULTS
            # ----------------------------------------------------

            st.subheader("📊 Voyage Prediction Results")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Predicted Fuel Rate",
                f"{fuel_rate:.4f} kg/s",
            )

            col2.metric(
                "Fuel Required",
                f"{fuel_tonnes:.3f} tonnes",
            )

            col3.metric(
                "Voyage Time",
                f"{voyage_time:.2f} hours",
            )

            col4.metric(
                "Fuel Cost",
                money(fuel_cost),
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "WTW GHG",
                f"{wtw_ghg:.3f} tCO2e",
            )

            col2.metric(
                "Distance",
                f"{distance_nmi:.0f} nmi",
            )

            col3.metric(
                "Selected Fuel",
                fuel,
            )

            st.success(
                f'Prediction completed for {selected_voyage["voyage_id"]}. '
                f'Fuel is calculated for the actual {distance_nmi:.0f} nmi voyage.'
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ============================================================
# VOYAGE ANALYSIS
# ============================================================

elif page == "Voyage Analysis":

    st.header("🗺️ Voyage Analysis")

    voyage_id = st.selectbox(
        "Select voyage",
        voyages_df["voyage_id"].tolist(),
    )

    voyage = voyages_df[
        voyages_df["voyage_id"] == voyage_id
    ].iloc[0]

    route = routes_df[
        routes_df["route_id"] == voyage["route_id"]
    ].iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Route",
        voyage["route_id"],
    )

    col2.metric(
        "Distance",
        f'{route["distance_nmi"]:.0f} nmi',
    )

    col3.metric(
        "Cargo",
        f'{voyage["cargo_demand_tonnes"]:,.0f} t',
    )

    col4.metric(
        "Deadline",
        f'{voyage["deadline_hours"]:.0f} h',
    )

    st.markdown("### Voyage Configuration")

    col1, col2 = st.columns(2)

    with col1:
        vessel_id = st.selectbox(
            "Vessel",
            vessels_df["vessel_id"].tolist(),
        )

        vessel = vessels_df[
            vessels_df["vessel_id"] == vessel_id
        ].iloc[0]

        compatible = [
            x.strip()
            for x in str(vessel["fuel_compatible"]).split(",")
        ]

        fuel = st.selectbox(
            "Fuel",
            compatible,
        )

    with col2:
        speed = st.number_input(
            "Speed (knots)",
            min_value=1.0,
            max_value=30.0,
            value=float(
                min(
                    vessel["max_speed_knots"],
                    route["speed_limit_knots"],
                )
            ),
            step=0.1,
        )

    if st.button(
        "Calculate Voyage",
        type="primary",
        use_container_width=True,
    ):

        try:
            # Demo weather conditions consistent with optimizer.
            fuel_rate = predict_fuel_rate(
                speed_knots=speed,
                draft_aft=8.0,
                draft_fore=8.0,
                wind_speed=8.0,
                wind_gusts=12.0,
                wave_height=1.0,
                wave_period=6.0,
                swell_height=0.5,
                swell_period=8.0,
                current_velocity=0.2,
                temperature=25.0,
            )

            voyage_time = calculate_voyage_time(
                float(route["distance_nmi"]),
                speed,
            )

            fuel_tonnes = calculate_voyage_fuel(
                fuel_rate,
                float(route["distance_nmi"]),
                speed,
            )

            fuel_row = fuels_df[
                fuels_df["fuel_type"] == fuel
            ].iloc[0]

            fuel_cost = (
                fuel_tonnes
                * float(fuel_row["price_per_tonne"])
            )

            wtw_ghg = (
                fuel_tonnes
                * float(fuel_row["wtw_ghg_kg_per_kg_fuel"])
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Fuel",
                f"{fuel_tonnes:.3f} t",
            )

            col2.metric(
                "Voyage Time",
                f"{voyage_time:.2f} h",
            )

            col3.metric(
                "Fuel Cost",
                money(fuel_cost),
            )

            col4.metric(
                "WTW GHG",
                f"{wtw_ghg:.3f} tCO2e",
            )

        except Exception as e:
            st.error(f"Voyage calculation failed: {e}")

# ============================================================
# OPTIMIZER COMPARISON
# ============================================================

elif page == "Optimizer Comparison":

    st.header("⚛️ Classical vs Quantum-Inspired Optimization")

    classical = load_result(CLASSICAL_FILE)
    qpso = load_result(QPSO_FILE)
    moqpso = load_result(MOQPSO_FILE)

    if classical is None:
        st.warning("Classical result file not found.")

    if qpso is None:
        st.warning("QPSO result file not found.")

    if moqpso is None:
        st.warning("MOQPSO Pareto file not found.")

    rows = []

    if classical is not None:
        classical_cost_col = (
            "fuel_cost_usd"
            if "fuel_cost_usd" in classical.columns
            else "fuel_cost"
        )
        classical_ghg_col = (
            "wtw_ghg_tonnes"
            if "wtw_ghg_tonnes" in classical.columns
            else "wtw_ghg"
        )

        rows.append({
            "Optimizer": "Classical",
            "Cost": classical[classical_cost_col].sum(),
            "GHG": classical[classical_ghg_col].sum(),
        })

    if qpso is not None:
        rows.append({
            "Optimizer": "QPSO",
            "Cost": qpso["fuel_cost"].sum(),
            "GHG": qpso["wtw_ghg"].sum(),
        })

    if moqpso is not None:

        cost_col = (
            "total_cost"
            if "total_cost" in moqpso.columns
            else "fuel_cost_usd"
        )

        ghg_col = (
            "total_wtw_ghg"
            if "total_wtw_ghg" in moqpso.columns
            else "wtw_ghg_tonnes"
        )

        best_cost_idx = moqpso[cost_col].idxmin()
        best_ghg_idx = moqpso[ghg_col].idxmin()

        rows.append({
            "Optimizer": "MOQPSO - Best Cost",
            "Cost": moqpso.loc[best_cost_idx, cost_col],
            "GHG": moqpso.loc[best_cost_idx, ghg_col],
        })

        rows.append({
            "Optimizer": "MOQPSO - Best GHG",
            "Cost": moqpso.loc[best_ghg_idx, cost_col],
            "GHG": moqpso.loc[best_ghg_idx, ghg_col],
        })

    if rows:

        comparison = pd.DataFrame(rows)

        st.subheader("Performance Summary")

        st.dataframe(
            comparison.style.format({
                "Cost": "${:,.2f}",
                "GHG": "{:,.3f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("💰 Cost")
            st.bar_chart(
                comparison.set_index("Optimizer")["Cost"]
            )

        with col2:
            st.subheader("🌱 WTW GHG")
            st.bar_chart(
                comparison.set_index("Optimizer")["GHG"]
            )

        if moqpso is not None:

            st.subheader("MOQPSO Pareto Front")

            pareto_chart = moqpso[
                [cost_col, ghg_col]
            ].copy()

            pareto_chart.columns = [
                "Total Cost (USD)",
                "WTW GHG (tCO2e)",
            ]

            st.scatter_chart(
                pareto_chart,
                x="Total Cost (USD)",
                y="WTW GHG (tCO2e)",
            )

            st.caption(
                "Each point represents a non-dominated MOQPSO solution "
                "showing the cost–emission trade-off."
            )

        st.markdown("---")
        st.subheader("Recommended Interpretation")

        if classical is not None and qpso is not None:

            classical_cost = rows[0]["Cost"]
            classical_ghg = rows[0]["GHG"]

            qpso_row = next(
                r for r in rows
                if r["Optimizer"] == "QPSO"
            )

            qpso_cost = qpso_row["Cost"]
            qpso_ghg = qpso_row["GHG"]

            cost_change = (
                (qpso_cost - classical_cost)
                / classical_cost
                * 100
            )

            ghg_change = (
                (qpso_ghg - classical_ghg)
                / classical_ghg
                * 100
            )

            st.write(
                f"**QPSO vs Classical:** QPSO changes total cost by "
                f"**{cost_change:+.2f}%** and WTW GHG by "
                f"**{ghg_change:+.2f}%**."
            )

            st.info(
                "The optimization problem is multi-objective: "
                "lower cost and lower WTW GHG can conflict. "
                "MOQPSO exposes this trade-off through its Pareto front."
            )

st.markdown("---")
st.caption(
    "Green Fleet Optimizer • SIH 2026 Prototype"
)
