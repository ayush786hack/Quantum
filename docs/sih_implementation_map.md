# SIH 2026 Implementation Map

This is the practical map from a feature to the code a team member should open first.

## Current MVP Features

| Feature | Start here | What it owns |
| --- | --- | --- |
| Synthetic operational data | `backend/data/synthetic_generator.py` | Vessels, routes, fuels, weather, demand, prices, voyage CSV |
| Fuel prediction | `backend/prediction/predictor.py` | Point estimate, confidence band, WtW emissions, cost, CII |
| Feature engineering | `backend/prediction/data_preprocessing.py` | Regression input columns and fuel/vessel encoding |
| Quantum-inspired search | `backend/optimization/quantum_inspired/qiga.py`, `qpso.py` | Qubit-pair encoding and search updates |
| Classical comparison | `backend/optimization/classical_baseline/` | GA and PSO baselines |
| Shared objectives | `backend/optimization/objectives.py` | Cost, emissions, delay, demand, fuel, CII, shore power |
| Solver orchestration | `backend/optimization/optimizer.py` | QIGA/QPSO/GA/PSO selection and common scenario constraints |
| Live benchmark | `backend/benchmarking/run_benchmarks.py` | Shared-problem comparison and convergence/scalability metrics |
| Prediction API | `backend/api/routes/predict.py` | `/api/predict-fuel` |
| Optimization API | `backend/api/routes/optimize.py` | `/api/optimize-fleet` |
| Scenario intelligence | `backend/api/routes/scenario.py` | Weather, route options, tracking, bunkering, retrofit |
| Dashboard state | `frontend/src/App.jsx` | Case study, benchmark, and optimizer state |
| Operator controls | `frontend/src/components/ScenarioControls/ScenarioControls.jsx` | Fleet, tax, demand, fuels, shore power |
| Pareto decisions | `frontend/src/components/ParetoChart/ParetoChart.jsx` | Cost/emissions trade-off selection |
| Fleet plan | `frontend/src/components/FleetPlanView/FleetPlanView.jsx` | Vessel, route, speed, fuel, shore power |
| Real-world route UI | `frontend/src/components/RouteGraph/RouteGraph.jsx` | Corridors, vessel marker, live fuel ledger |
| Dashboard styling | `frontend/src/styles.css` | Operational visual language and responsive layout |

## Best Next Features For SIH Impact

### 1. Real AIS and weather feed

Implement the provider adapter under `backend/data/` and expose it through `backend/api/routes/scenario.py`. Keep the current synthetic response as a fallback so the demo never depends on network availability.

### 2. True multi-leg route graph

Replace the three demo corridors in `route_options()` with a graph model under `backend/data/raw/routes.json`. Use Dijkstra or A* for feasible paths, then send the selected edge list to `RouteGraph.jsx`.

### 3. Live solver convergence stream

Add WebSocket or Server-Sent Events under `backend/api/` and emit each generation from `backend/optimization/optimizer.py`. Render the stream beside `ParetoChart.jsx` so judges can see QIGA and GA evolve side by side.

### 4. Persistent scenario history

Use PostgreSQL for vessels, routes, demand, and ports; use MongoDB for flexible optimization and benchmark results. Add repository modules under `backend/` rather than putting database calls in route handlers.

### 5. Calibrated uncertainty

Replace the tree-spread approximation in `backend/prediction/predictor.py` with conformal prediction or quantile regression. Store calibration metrics beside the model artifact and show coverage in the benchmark view.

### 6. Stronger maritime compliance

Extend `calculate_cii_rating()` and `backend/optimization/objectives.py` with vessel-specific CII reference lines, EEXI technical limits, lifecycle fuel factors, and port emission zones. Keep every violation in the explanation payload.

### 7. Judge-friendly explainability

Add a waterfall or contribution chart using the existing `explainability` payload. Use operational language: speed increased fuel by X, LNG reduced WtW emissions by Y, shore power avoided auxiliary fuel by Z.

## Five-Minute Presentation Story

1. Show the same voyage under HFO and its fuel, emissions, and CII risk.
2. Change weather and click reroute; show the cost delta and safer corridor.
3. Run QIGA; show the Pareto frontier and route/fuel/shore-power plan.
4. Start live tracking; show position, fuel consumed, burn rate, and ETA changing.
5. Open benchmarking and explain that quantum-inspired means a classical simulation with transparent baselines.

This tells a complete decision loop instead of showing a static optimizer screenshot.
