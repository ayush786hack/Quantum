# Green Fleet Optimizer

Quantum-inspired fuel consumption prediction and fleet optimization framework.

## Run The Demo

The Python service is the ML and optimization microservice. It generates the
synthetic fleet and voyage data on first use, so no private operational data is
required.

```powershell
cd backend
python -m uvicorn api.app:app --reload --port 8000
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The dashboard loads the preloaded Green Corridor
case study immediately and replaces it with live QIGA output when the API is
available. The optional Express gateway in `backend/gateway` proxies `/api` to
the Python service on port 8000 for a Node-orchestrated deployment.

## Honest Algorithm Note

QIGA uses qubit-pair probability amplitudes and rotation-gate updates; QPSO
uses quantum potential-well movement. GA and PSO are classical continuous
baselines. They share the same prediction engine and constraint evaluator, so
the benchmark compares search behavior on the same problem rather than
claiming quantum hardware speedup.

## Structure
- `backend/` - prediction, optimization (quantum-inspired + classical baselines), benchmarking, API
- `frontend/` - dashboard, fleet plan view, Pareto chart, scenario controls
- `docs/` - architecture, math formulation, benchmarking report
- `notebooks/` - exploratory analysis

## SIH 2026 MVP At A Glance

This is a demo-first quantum-inspired maritime fleet planning platform. It predicts fuel consumption with an uncertainty band, jointly optimizes route, speed, fuel, and shore power, and returns a Pareto frontier for cost, well-to-wake emissions, and schedule delay.

The complete demo runs on deterministic synthetic vessels, routes, weather, demand, port availability, fuel prices, and voyage records. No private operational dataset is required.

### Architecture

```text
React/Vite dashboard :5173
	|
	v
Express orchestration gateway :5000
	|
	v
FastAPI ML/optimization service :8000
  |-- prediction + uncertainty
  |-- QIGA / QPSO
  |-- classical GA / PSO
  |-- objectives and maritime constraints
  |-- benchmarks and scenario intelligence
	|
	v
Synthetic CSV + JSON data in backend/data
```

### Demo Flow

1. Change fleet size, carbon tax, cargo demand, fuels, or shore power.
2. Click **Run Optimizer** and watch the QIGA activity stream.
3. Select a Pareto point to inspect vessel, route, speed, fuel, and berth power.
4. Read the route map to compare direct, sheltered, and green corridors.
5. Click **Start Live Tracking** to see synthetic AIS position, fuel consumed, burn rate, remaining fuel, and ETA update every two seconds.
6. Try weather rerouting, bunkering arbitrage, retrofit ROI, and the live benchmark.

### Key API Routes

| Route | Purpose |
| --- | --- |
| `POST /api/predict-fuel` | Fuel prediction, confidence band, emissions, cost, CII |
| `POST /api/optimize-fleet` | Joint route/speed/fuel/shore-power optimization |
| `GET /api/benchmark-results` | QIGA/QPSO versus GA/PSO comparison |
| `GET /api/route-options` | Fuel-aware route corridor comparison |
| `GET /api/voyage-track` | Synthetic AIS position and live fuel ledger |
| `POST /api/weather-reroute` | Weather-adaptive route and speed recommendation |
| `GET /api/bunkering-recommendations` | Port availability and fuel-price arbitrage |
| `GET /api/retrofit-roi` | Alternative-fuel conversion ranking |

### Five-Person Team Split

- Prediction/data: synthetic generator, feature engineering, uncertainty validation.
- Quantum optimization: QIGA/QPSO encoding and search updates.
- Baselines/benchmarking: GA, PSO, convergence, scalability, and metrics.
- Backend/integration: FastAPI, Express gateway, scenario APIs, reliability.
- Product/demo: React dashboard, route map, explainability, and SIH pitch.

### SIH Standout Roadmap

The current MVP already demonstrates the differentiators. The next strongest additions are real AIS/weather adapters with synthetic fallback, a true multi-leg route graph, streamed solver convergence, PostgreSQL/MongoDB persistence, calibrated uncertainty, and vessel-specific CII/EEXI rules.

See [docs/sih_implementation_map.md](docs/sih_implementation_map.md) for the exact file to edit for every feature.

### Honest Quantum-Inspired Positioning

QIGA and QPSO run on classical hardware. They simulate qubit probability amplitudes and quantum-inspired update rules; they do not claim physical quantum hardware speedup. Classical GA and PSO share the same evaluator, prediction engine, and constraints for a fair comparison.
