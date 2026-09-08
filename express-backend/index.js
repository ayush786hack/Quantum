const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 5000;
const PYTHON_MICROSERVICE_URL = process.env.PYTHON_MICROSERVICE_URL || 'http://127.0.0.1:8000';

app.use(cors());
app.use(express.json());

// In-Memory Scenario Cache
let activeScenario = {
  id: "scenario-default",
  name: "Baseline Fleet Operation 2026",
  fuelPrices: { HFO: 600, LNG: 850, Methanol: 750, Hydrogen: 2200, Ammonia: 1100 },
  carbonTaxUsd: 0,
  demandMultiplier: 1.0,
  shorePowerEnabled: true,
  selectedAlgorithm: "QIGA"
};

// Health Check
app.get('/api/health', (req, res) => {
  res.json({
    status: "healthy",
    gateway: "Express Node.js Backend",
    python_microservice_url: PYTHON_MICROSERVICE_URL,
    timestamp: new Date().toISOString()
  });
});

// Scenario Endpoints
app.get('/api/scenario', (req, res) => {
  res.json(activeScenario);
});

app.post('/api/scenario', (req, res) => {
  activeScenario = { ...activeScenario, ...req.body };
  res.json({ status: "success", scenario: activeScenario });
});

// Proxy Fuel Prediction
app.post('/api/predict', async (req, res) => {
  try {
    const response = await axios.post(`${PYTHON_MICROSERVICE_URL}/api/predict-fuel`, req.body);
    res.json(response.data);
  } catch (error) {
    console.error('[Express Proxy Error - Predict]:', error.message);
    res.status(500).json({
      status: 'error',
      message: 'Python ML service unavailable. Ensure Python microservice is running on port 8000.',
      error: error.message
    });
  }
});

// Proxy Fleet Optimization
app.post('/api/optimize', async (req, res) => {
  try {
    const payload = {
      algorithm: req.body.algorithm || activeScenario.selectedAlgorithm,
      fleet_size: req.body.fleet_size || 10,
      generations: req.body.generations || 25,
      pop_size: req.body.pop_size || 35,
      fuel_prices: req.body.fuel_prices || activeScenario.fuelPrices,
      carbon_tax: req.body.carbon_tax !== undefined ? req.body.carbon_tax : activeScenario.carbonTaxUsd,
      demand_mult: req.body.demand_mult !== undefined ? req.body.demand_mult : activeScenario.demandMultiplier
    };
    
    const response = await axios.post(`${PYTHON_MICROSERVICE_URL}/api/optimize-fleet`, payload);
    res.json(response.data);
  } catch (error) {
    console.error('[Express Proxy Error - Optimize]:', error.message);
    res.status(500).json({
      status: 'error',
      message: 'Optimization microservice error.',
      error: error.message
    });
  }
});

// Proxy Benchmarks
app.get('/api/benchmarks', async (req, res) => {
  try {
    const response = await axios.get(`${PYTHON_MICROSERVICE_URL}/api/benchmark-results`);
    res.json(response.data);
  } catch (error) {
    console.error('[Express Proxy Error - Benchmarks]:', error.message);
    res.status(500).json({
      status: 'error',
      message: 'Benchmarking microservice error.',
      error: error.message
    });
  }
});

// Pre-loaded Case Study Scenario Endpoint
app.get('/api/case-study', async (req, res) => {
  try {
    const response = await axios.get(`${PYTHON_MICROSERVICE_URL}/api/case-study`);
    res.json(response.data);
  } catch (error) {
    // Return pre-loaded JSON if microservice is offline
    res.json({
      scenario_name: "SIH 2026 Green Corridor 2030 Demonstration",
      description: "Pre-computed high-performance scenario illustrating full fleet decarbonization across 12 vessels operating on trans-oceanic trade routes under a $100/t Carbon Tax and Shore Power at Berth.",
      fleet_size: 12,
      carbon_tax_usd_tco2e: 100.0,
      summary: {
        hfo_baseline_cost_usd: 4850000.0,
        hfo_baseline_emissions_tco2e: 38500.0,
        optimized_cost_usd: 3920000.0,
        optimized_emissions_tco2e: 14200.0,
        cost_reduction_pct: 19.2,
        emission_reduction_pct: 63.1,
        cii_grade_dist: { "A": 7, "B": 4, "C": 1, "D": 0, "E": 0 }
      }
    });
  }
});

app.listen(PORT, () => {
  console.log(`[Express Gateway] Server running on http://localhost:${PORT}`);
  console.log(`[Express Gateway] Proxying Python microservice at ${PYTHON_MICROSERVICE_URL}`);
});
