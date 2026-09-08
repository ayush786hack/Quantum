import json
import os
import sys
from urllib.parse import urlencode
from urllib.request import urlopen

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from prediction.predictor import predict_fuel_consumption, calculate_voyage_cost, calculate_wtw_emissions
from optimization.quantum_inspired.qaoa_demo import run_fuel_qaoa_demo

router = APIRouter(prefix="/api", tags=["Scenario Intelligence"])
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PORT_COORDINATES = {"R01": (31.2, 121.5), "R02": (1.3, 103.8), "R03": (53.5, 9.9), "R04": (25.2, 55.3), "R05": (35.7, 139.7), "R06": (29.9, 122.0), "R07": (-23.9, -46.3), "R08": (35.1, 129.0)}


def _load_routes():
    with open(os.path.join(RAW_DIR, "routes.json")) as handle:
        return json.load(handle)


class WeatherUpdate(BaseModel):
    route_id: str = Field(default="R03")
    vessel_type: str = Field(default="Container Ship")
    capacity: float = Field(default=10000, gt=0)
    engine_power_kw: float = Field(default=35000, gt=0)
    speed_knots: float = Field(default=18, gt=0)
    fuel_type: str = Field(default="LNG")
    new_sea_state: float = Field(default=5.0, ge=1, le=8)
    use_live_weather: bool = Field(default=True)


def _fetch_open_meteo_weather(route_id: str) -> dict:
    latitude, longitude = PORT_COORDINATES.get(route_id, (35.0, 10.0))
    query = urlencode({"latitude": latitude, "longitude": longitude, "current": "wave_height,wave_direction,wave_period"})
    try:
        with urlopen(f"https://marine-api.open-meteo.com/v1/marine?{query}", timeout=4) as response:
            current = json.loads(response.read().decode("utf-8")).get("current", {})
        wave_height = float(current.get("wave_height", 1.5))
        return {"source": "Open-Meteo Marine API", "wave_height_m": wave_height, "wave_direction_deg": current.get("wave_direction"), "wave_period_s": current.get("wave_period"), "sea_state": round(max(1.0, min(8.0, 1.0 + wave_height * 1.7)), 1)}
    except Exception as error:
        return {"source": "synthetic live-weather fallback", "wave_height_m": None, "sea_state": 5.0, "error": str(error)}


@router.post("/weather-reroute")
def weather_reroute(req: WeatherUpdate):
    routes = _load_routes()
    route = next((item for item in routes if item["id"] == req.route_id), None)
    if route is None:
        raise HTTPException(status_code=404, detail="Unknown route")
    live_weather = _fetch_open_meteo_weather(req.route_id) if req.use_live_weather else {"source": "manual scenario", "sea_state": req.new_sea_state}
    observed_sea_state = live_weather.get("sea_state", req.new_sea_state)
    safer = dict(route)
    safer["id"] = f"{route['id']}-WEATHER"
    safer["distance_nmi"] = round(route["distance_nmi"] * 1.06, 1)
    safer["avg_sea_state"] = max(1.0, observed_sea_state - 1.2)
    old_fc = predict_fuel_consumption(req.vessel_type, req.capacity, req.engine_power_kw, route["distance_nmi"], req.speed_knots, route["avg_sea_state"], .85, req.fuel_type)
    safer_speed = max(9.0, req.speed_knots - 1.0)
    new_fc = predict_fuel_consumption(req.vessel_type, req.capacity, req.engine_power_kw, safer["distance_nmi"], safer_speed, safer["avg_sea_state"], .85, req.fuel_type)
    old_cost = calculate_voyage_cost(old_fc, req.fuel_type, route["distance_nmi"], req.speed_knots, False)
    new_cost = calculate_voyage_cost(new_fc, req.fuel_type, safer["distance_nmi"], safer_speed, False)
    return {"status": "success", "live_weather": live_weather, "original_route": route, "recommended_route": safer, "recommended_speed_knots": safer_speed, "fuel_cost_delta_usd": round(new_cost - old_cost, 2), "fuel_delta_tonnes": round(new_fc - old_fc, 2), "reason": "Live weather detected; safer corridor trades distance for lower weather resistance."}


@router.get("/quantum-fuel-demo")
def quantum_fuel_demo():
    return run_fuel_qaoa_demo()


@router.get("/route-options")
def route_options(
    route_id: str = "R03",
    vessel_type: str = "Container Ship",
    capacity: float = 10000,
    engine_power_kw: float = 35000,
    speed_knots: float = 18,
    sea_state: float = 4.0,
    fuel_type: str = "LNG",
):
    """Compare route corridors using the same prediction engine as QIGA."""
    routes = _load_routes()
    route = next((item for item in routes if item["id"] == route_id), None)
    if route is None:
        raise HTTPException(status_code=404, detail="Unknown route")

    candidates = [
        {"id": f"{route_id}-DIRECT", "label": "Direct corridor", "distance_multiplier": 1.0, "sea_state_delta": 0.0, "waypoints": [route["origin"], route["destination"]]},
        {"id": f"{route_id}-SHELTERED", "label": "Sheltered coastal route", "distance_multiplier": 1.12, "sea_state_delta": -0.9, "waypoints": [route["origin"], "Coastal shelter", route["destination"]]},
        {"id": f"{route_id}-GREEN", "label": "Green corridor", "distance_multiplier": 1.06, "sea_state_delta": -0.5, "waypoints": [route["origin"], "Green corridor", route["destination"]]},
    ]
    options = []
    for candidate in candidates:
        distance = round(route["distance_nmi"] * candidate["distance_multiplier"], 1)
        candidate_sea_state = max(1.0, sea_state + candidate["sea_state_delta"])
        fuel_tonnes = predict_fuel_consumption(vessel_type, capacity, engine_power_kw, distance, speed_knots, candidate_sea_state, .85, fuel_type)
        options.append({
            **candidate,
            "distance_nmi": distance,
            "sea_state": round(candidate_sea_state, 1),
            "fuel_consumption_tonnes": fuel_tonnes,
            "fuel_cost_usd": calculate_voyage_cost(fuel_tonnes, fuel_type, distance, speed_knots, False),
            "wtw_emissions_tco2e": calculate_wtw_emissions(fuel_tonnes, fuel_type),
        })
    recommended = min(options, key=lambda item: item["fuel_consumption_tonnes"])
    return {"route_id": route_id, "origin": route["origin"], "destination": route["destination"], "fuel_type": fuel_type, "recommended_route_id": recommended["id"], "recommendation_reason": "Lowest predicted fuel consumption for the selected fuel and weather.", "options": options}


@router.get("/voyage-track")
def voyage_track(
    route_id: str = "R03",
    progress_pct: float = 0.0,
    vessel_type: str = "Container Ship",
    capacity: float = 10000,
    engine_power_kw: float = 35000,
    speed_knots: float = 18,
    fuel_type: str = "LNG",
):
    """Return a synthetic AIS-like position and live fuel ledger for demo tracking."""
    routes = _load_routes()
    route = next((item for item in routes if item["id"] == route_id), None)
    if route is None:
        raise HTTPException(status_code=404, detail="Unknown route")
    coordinates = {
        "R01": ((31.2, 121.5), (51.9, 4.5)), "R02": ((1.3, 103.8), (18.9, 72.8)),
        "R03": ((53.5, 9.9), (40.7, -74.0)), "R04": ((25.2, 55.3), (6.9, 79.9)),
        "R05": ((35.7, 139.7), (33.7, -118.2)), "R06": ((29.9, 122.0), (1.3, 103.8)),
        "R07": ((-23.9, -46.3), (51.2, 4.4)), "R08": ((35.1, 129.0), (49.3, -123.1)),
    }
    start, end = coordinates.get(route_id, ((0.0, 0.0), (10.0, 10.0)))
    progress = max(0.0, min(100.0, float(progress_pct))) / 100.0
    total_fuel = predict_fuel_consumption(vessel_type, capacity, engine_power_kw, route["distance_nmi"], speed_knots, route.get("avg_sea_state", 3.0), .85, fuel_type)
    total_hours = route["distance_nmi"] / max(1.0, speed_knots)
    consumed = round(total_fuel * progress, 2)
    remaining = round(max(0.0, total_fuel - consumed), 2)
    latitude = round(start[0] + ((end[0] - start[0]) * progress), 4)
    longitude = round(start[1] + ((end[1] - start[1]) * progress), 4)
    return {
        "status": "tracking",
        "route_id": route_id,
        "position": {"latitude": latitude, "longitude": longitude, "label": f"{latitude:.2f}°, {longitude:.2f}°"},
        "progress_pct": round(progress * 100, 1),
        "distance_completed_nmi": round(route["distance_nmi"] * progress, 1),
        "distance_remaining_nmi": round(route["distance_nmi"] * (1 - progress), 1),
        "fuel_consumed_tonnes": consumed,
        "fuel_remaining_tonnes": remaining,
        "fuel_rate_tonnes_per_hour": round(total_fuel / total_hours, 3),
        "eta_hours": round(total_hours * (1 - progress), 1),
        "fuel_type": fuel_type,
        "source": "synthetic AIS + prediction engine",
    }


@router.get("/bunkering-recommendations")
def bunkering_recommendations(route_id: str = "R03", fuel_type: str = "LNG"):
    routes = _load_routes()
    route = next((item for item in routes if item["id"] == route_id), None)
    if route is None:
        raise HTTPException(status_code=404, detail="Unknown route")
    ports = [
        {"port": route["origin"], "availability": ["HFO", "LNG", "Methanol"], "price_usd_t": {"HFO": 610, "LNG": 810, "Methanol": 735}},
        {"port": "Colombo Green Bunker Hub", "availability": ["LNG", "Methanol", "Ammonia"], "price_usd_t": {"LNG": 760, "Methanol": 690, "Ammonia": 1020}},
        {"port": route["destination"], "availability": ["HFO", "LNG", "Hydrogen", "Ammonia"], "price_usd_t": {"HFO": 625, "LNG": 830, "Hydrogen": 2050, "Ammonia": 990}},
    ]
    candidates = [item for item in ports if fuel_type in item["availability"]]
    ranked = sorted(candidates, key=lambda item: item["price_usd_t"][fuel_type])
    return {"route_id": route_id, "fuel_type": fuel_type, "recommendation": ranked[0], "ranked_stops": ranked, "arbitrage_saving_usd_t": round(ranked[-1]["price_usd_t"][fuel_type] - ranked[0]["price_usd_t"][fuel_type], 2)}


@router.get("/retrofit-roi")
def retrofit_roi(vessel_type: str = "Container Ship", capacity: float = 10000):
    options = [("LNG", 0.28, 1200000), ("Methanol", 0.42, 1750000), ("Ammonia", 0.78, 2600000), ("Hydrogen", 0.86, 3400000)]
    ranked = [{"fuel_type": fuel, "emission_reduction_pct": reduction * 100, "conversion_cost_usd": cost, "reduction_per_million_usd": round(reduction * 100 / (cost / 1000000), 2)} for fuel, reduction, cost in options]
    return {"vessel_type": vessel_type, "capacity": capacity, "ranked_options": sorted(ranked, key=lambda item: item["reduction_per_million_usd"], reverse=True)}