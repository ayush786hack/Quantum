import sys
import os

# Add parent directory to sys.path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prediction.predictor import predict_fuel_consumption, calculate_wtw_emissions, calculate_voyage_cost, calculate_cii_rating

def evaluate_deployment_plan(
    fleet_plan: list,
    fuel_prices: dict = None,
    carbon_tax_usd: float = 0.0,
    demand_multiplier: float = 1.0,
    cargo_demand: float = None
) -> dict:
    """
    Evaluates a candidate fleet deployment solution vector.
    
    fleet_plan: list of dicts, each dict representing an assignment:
      {
        'vessel': dict,
        'route': dict,
        'speed_knots': float,
        'fuel_type': str,
        'shore_power_used': bool
      }
      
    Returns multi-objective metrics:
      - total_cost_usd: float (minimize)
      - total_wtw_emissions_tco2e: float (minimize)
      - total_delay_hours: float (minimize)
      - cii_breakdown: list
      - constraint_violations: int
    """
    total_cost = 0.0
    total_emissions = 0.0
    total_delay = 0.0
    total_capacity_provided = 0.0
    total_demand_required = float(cargo_demand) if cargo_demand is not None else 0.0
    cii_list = []
    explainability = []
    violations = 0
    
    for item in fleet_plan:
        v = item['vessel']
        r = item['route']
        speed = max(9.0, min(24.0, float(item['speed_knots'])))
        fuel = item['fuel_type']
        sp_used = item['shore_power_used']
        
        distance = r['distance_nmi']
        capacity = v['capacity']
        
        # Predict fuel consumption
        fc = predict_fuel_consumption(
            vessel_type=v['vessel_type'],
            capacity=capacity,
            engine_power_kw=v.get('engine_power_kw', 25000),
            distance_nmi=distance,
            speed_knots=speed,
            sea_state=r.get('avg_sea_state', 3.0),
            payload_pct=0.85,
            fuel_type=fuel
        )
        
        # WtW Emissions
        emissions = calculate_wtw_emissions(fc, fuel)
        total_emissions += emissions
        
        # Voyage Cost
        cost = calculate_voyage_cost(
            fuel_consumption_tonnes=fc,
            fuel_type=fuel,
            distance_nmi=distance,
            speed_knots=speed,
            shore_power_used=sp_used,
            custom_fuel_prices=fuel_prices,
            carbon_tax_usd_tco2e=carbon_tax_usd
        )
        total_cost += cost
        
        # Schedule Delay Risk: baseline design speed vs chosen speed
        design_speed = v.get('base_speed_knots', 16.0)
        sailing_hours = distance / speed
        nominal_hours = distance / design_speed
        delay_hours = max(0.0, sailing_hours - nominal_hours)
        total_delay += delay_hours
        
        # CII rating
        cii_grade = calculate_cii_rating(emissions, capacity, distance)
        cii_list.append({"vessel_id": v['vessel_id'], "grade": cii_grade})
        item["predicted_fuel_tonnes"] = fc
        item["wtw_emissions_tco2e"] = emissions
        item["voyage_cost_usd"] = cost
        item["cii_rating"] = cii_grade
        item["compliance_status"] = "green" if cii_grade in ["A", "B"] else "amber" if cii_grade == "C" else "red"
        explainability.append({
            "vessel_id": v["vessel_id"],
            "cost_driver": f"{fuel} at {speed:.1f} kn",
            "emissions_driver": f"{fuel} WtW factor with {r.get('avg_sea_state', 3.0):.1f} sea state",
            "schedule_driver": f"{round(delay_hours, 1)} h speed trade-off",
            "shore_power_decision": "cold ironing" if sp_used else "auxiliary engine",
        })
        
        # Check constraints
        total_capacity_provided += capacity
        if cargo_demand is None:
            total_demand_required += capacity * demand_multiplier
        
        if cii_grade in ["D", "E"]:
            violations += 1
            
        if fuel not in v.get('supported_fuels', ['HFO']):
            violations += 2
            
    # Demand satisfaction constraint penalty
    if total_demand_required > 0 and total_capacity_provided < total_demand_required:
        deficit_pct = (total_demand_required - total_capacity_provided) / total_demand_required
        violations += int(deficit_pct * 10) + 1
        # Apply cost penalty for unserved cargo demand
        total_cost += deficit_pct * 500000.0
        
    return {
        "total_cost_usd": round(total_cost, 2),
        "total_wtw_emissions_tco2e": round(total_emissions, 2),
        "total_delay_hours": round(total_delay, 1),
        "cii_breakdown": cii_list,
        "constraint_violations": violations,
        "capacity_provided": round(total_capacity_provided, 1),
        "cargo_demand": round(total_demand_required, 1),
        "fitness_vector": [round(total_cost, 2), round(total_emissions, 2), round(total_delay, 1)]
        ,"compliance_status": "red" if violations else "green"
        ,"explainability": explainability
    }
