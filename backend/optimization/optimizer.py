import time
import os
import json
from .objectives import evaluate_deployment_plan
from .quantum_inspired.qiga import QIGASolver
from .quantum_inspired.qpso import QPSOSolver
from .classical_baseline.ga import ClassicalGASolver
from .classical_baseline.pso import ClassicalPSOSolver

def run_fleet_optimization(
    vessels: list,
    routes: list,
    algorithm: str = "QIGA",
    pop_size: int = 40,
    generations: int = 30,
    fuel_prices: dict = None,
    carbon_tax: float = 0.0,
    demand_mult: float = 1.0,
    cargo_demand: float = None,
    allowed_fuels: list = None,
    shore_power_enabled: bool = True
) -> dict:
    """
    Main entrypoint for fleet deployment optimization.
    Algorithms: 'QIGA', 'QPSO', 'GA', 'PSO', or 'ALL'
    """
    algo_upper = algorithm.upper()
    start_time = time.time()
    
    # The shared evaluator is wrapped so every solver receives identical scenario constraints.
    def scenario_evaluator(fleet_plan, fuel_prices_arg=None, carbon_tax_arg=0.0, demand_mult_arg=1.0):
        if allowed_fuels:
            for item in fleet_plan:
                if item["fuel_type"] not in allowed_fuels:
                    item["fuel_type"] = allowed_fuels[0]
        if not shore_power_enabled:
            for item in fleet_plan:
                item["shore_power_used"] = False
        return evaluate_deployment_plan(
            fleet_plan,
            fuel_prices_arg,
            carbon_tax_arg,
            demand_mult_arg,
            cargo_demand=cargo_demand
        )

    solvers = {}
    if algo_upper == "QIGA":
        solvers["QIGA"] = QIGASolver(vessels, routes, pop_size, generations, fuel_prices=fuel_prices, carbon_tax=carbon_tax, demand_mult=demand_mult)
    elif algo_upper == "QPSO":
        solvers["QPSO"] = QPSOSolver(vessels, routes, pop_size, generations, fuel_prices=fuel_prices, carbon_tax=carbon_tax, demand_mult=demand_mult)
    elif algo_upper == "GA":
        solvers["GA"] = ClassicalGASolver(vessels, routes, pop_size, generations, fuel_prices=fuel_prices, carbon_tax=carbon_tax, demand_mult=demand_mult)
    elif algo_upper == "PSO":
        solvers["PSO"] = ClassicalPSOSolver(vessels, routes, pop_size, generations, fuel_prices=fuel_prices, carbon_tax=carbon_tax, demand_mult=demand_mult)
    elif algo_upper == "ALL":
        solvers["QIGA"] = QIGASolver(vessels, routes, pop_size, generations, fuel_prices=fuel_prices, carbon_tax=carbon_tax, demand_mult=demand_mult)
        solvers["QPSO"] = QPSOSolver(vessels, routes, pop_size, generations, fuel_prices=fuel_prices, carbon_tax=carbon_tax, demand_mult=demand_mult)
        solvers["GA"] = ClassicalGASolver(vessels, routes, pop_size, generations, fuel_prices=fuel_prices, carbon_tax=carbon_tax, demand_mult=demand_mult)
        solvers["PSO"] = ClassicalPSOSolver(vessels, routes, pop_size, generations, fuel_prices=fuel_prices, carbon_tax=carbon_tax, demand_mult=demand_mult)
    else:
        # Default to QIGA
        solvers["QIGA"] = QIGASolver(vessels, routes, pop_size, generations, fuel_prices=fuel_prices, carbon_tax=carbon_tax, demand_mult=demand_mult)

    results = {}
    for name, solver in solvers.items():
        t0 = time.time()
        res = solver.solve(scenario_evaluator)
        for pareto_item in res.get("pareto_front", []):
            evaluated = scenario_evaluator(pareto_item["fleet_plan"], fuel_prices, carbon_tax, demand_mult)
            pareto_item.update({
                "compliance_status": evaluated["compliance_status"],
                "compliance_forecast": evaluated["compliance_forecast"],
                "explainability": evaluated["explainability"],
                "cii_breakdown": evaluated["cii_breakdown"],
            })
        elapsed = round(time.time() - t0, 3)
        res["execution_time_seconds"] = elapsed
        results[name] = res

    total_elapsed = round(time.time() - start_time, 3)
    
    primary_name = list(results.keys())[0]
    primary_result = results[primary_name]
    
    return {
        "status": "success",
        "primary_algorithm": primary_name,
        "total_execution_time_seconds": total_elapsed,
        "result": primary_result,
        "all_results": results
    }
