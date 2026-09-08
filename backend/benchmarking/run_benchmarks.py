import sys
import os
import json
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization.optimizer import run_fleet_optimization
from benchmarking.metrics import summarize_algorithm_performance

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def load_data():
    vessels_path = os.path.join(RAW_DIR, "vessels.json")
    routes_path = os.path.join(RAW_DIR, "routes.json")
    
    if not os.path.exists(vessels_path) or not os.path.exists(routes_path):
        from data.synthetic_generator import run as run_gen
        run_gen()
        
    with open(vessels_path) as f:
        vessels = json.load(f)
    with open(routes_path) as f:
        routes = json.load(f)
        
    return vessels, routes

def run_comparative_benchmark(fleet_size: int = 15, generations: int = 20):
    print(f"[Benchmark Suite] Starting comparative benchmark (Fleet Size = {fleet_size}, Generations = {generations})...")
    vessels, routes = load_data()
    test_vessels = vessels[:fleet_size]
    
    opt_res = run_fleet_optimization(
        vessels=test_vessels,
        routes=routes,
        algorithm="ALL",
        pop_size=35,
        generations=generations
    )
    
    all_results = opt_res["all_results"]
    summary_table = []
    
    for name, res in all_results.items():
        summary = summarize_algorithm_performance(res)
        summary_table.append(summary)
        
    # Scalability Benchmark (Runtime vs Fleet Size)
    fleet_sizes = [5, 10, 15, 20]
    scalability_data = []
    
    for fs in fleet_sizes:
        sub_vessels = vessels[:fs]
        scal_entry = {"fleet_size": fs}
        
        for algo in ["QIGA", "QPSO", "GA", "PSO"]:
            t0 = time.time()
            _ = run_fleet_optimization(sub_vessels, routes, algorithm=algo, pop_size=20, generations=12)
            elapsed = round(time.time() - t0, 3)
            scal_entry[f"runtime_{algo.lower()}"] = elapsed
            
        scalability_data.append(scal_entry)
        
    benchmark_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fleet_size_tested": fleet_size,
        "generations_tested": generations,
        "summary_comparison": summary_table,
        "scalability_benchmark": scalability_data,
        "detailed_results": {name: {
            "algorithm": res["algorithm"],
            "execution_time_seconds": res["execution_time_seconds"],
            "convergence_history": res["convergence_history"],
            "pareto_front": res["pareto_front"][:10]
        } for name, res in all_results.items()}
    }
    
    clean_report = sanitize_for_json(benchmark_report)
    
    output_path = os.path.join(BASE_DIR, "data", "processed", "benchmark_report.json")
    with open(output_path, "w") as f:
        json.dump(clean_report, f, indent=2)
        
    print(f"[Benchmark Suite] Done! Report saved to {output_path}")
    return clean_report

if __name__ == "__main__":
    run_comparative_benchmark(fleet_size=12, generations=15)
