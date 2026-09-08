import numpy as np

def calculate_hypervolume(pareto_points: list, ref_cost: float = 2000000.0, ref_emissions: float = 15000.0) -> float:
    """
    Calculates 2D Hypervolume indicator for Pareto front (Cost vs Emissions).
    Higher hypervolume indicates superior solution quality and diversity.
    """
    if not pareto_points:
        return 0.0
        
    # Sort points by cost ascending
    sorted_pts = sorted(pareto_points, key=lambda p: p[0])
    
    hv = 0.0
    prev_cost = ref_cost
    
    for cost, emissions in sorted_pts:
        if cost >= ref_cost or emissions >= ref_emissions:
            continue
        width = max(0.0, ref_cost - cost)
        height = max(0.0, ref_emissions - emissions)
        hv += width * height
        prev_cost = cost
        
    # Normalized score between 0 and 100
    normalized_hv = min(100.0, (hv / (ref_cost * ref_emissions)) * 100.0 * 2.5)
    return round(normalized_hv, 2)

def calculate_convergence_generation(convergence_history: list, threshold_pct: float = 0.90) -> int:
    """Finds the generation number at which 90% convergence was reached."""
    if not convergence_history:
        return 0
        
    costs = [item["min_cost"] for item in convergence_history]
    initial_cost = costs[0]
    best_cost = min(costs)
    
    if initial_cost == best_cost:
        return 1
        
    target_cost = initial_cost - threshold_pct * (initial_cost - best_cost)
    
    for idx, c in enumerate(costs):
        if c <= target_cost:
            return idx + 1
            
    return len(convergence_history)

def summarize_algorithm_performance(algo_result: dict) -> dict:
    """Extracts summary metrics for an algorithm run."""
    pf = algo_result.get("pareto_front", [])
    history = algo_result.get("convergence_history", [])
    exec_time = algo_result.get("execution_time_seconds", 0.0)
    
    pts = [(item["cost_usd"], item["emissions_tco2e"]) for item in pf]
    
    min_cost = min((p[0] for p in pts), default=0.0)
    min_emissions = min((p[1] for p in pts), default=0.0)
    
    hv_score = calculate_hypervolume(pts)
    conv_gen = calculate_convergence_generation(history)
    
    return {
        "algorithm": algo_result.get("algorithm", "Unknown"),
        "min_cost_usd": min_cost,
        "min_emissions_tco2e": min_emissions,
        "hypervolume_score": hv_score,
        "convergence_generation": conv_gen,
        "execution_time_seconds": exec_time,
        "pareto_solutions_count": len(pf)
    }
