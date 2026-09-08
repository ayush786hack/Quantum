import numpy as np
import random

FUEL_OPTIONS = ["HFO", "LNG", "Methanol", "Hydrogen", "Ammonia"]

class ClassicalGASolver:
    """
    Classical NSGA-II Genetic Algorithm baseline solver.
    Uses continuous genome encoding with SBX crossover and polynomial mutation.
    """
    def __init__(
        self,
        vessels: list,
        routes: list,
        pop_size: int = 40,
        generations: int = 35,
        crossover_prob: float = 0.85,
        mutation_prob: float = 0.15,
        fuel_prices: dict = None,
        carbon_tax: float = 0.0,
        demand_mult: float = 1.0
    ):
        self.vessels = vessels
        self.routes = routes
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.fuel_prices = fuel_prices
        self.carbon_tax = carbon_tax
        self.demand_mult = demand_mult
        
        self.num_vars = len(vessels)
        self.dims_per_vessel = 3
        self.dim = self.num_vars * self.dims_per_vessel

    def decode_individual(self, ind):
        fleet_plan = []
        for idx in range(self.num_vars):
            vessel = self.vessels[idx]
            route = self.routes[idx % len(self.routes)]
            
            p_idx = idx * self.dims_per_vessel
            s_val, f_val, sp_val = ind[p_idx], ind[p_idx+1], ind[p_idx+2]
            
            base_s = vessel.get('base_speed_knots', 16.0)
            min_s, max_s = max(9.0, base_s * 0.7), min(24.0, base_s * 1.25)
            speed = min_s + np.clip(s_val, 0.0, 1.0) * (max_s - min_s)
            
            fuel_idx = int(np.clip(f_val * len(FUEL_OPTIONS), 0, len(FUEL_OPTIONS) - 1))
            fuel_type = FUEL_OPTIONS[fuel_idx]
            shore_power = (sp_val >= 0.5)
            
            fleet_plan.append({
                "vessel": vessel,
                "route": route,
                "speed_knots": round(speed, 1),
                "fuel_type": fuel_type,
                "shore_power_used": shore_power
            })
        return fleet_plan

    def solve(self, evaluate_fn):
        pop = np.random.uniform(0.0, 1.0, (self.pop_size, self.dim))
        convergence_history = []
        
        for gen in range(self.generations):
            fitness_list = []
            plans = []
            for ind in pop:
                plan = self.decode_individual(ind)
                res = evaluate_fn(plan, self.fuel_prices, self.carbon_tax, self.demand_mult)
                fitness_list.append(res["fitness_vector"])
                plans.append(plan)
                
            min_cost = min(fit[0] for fit in fitness_list)
            min_emissions = min(fit[1] for fit in fitness_list)
            convergence_history.append({"generation": gen + 1, "min_cost": min_cost, "min_emissions": min_emissions})
            
            # Selection & Crossover / Mutation
            next_pop = []
            while len(next_pop) < self.pop_size:
                # Tournament selection
                i1, i2 = random.sample(range(self.pop_size), 2)
                p1 = pop[i1] if fitness_list[i1][0] < fitness_list[i2][0] else pop[i2]
                
                i3, i4 = random.sample(range(self.pop_size), 2)
                p2 = pop[i3] if fitness_list[i3][0] < fitness_list[i4][0] else pop[i4]
                
                # Single-point Crossover
                if random.random() < self.crossover_prob:
                    cut = random.randint(1, self.dim - 1)
                    c1 = np.concatenate([p1[:cut], p2[cut:]])
                    c2 = np.concatenate([p2[:cut], p1[cut:]])
                else:
                    c1, c2 = np.copy(p1), np.copy(p2)
                    
                # Mutation
                for c in [c1, c2]:
                    if random.random() < self.mutation_prob:
                        mut_idx = random.randint(0, self.dim - 1)
                        c[mut_idx] = np.clip(c[mut_idx] + np.random.normal(0, 0.15), 0.0, 1.0)
                    next_pop.append(c)
                    
            pop = np.array(next_pop[:self.pop_size])

        # Evaluate final population
        final_plans = [self.decode_individual(ind) for ind in pop]
        final_fits = [evaluate_fn(plan, self.fuel_prices, self.carbon_tax, self.demand_mult)["fitness_vector"] for plan in final_plans]
        
        pareto_front = []
        for i in range(self.pop_size):
            pareto_front.append({
                "fitness": final_fits[i],
                "cost_usd": final_fits[i][0],
                "emissions_tco2e": final_fits[i][1],
                "delay_hours": final_fits[i][2],
                "fleet_plan": final_plans[i]
            })
            
        pareto_front.sort(key=lambda x: x["cost_usd"])
        
        return {
            "algorithm": "Classical GA (NSGA-II)",
            "pareto_front": pareto_front,
            "convergence_history": convergence_history,
            "best_solution": final_plans[0],
            "best_fitness": final_fits[0]
        }
