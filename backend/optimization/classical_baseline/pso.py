import numpy as np

FUEL_OPTIONS = ["HFO", "LNG", "Methanol", "Hydrogen", "Ammonia"]

class ClassicalPSOSolver:
    """
    Classical Particle Swarm Optimization (PSO) baseline solver.
    Uses position and velocity dynamics with inertia weight w, cognitive c1, and social c2.
    """
    def __init__(
        self,
        vessels: list,
        routes: list,
        pop_size: int = 40,
        generations: int = 35,
        w: float = 0.7,
        c1: float = 1.4,
        c2: float = 1.4,
        fuel_prices: dict = None,
        carbon_tax: float = 0.0,
        demand_mult: float = 1.0
    ):
        self.vessels = vessels
        self.routes = routes
        self.pop_size = pop_size
        self.generations = generations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.fuel_prices = fuel_prices
        self.carbon_tax = carbon_tax
        self.demand_mult = demand_mult
        
        self.num_vars = len(vessels)
        self.dims_per_vessel = 3
        self.dim = self.num_vars * self.dims_per_vessel

    def decode_position(self, pos):
        fleet_plan = []
        for idx in range(self.num_vars):
            vessel = self.vessels[idx]
            route = self.routes[idx % len(self.routes)]
            
            p_idx = idx * self.dims_per_vessel
            s_val, f_val, sp_val = pos[p_idx], pos[p_idx+1], pos[p_idx+2]
            
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
        pos = np.random.uniform(0.0, 1.0, (self.pop_size, self.dim))
        vel = np.random.uniform(-0.1, 0.1, (self.pop_size, self.dim))
        
        pbest_pos = np.copy(pos)
        pbest_fit = []
        pbest_plans = []
        
        for p in pos:
            plan = self.decode_position(p)
            res = evaluate_fn(plan, self.fuel_prices, self.carbon_tax, self.demand_mult)
            pbest_fit.append(res["fitness_vector"])
            pbest_plans.append(plan)
            
        convergence_history = []
        
        for gen in range(self.generations):
            gbest_idx = np.argmin([fit[0] for fit in pbest_fit])
            gbest_pos = pbest_pos[gbest_idx]
            
            min_cost = pbest_fit[gbest_idx][0]
            min_emissions = pbest_fit[gbest_idx][1]
            convergence_history.append({"generation": gen + 1, "min_cost": min_cost, "min_emissions": min_emissions})
            
            for i in range(self.pop_size):
                r1 = np.random.uniform(0.0, 1.0, self.dim)
                r2 = np.random.uniform(0.0, 1.0, self.dim)
                
                vel[i] = self.w * vel[i] + self.c1 * r1 * (pbest_pos[i] - pos[i]) + self.c2 * r2 * (gbest_pos - pos[i])
                vel[i] = np.clip(vel[i], -0.2, 0.2)
                
                pos[i] = np.clip(pos[i] + vel[i], 0.0, 1.0)
                
                new_plan = self.decode_position(pos[i])
                res = evaluate_fn(new_plan, self.fuel_prices, self.carbon_tax, self.demand_mult)
                new_fit = res["fitness_vector"]
                
                if new_fit[0] <= pbest_fit[i][0]:
                    pbest_pos[i] = np.copy(pos[i])
                    pbest_fit[i] = new_fit
                    pbest_plans[i] = new_plan

        pareto_front = []
        for i in range(self.pop_size):
            pareto_front.append({
                "fitness": pbest_fit[i],
                "cost_usd": pbest_fit[i][0],
                "emissions_tco2e": pbest_fit[i][1],
                "delay_hours": pbest_fit[i][2],
                "fleet_plan": pbest_plans[i]
            })
            
        pareto_front.sort(key=lambda x: x["cost_usd"])
        
        return {
            "algorithm": "Classical PSO",
            "pareto_front": pareto_front,
            "convergence_history": convergence_history,
            "best_solution": pbest_plans[0],
            "best_fitness": pbest_fit[0]
        }
