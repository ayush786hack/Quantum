import numpy as np
import random

FUEL_OPTIONS = ["HFO", "LNG", "Methanol", "Hydrogen", "Ammonia"]

class QPSOSolver:
    """
    Quantum-Inspired Particle Swarm Optimization (QPSO) Multi-Objective Solver.
    Uses quantum delta potential well wave function position updates and mean-best (mbest) attractor.
    """
    def __init__(
        self,
        vessels: list,
        routes: list,
        pop_size: int = 40,
        generations: int = 35,
        alpha_coeff: float = 0.6,
        fuel_prices: dict = None,
        carbon_tax: float = 0.0,
        demand_mult: float = 1.0
    ):
        self.vessels = vessels
        self.routes = routes
        self.pop_size = pop_size
        self.generations = generations
        self.alpha_coeff = alpha_coeff
        self.fuel_prices = fuel_prices
        self.carbon_tax = carbon_tax
        self.demand_mult = demand_mult
        
        self.num_vars = len(vessels)
        # Each vessel has 3 decision variables: [speed_float, fuel_idx_float, shore_power_float]
        self.dims_per_vessel = 3
        self.dim = self.num_vars * self.dims_per_vessel
        
    def initialize_particles(self):
        """Initializes particle positions continuously in range [0, 1]."""
        return np.random.uniform(0.0, 1.0, (self.pop_size, self.dim))
        
    def decode_position(self, pos):
        """Decodes continuous particle vector into discrete fleet deployment plan."""
        fleet_plan = []
        
        for idx in range(self.num_vars):
            vessel = self.vessels[idx]
            route = self.routes[idx % len(self.routes)]
            
            p_idx = idx * self.dims_per_vessel
            s_val, f_val, sp_val = pos[p_idx], pos[p_idx+1], pos[p_idx+2]
            
            # Speed mapping
            base_s = vessel.get('base_speed_knots', 16.0)
            min_s, max_s = max(9.0, base_s * 0.7), min(24.0, base_s * 1.25)
            speed = min_s + np.clip(s_val, 0.0, 1.0) * (max_s - min_s)
            
            # Fuel mapping
            fuel_idx = int(np.clip(f_val * len(FUEL_OPTIONS), 0, len(FUEL_OPTIONS) - 1))
            fuel_type = FUEL_OPTIONS[fuel_idx]
            
            # Shore power mapping
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
        particles = self.initialize_particles()
        pbest_pos = np.copy(particles)
        pbest_fit = []
        pbest_plans = []
        
        for p in particles:
            plan = self.decode_position(p)
            res = evaluate_fn(plan, self.fuel_prices, self.carbon_tax, self.demand_mult)
            pbest_fit.append(res["fitness_vector"])
            pbest_plans.append(plan)
            
        convergence_history = []
        
        for gen in range(self.generations):
            # Compute mean best (mbest) position vector across swarm
            mbest = np.mean(pbest_pos, axis=0)
            
            # Find current global best index (min cost)
            gbest_idx = np.argmin([fit[0] for fit in pbest_fit])
            gbest_pos = pbest_pos[gbest_idx]
            
            min_cost = pbest_fit[gbest_idx][0]
            min_emissions = pbest_fit[gbest_idx][1]
            convergence_history.append({"generation": gen + 1, "min_cost": min_cost, "min_emissions": min_emissions})
            
            # QPSO Quantum wave function position updates
            for i in range(self.pop_size):
                phi = np.random.uniform(0.0, 1.0, self.dim)
                p_attractor = phi * pbest_pos[i] + (1.0 - phi) * gbest_pos
                
                u = np.random.uniform(0.0, 1.0, self.dim)
                u = np.clip(u, 1e-6, 1.0 - 1e-6)
                
                ln_u = np.log(1.0 / u)
                sign = np.where(np.random.uniform(0.0, 1.0, self.dim) < 0.5, 1.0, -1.0)
                
                # Quantum delta potential well update rule
                particles[i] = p_attractor + sign * self.alpha_coeff * np.abs(mbest - particles[i]) * ln_u
                particles[i] = np.clip(particles[i], 0.0, 1.0)
                
                # Evaluate new position
                new_plan = self.decode_position(particles[i])
                res = evaluate_fn(new_plan, self.fuel_prices, self.carbon_tax, self.demand_mult)
                new_fit = res["fitness_vector"]
                
                # Update personal best if cost is lower
                if new_fit[0] <= pbest_fit[i][0]:
                    pbest_pos[i] = np.copy(particles[i])
                    pbest_fit[i] = new_fit
                    pbest_plans[i] = new_plan

        # Form Pareto front from personal bests
        pareto_front = []
        for i in range(self.pop_size):
            pareto_front.append({
                "fitness": pbest_fit[i],
                "cost_usd": pbest_fit[i][0],
                "emissions_tco2e": pbest_fit[i][1],
                "delay_hours": pbest_fit[i][2],
                "fleet_plan": pbest_plans[i]
            })
            
        # Sort by cost
        pareto_front.sort(key=lambda x: x["cost_usd"])
        
        return {
            "algorithm": "QPSO (Quantum-Inspired PSO)",
            "pareto_front": pareto_front,
            "convergence_history": convergence_history,
            "best_solution": pbest_plans[0],
            "best_fitness": pbest_fit[0]
        }
