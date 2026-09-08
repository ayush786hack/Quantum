import numpy as np
import random

FUEL_OPTIONS = ["HFO", "LNG", "Methanol", "Hydrogen", "Ammonia"]

class QIGASolver:
    """
    Quantum-Inspired Genetic Algorithm (QIGA) Multi-Objective Solver.
    Uses Qubit representation [alpha, beta], quantum rotation gates U(delta_theta),
    and non-dominated sorting (NSGA-II style) to return a Pareto front.
    """
    
    def __init__(
        self,
        vessels: list,
        routes: list,
        pop_size: int = 40,
        generations: int = 35,
        rotation_step: float = 0.05 * np.pi,
        fuel_prices: dict = None,
        carbon_tax: float = 0.0,
        demand_mult: float = 1.0
    ):
        self.vessels = vessels
        self.routes = routes
        self.pop_size = pop_size
        self.generations = generations
        self.rotation_step = rotation_step
        self.fuel_prices = fuel_prices
        self.carbon_tax = carbon_tax
        self.demand_mult = demand_mult
        
        self.num_vars = len(vessels)
        # Each vessel decision has: speed bits (4), fuel bits (3), shore power bit (1) => 8 qubits per vessel
        self.qubits_per_vessel = 8
        self.total_qubits = self.num_vars * self.qubits_per_vessel
        
    def initialize_qpopulation(self):
        """Initializes qubit population into equal superposition (alpha = beta = 1/sqrt(2))."""
        q_pop = np.ones((self.pop_size, self.total_qubits, 2), dtype=float) / np.sqrt(2.0)
        return q_pop

    def observe_qindividual(self, q_ind):
        """Collapses quantum states to classical binary genome using probability |alpha|^2."""
        binary_genome = []
        for q in q_ind:
            alpha, beta = q[0], q[1]
            prob_zero = alpha ** 2
            if random.random() < prob_zero:
                binary_genome.append(0)
            else:
                binary_genome.append(1)
        return np.array(binary_genome)

    def decode_genome(self, binary_genome):
        """Decodes binary string into fleet deployment plan."""
        fleet_plan = []
        
        for idx in range(self.num_vars):
            vessel = self.vessels[idx]
            route = self.routes[idx % len(self.routes)]
            
            start_bit = idx * self.qubits_per_vessel
            bits = binary_genome[start_bit : start_bit + self.qubits_per_vessel]
            
            # Speed (4 bits -> int 0..15 mapped to range [min_speed, max_speed])
            speed_val = bits[0]*8 + bits[1]*4 + bits[2]*2 + bits[3]*1
            base_s = vessel.get('base_speed_knots', 16.0)
            min_s, max_s = max(9.0, base_s * 0.7), min(24.0, base_s * 1.25)
            speed = min_s + (speed_val / 15.0) * (max_s - min_s)
            
            # Fuel type (3 bits -> 0..4 index)
            fuel_val = (bits[4]*4 + bits[5]*2 + bits[6]*1) % len(FUEL_OPTIONS)
            fuel_type = FUEL_OPTIONS[fuel_val]
            
            # Shore power (1 bit)
            shore_power = (bits[7] == 1)
            
            fleet_plan.append({
                "vessel": vessel,
                "route": route,
                "speed_knots": round(speed, 1),
                "fuel_type": fuel_type,
                "shore_power_used": shore_power
            })
            
        return fleet_plan

    def apply_rotation_gate(self, q_ind, binary_genome, best_binary):
        """Updates qubit amplitudes using quantum rotation gate U(delta_theta)."""
        updated_q_ind = np.copy(q_ind)
        
        for i in range(len(q_ind)):
            x_i = binary_genome[i]
            b_i = best_binary[i]
            alpha, beta = q_ind[i][0], q_ind[i][1]
            
            if x_i != b_i:
                # Determine rotation direction
                if x_i == 0 and b_i == 1:
                    delta_theta = self.rotation_step if (alpha * beta > 0) else -self.rotation_step
                else:
                    delta_theta = -self.rotation_step if (alpha * beta > 0) else self.rotation_step
            else:
                delta_theta = 0.0
                
            # Rotation matrix multiplication
            new_alpha = alpha * np.cos(delta_theta) - beta * np.sin(delta_theta)
            new_beta = alpha * np.sin(delta_theta) + beta * np.cos(delta_theta)
            
            # Normalize state vector
            norm = np.sqrt(new_alpha**2 + new_beta**2)
            updated_q_ind[i][0] = new_alpha / norm
            updated_q_ind[i][1] = new_beta / norm
            
        return updated_q_ind

    def evaluate_population(self, q_pop, evaluate_fn):
        classical_pop = []
        fitness_list = []
        fleet_plans = []
        
        for ind in q_pop:
            bin_genome = self.observe_qindividual(ind)
            fleet_plan = self.decode_genome(bin_genome)
            res = evaluate_fn(fleet_plan, self.fuel_prices, self.carbon_tax, self.demand_mult)
            
            classical_pop.append(bin_genome)
            fitness_list.append(res["fitness_vector"])
            fleet_plans.append(fleet_plan)
            
        return classical_pop, fitness_list, fleet_plans

    def non_dominated_sort(self, fitness_list):
        """Performs NSGA-II non-dominated sorting returning Pareto fronts."""
        n_pop = len(fitness_list)
        domination_counts = [0] * n_pop
        dominated_solutions = [[] for _ in range(n_pop)]
        fronts = [[]]
        
        for i in range(n_pop):
            for j in range(n_pop):
                if i == j:
                    continue
                # Pareto dominance check (Cost, Emissions, Delay)
                fit_i = fitness_list[i]
                fit_j = fitness_list[j]
                
                i_dominates_j = (
                    all(fit_i[k] <= fit_j[k] for k in range(3)) and
                    any(fit_i[k] < fit_j[k] for k in range(3))
                )
                
                if i_dominates_j:
                    dominated_solutions[i].append(j)
                elif all(fit_j[k] <= fit_i[k] for k in range(3)) and any(fit_j[k] < fit_i[k] for k in range(3)):
                    domination_counts[i] += 1
                    
            if domination_counts[i] == 0:
                fronts[0].append(i)
                
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in dominated_solutions[p]:
                    domination_counts[q] -= 1
                    if domination_counts[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
            
        return [f for f in fronts if len(f) > 0]

    def solve(self, evaluate_fn):
        """Executes QIGA multi-objective optimization loop."""
        q_pop = self.initialize_qpopulation()
        convergence_history = []
        best_overall_plan = None
        best_overall_fitness = None
        
        for gen in range(self.generations):
            classical_pop, fitness_list, fleet_plans = self.evaluate_population(q_pop, evaluate_fn)
            fronts = self.non_dominated_sort(fitness_list)
            
            first_front = fronts[0]
            best_idx = first_front[0]
            best_binary = classical_pop[best_idx]
            
            min_cost = min(fit[0] for fit in fitness_list)
            min_emissions = min(fit[1] for fit in fitness_list)
            convergence_history.append({"generation": gen + 1, "min_cost": min_cost, "min_emissions": min_emissions})
            
            if best_overall_fitness is None or fitness_list[best_idx][0] < best_overall_fitness[0]:
                best_overall_fitness = fitness_list[best_idx]
                best_overall_plan = fleet_plans[best_idx]
                
            # Quantum Rotation Gate Update across population
            for i in range(self.pop_size):
                q_pop[i] = self.apply_rotation_gate(q_pop[i], classical_pop[i], best_binary)
                
        # Generate final Pareto front items
        final_classical_pop, final_fitness_list, final_plans = self.evaluate_population(q_pop, evaluate_fn)
        final_fronts = self.non_dominated_sort(final_fitness_list)
        
        pareto_front = []
        for idx in final_fronts[0]:
            pareto_front.append({
                "fitness": final_fitness_list[idx],
                "cost_usd": final_fitness_list[idx][0],
                "emissions_tco2e": final_fitness_list[idx][1],
                "delay_hours": final_fitness_list[idx][2],
                "fleet_plan": final_plans[idx]
            })
            
        return {
            "algorithm": "QIGA (Quantum-Inspired GA)",
            "pareto_front": pareto_front,
            "convergence_history": convergence_history,
            "best_solution": best_overall_plan,
            "best_fitness": best_overall_fitness
        }
