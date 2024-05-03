import numpy as np
from astropy import units as u
from platypus import NSGAII, Problem, Real
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from deap import creator, base, tools, algorithms
import pyswarms as ps
import simanneal
import logging

class StarDriveRelayOptimization(StarDriveRelaySimulation):
    def __init__(self, num_rings, spacecraft_mass, ring_diameters, ring_thicknesses,
                 magnetic_fields, electric_fields, current_densities, em_losses,
                 plasma_losses, g_force_manned, g_force_unmanned, initial_velocity,
                 pulse_frequency, pulse_duration, mhd_coefficient, parameter_ranges,
                 spacecraft_mass_max, ring_diameters_min, ring_diameters_max):
        super().__init__(num_rings, spacecraft_mass, ring_diameters, ring_thicknesses,
                         magnetic_fields, electric_fields, current_densities, em_losses,
                         plasma_losses, g_force_manned, g_force_unmanned, initial_velocity,
                         pulse_frequency, pulse_duration, mhd_coefficient)
        self.parameter_ranges = parameter_ranges
        self.spacecraft_mass_max = spacecraft_mass_max
        self.ring_diameters_min = ring_diameters_min
        self.ring_diameters_max = ring_diameters_max

    def objective_function(self, parameters):
        try:
            # Ensure desired_delta_v has the correct units (velocity, m/s)
            desired_delta_v = 100 * u.m / u.s

            results = self.simulate(desired_delta_v, efficiency, *parameters)
            if "velocity" in results and len(results["velocity"]) > 0:
                return [
                    -results["power_requirements_manned"].value,
                    -results["thermal_management_manned"].value,
                    results["interaction_time_manned"].value,
                    -results["debye_length_manned"].value,  # Minimize Debye length (plasma interaction)
                    results["temperature_manned"].value,  # Minimize temperature (thermal effects)
                    -results["orientation_manned"][2]  # Maximize orientation along z-axis (g-force limit)
                ]
            else:
                logging.error("Simulation results are empty or invalid.")
                return [np.inf, np.inf, np.inf, np.inf, np.inf, -np.inf]
        except Exception as e:
            logging.error(f"Error in objective_function: {e}")
            return [np.inf, np.inf, np.inf, np.inf, np.inf, -np.inf]


    def unpack_solution(self, solution):
        parameters = [
            solution.variables[0] * u.kg,
            np.array([solution.variables[1]]) * u.m,
            np.array([solution.variables[2]]) * u.m,
            np.array([solution.variables[3]]) * u.T,
            np.array([solution.variables[4]]) * u.V / u.m,
            np.array([solution.variables[5]]) * u.A / u.m**2,
            solution.variables[6],
            solution.variables[7],
            solution.variables[8],
            solution.variables[9],
            solution.variables[10] * u.m / u.s,
            solution.variables[11] * u.Hz,
            solution.variables[12] * u.s,
            solution.variables[13],
        ]
        return parameters

    def optimize(self, desired_delta_v, efficiency, max_iterations=100):
        self.desired_delta_v = desired_delta_v
        self.efficiency = efficiency

        # Define the optimization problem using Platypus and NSGA-II
        problem = Problem(len(self.parameter_ranges), 3)
        problem.types[:] = [Real(low, high) for low, high in self.parameter_ranges]

        # Define the objective functions
        def obj_func_1(solution):
            parameters = self.unpack_solution(solution)
            results = self.simulate(desired_delta_v, efficiency, *parameters)
            return [-results["power_requirements_manned"].value]

        def obj_func_2(solution):
            parameters = self.unpack_solution(solution)
            results = self.simulate(desired_delta_v, efficiency, *parameters)
            return [-results["thermal_management_manned"].value]

        def obj_func_3(solution):
            parameters = self.unpack_solution(solution)
            results = self.simulate(desired_delta_v, efficiency, *parameters)
            return [results["interaction_time_manned"].value]

        problem.function = lambda solution: [obj_func_1(solution), obj_func_2(solution), obj_func_3(solution)]

        # Define constraints
        problem.constraints[:] = [
            lambda solution: self.spacecraft_mass_max - solution.variables[0],  # Spacecraft mass constraint
            lambda solution: solution.variables[1] - self.ring_diameters_min,  # Ring diameter constraint (lower bound)
            lambda solution: self.ring_diameters_max - solution.variables[1],  # Ring diameter constraint (upper bound)
            # Add more constraints as needed
        ]

        algorithm = NSGAII(problem)
        algorithm.run(max_iterations)
        return algorithm.result

    def plot_pareto_front(self, pareto_front):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        for solution in pareto_front:
            plt.plot(solution.objectives[0], solution.objectives[1], "ro")
        plt.xlabel("Power Requirements (W)")
        plt.ylabel("Thermal Management (W)")
        plt.title("Pareto Front")
        plt.show()

    def random_forest_surrogate(self, parameters, results, test_size=0.2, n_estimators=100):
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor

        try:
            X = np.array(parameters)
            y = np.array([result["power_requirements_manned"].value for result in results])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
            rf.fit(X_train, y_train)
            score = rf.score(X_test, y_test)
            return rf, score
        except Exception as e:
            logging.error(f"Error in random_forest_surrogate: {e}")
            return None, None

    def genetic_algorithm(self, desired_delta_v, efficiency, optimization_criteria, parameter_ranges, max_iterations):
        import deap
        from deap import creator, base, tools, algorithms

        try:
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)

            toolbox = base.Toolbox()
            toolbox.register("attr_float", np.random.uniform, parameter_ranges[0][0], parameter_ranges[0][1])
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(parameter_ranges))
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            def evaluate_individual(individual):
                parameters = individual
                results = self.simulate(desired_delta_v, efficiency, *parameters)
                score = self.evaluate_optimization_criteria(results, optimization_criteria)
                return score,

            toolbox.register("evaluate", evaluate_individual)
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
            toolbox.register("select", tools.selTournament, tournsize=3)

            population = toolbox.population(n=100)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)

            population, _ = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=max_iterations, stats=stats, verbose=True)

            best_individual = tools.selBest(population, k=1)[0]
            best_solution = [gene for gene in best_individual]
            best_score = best_individual.fitness.values[0]

            return best_solution, best_score
        except Exception as e:
            logging.error(f"Error in genetic_algorithm: {e}")
            return [], np.inf

    def simulated_annealing(self, desired_delta_v, efficiency, optimization_criteria, parameter_ranges, max_iterations):
        import simanneal

        try:
            def objective_function(parameters):
                results = self.simulate(desired_delta_v, efficiency, *parameters)
                score = self.evaluate_optimization_criteria(results, optimization_criteria)
                return score

            initial_parameters = self.generate_parameters(parameter_ranges)
            initial_score = objective_function(initial_parameters)

            def random_neighbor(parameters):
                neighbor = parameters.copy()
                index = np.random.randint(len(neighbor))
                neighbor[index] = np.random.uniform(parameter_ranges[index][0], parameter_ranges[index][1])
                return neighbor

            annealer = simanneal.Annealer(objective_function, initial_parameters, random_neighbor, initial_score)
            schedule = annealer.auto(max_iterations)
            best_parameters, best_score = annealer.anneal(schedule)

            best_solution = list(best_parameters)

            return best_solution, best_score
        except Exception as e:
            logging.error(f"Error in simulated_annealing: {e}")
            return [], np.inf

    def particle_swarm_optimization(self, desired_delta_v, efficiency, optimization_criteria, parameter_ranges, max_iterations):
        import pyswarms as ps

        try:
            def objective_function(parameters):
                results = self.simulate(desired_delta_v, efficiency, *parameters)
                score = self.evaluate_optimization_criteria(results, optimization_criteria)
                return -score  # Pyswarms minimizes the objective function

            bounds = [(low, high) for low, high in parameter_ranges]
            options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

            optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=len(parameter_ranges), options=options, bounds=bounds)
            cost, pos = optimizer.optimize(objective_function, iters=max_iterations)

            best_solution = pos
            best_score = -cost

            return best_solution, best_score
        except Exception as e:
            logging.error(f"Error in particle_swarm_optimization: {e}")
            return [], np.inf

    def evaluate_optimization_criteria(self, results, optimization_criteria):
        try:
            score = 0
            for criterion, weight in optimization_criteria.items():
                score += weight * results[criterion]
            return score
        except Exception as e:
            logging.error(f"Error in evaluate_optimization_criteria: {e}")
            return np.inf

    def generate_parameters(self, parameter_ranges):
        try:
            return [np.random.uniform(low, high) for low, high in parameter_ranges]
        except Exception as e:
            logging.error(f"Error in generate_parameters: {e}")
            return []

    def select_algorithm(self, algorithm_name, desired_delta_v, efficiency, optimization_criteria, parameter_ranges, max_iterations):
        if algorithm_name == "Genetic Algorithm":
            return self.genetic_algorithm(desired_delta_v, efficiency, optimization_criteria, parameter_ranges, max_iterations)
        elif algorithm_name == "Simulated Annealing":
            return self.simulated_annealing(desired_delta_v, efficiency, optimization_criteria, parameter_ranges, max_iterations)
        elif algorithm_name == "Particle Swarm Optimization":
            return self.particle_swarm_optimization(desired_delta_v, efficiency, optimization_criteria, parameter_ranges, max_iterations)
        else:
            raise ValueError(f"Unsupported optimization algorithm: {algorithm_name}")