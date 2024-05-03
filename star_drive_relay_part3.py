import numpy as np
import matplotlib.pyplot as plt
from deap import algorithms, base, creator, tools
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from platypus import NSGAII, Problem, Real
from star_drive_relay_part2 import StarDriveRelaySimulation

class StarDriveRelayOptimization(StarDriveRelaySimulation):
    def objective_function(self, parameters):
        results = self.simulate(*parameters)
        return [
            -results["delta_v_achieved"].value,
            results["energy_consumed"].value,
            results["interaction_time"].value,
        ]

    def generate_parameters(self, parameter_ranges):
        return [np.random.uniform(low, high) for low, high in parameter_ranges]

    def evaluate_optimization_criteria(self, results, optimization_criteria):
        score = 0
        for criterion, weight in optimization_criteria.items():
            score += weight * getattr(results, criterion).value
        return score

    def multi_objective_optimization(self, desired_delta_v, efficiency, parameter_ranges, max_iterations):
        problem = Problem(len(parameter_ranges), 3)
        problem.types[:] = [Real(low, high) for low, high in parameter_ranges]
        problem.function = self.objective_function
        algorithm = NSGAII(problem)
        algorithm.run(max_iterations)
        return algorithm.result

    def plot_pareto_front(self, pareto_front):
        plt.figure(figsize=(8, 6))
        for solution in pareto_front:
            plt.plot(-solution.objectives[0], solution.objectives[1], "ro")
        plt.xlabel("Delta-V (km/s)")
        plt.ylabel("Energy Consumed (Joules)")
        plt.title("Pareto Front")
        plt.show()

    def random_forest_surrogate(self, parameters, results, test_size=0.2, n_estimators=100):
        X = np.array(parameters)
        y = np.array([result["delta_v_achieved"].value for result in results])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        rf.fit(X_train, y_train)
        score = rf.score(X_test, y_test)
        return rf, score

    def genetic_algorithm(self, desired_delta_v, efficiency, optimization_criteria, parameter_ranges, max_iterations):
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

    def simulated_annealing(self, desired_delta_v, efficiency, optimization_criteria, parameter_ranges, max_iterations):
        import simanneal

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

    def particle_swarm_optimization(self, desired_delta_v, efficiency, optimization_criteria, parameter_ranges, max_iterations):
        import pyswarms as ps

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

    def select_algorithm(self, algorithm_name, desired_delta_v, efficiency, optimization_criteria, parameter_ranges, max_iterations):
        if algorithm_name == "Genetic Algorithm":
            return self.genetic_algorithm(desired_delta_v, efficiency, optimization_criteria, parameter_ranges, max_iterations)
        elif algorithm_name == "Simulated Annealing":
            return self.simulated_annealing(desired_delta_v, efficiency, optimization_criteria, parameter_ranges, max_iterations)
        elif algorithm_name == "Particle Swarm Optimization":
            return self.particle_swarm_optimization(desired_delta_v, efficiency, optimization_criteria, parameter_ranges, max_iterations)
        else:
            raise ValueError(f"Unsupported optimization algorithm: {algorithm_name}")

# Usage example
if __name__ == "__main__":
    # Example usage of the class
    optimization = StarDriveRelayOptimization(
        # ... initialize with required parameters ...
    )

    # Set up the optimization criteria
    optimization_criteria = {
        "delta_v_achieved": 1.0,  # weight for delta-v
        "energy_consumed": 1.0,   # weight for energy consumption
        "interaction_time": 1.0   # weight for interaction time
    }

    # Call an optimization algorithm
    desired_delta_v = 10000 * u.m / u.s  # Example desired delta-v
    efficiency = 80  # Example efficiency in percent
    parameter_ranges = [(1, 1000), (0.1, 10), (0.5, 5), (0.1, 1)]  # Define parameter ranges
    max_iterations = 100  # Define the number of iterations
    
    # Genetic Algorithm
    print("Running Genetic Algorithm...")
    ga_solution, ga_score = optimization.select_algorithm("Genetic Algorithm", desired_delta_v, efficiency, optimization_criteria, parameter_ranges, max_iterations)
    print("Genetic Algorithm - Best solution:", ga_solution)
    print("Genetic Algorithm - Best score:", ga_score)
    
    # Simulated Annealing
    print("\nRunning Simulated Annealing...")
    sa_solution, sa_score = optimization.select_algorithm("Simulated Annealing", desired_delta_v, efficiency, optimization_criteria, parameter_ranges, max_iterations)
    print("Simulated Annealing - Best solution:", sa_solution)
    print("Simulated Annealing - Best score:", sa_score)
    
    # Particle Swarm Optimization
    print("\nRunning Particle Swarm Optimization...")
    pso_solution, pso_score = optimization.select_algorithm("Particle Swarm Optimization", desired_delta_v, efficiency, optimization_criteria, parameter_ranges, max_iterations)
    print("Particle Swarm Optimization - Best solution:", pso_solution)
    print("Particle Swarm Optimization - Best score:", pso_score)
    
    # Multi-Objective Optimization
    print("\nRunning Multi-Objective Optimization...")
    pareto_front = optimization.multi_objective_optimization(desired_delta_v, efficiency, parameter_ranges, max_iterations)
    print("Multi-Objective Optimization - Pareto Front:")
    for solution in pareto_front:
        print(solution)
    
    # Plot the Pareto Front
    optimization.plot_pareto_front(pareto_front)
    
    # Random Forest Surrogate
    print("\nRunning Random Forest Surrogate...")
    parameters = [optimization.generate_parameters(parameter_ranges) for _ in range(100)]
    results = [optimization.simulate(desired_delta_v, efficiency, *params) for params in parameters]
    rf, score = optimization.random_forest_surrogate(parameters, results)
    print("Random Forest Surrogate - Score:", score)