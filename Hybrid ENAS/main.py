# main.py

import numpy as np
import tensorflow as tf
from encoding import decode
from model_builder import get_model
from evaluation import evaluate_model, calculate_model_flops, count_params
from local_search import HillClimbing, TabuSearch, SimulatedAnnealing
from nsga3 import NSGA3
from config import EVALUATION_METRIC, LOCAL_SEARCH_CONFIG

class ModifiedMyPMOP:
    """
    The problem class that includes evaluation functions for the optimization problem.
    """
    def __init__(self, n_var=84, n_obj=3):
        self.n_var = n_var
        self.n_obj = n_obj
        self.xl = np.zeros(self.n_var)  # Lower bounds
        self.xu = np.ones(self.n_var)   # Upper bounds

    def evaluate(self, ind, n_eval):
        """
        Evaluate the individual and return the objective values.
        """
        # Decode the individual (bitstring to genotype)
        genotype = decode([ind])
        f1 = evaluate_model(genotype, n_eval)  # Evaluation metric (PSNR or SynFlow)
        f2 = self.func_eval_params(genotype)
        f3 = self.func_eval_flops(genotype)
        return [f1, f2, f3]

    def func_eval_params(self, genotype):
        """
        Calculate the number of parameters of the model.
        """
        model = get_model(genotype)
        params = count_params(model)
        tf.keras.backend.clear_session()
        return params

    def func_eval_flops(self, genotype):
        """
        Calculate the FLOPs of the model.
        """
        model = get_model(genotype)
        flops = calculate_model_flops(model)
        tf.keras.backend.clear_session()
        return flops

if __name__ == "__main__":
    # Initialize the problem
    problem = ModifiedMyPMOP()

    # Choose the algorithm
    algorithm_choice = 'NSGA-III'  # Options: 'HillClimbing', 'TabuSearch', 'SimulatedAnnealing', 'NSGA-III'

    if algorithm_choice == 'HillClimbing':
        # Generate the initial population of 20 individuals
        population_size = 20
        seeds = np.random.choice(range(1, 31), population_size, replace=False)
        population = []
        for seed in seeds:
            np.random.seed(seed)
            individual = np.random.randint(0, 2, size=problem.n_var)
            population.append(individual)

        optimizer = HillClimbing(problem, population)
        optimizer.optimize()
        best_solution = optimizer.best_solution
        best_objectives = optimizer.best_obj

        print("Best solution found:", best_solution)
        print("Objectives:", best_objectives)

    elif algorithm_choice == 'TabuSearch':
        # Similar setup as HillClimbing
        population_size = 20
        seeds = np.random.choice(range(1, 31), population_size, replace=False)
        population = []
        for seed in seeds:
            np.random.seed(seed)
            individual = np.random.randint(0, 2, size=problem.n_var)
            population.append(individual)

        optimizer = TabuSearch(problem, population)
        optimizer.optimize()
        best_solution = optimizer.best_solution
        best_objectives = optimizer.best_obj

        print("Best solution found:", best_solution)
        print("Objectives:", best_objectives)

    elif algorithm_choice == 'SimulatedAnnealing':
        # Similar setup as HillClimbing
        population_size = 20
        seeds = np.random.choice(range(1, 31), population_size, replace=False)
        population = []
        for seed in seeds:
            np.random.seed(seed)
            individual = np.random.randint(0, 2, size=problem.n_var)
            population.append(individual)

        optimizer = SimulatedAnnealing(problem, population)
        optimizer.optimize()
        best_solution = optimizer.best_solution
        best_objectives = optimizer.best_obj

        print("Best solution found:", best_solution)
        print("Objectives:", best_objectives)

    elif algorithm_choice == 'NSGA-III':
        # Set NSGA-III parameters
        pop_size = 100
        n_gen = 100  # Adjust the number of generations as needed

        optimizer = NSGA3(problem, pop_size=pop_size, n_gen=n_gen, verbose=True)
        pop, nds = optimizer.run()
        # You can process the final population and non-dominated solutions as needed

        # For example, print the non-dominated solutions
        print("Non-dominated solutions:")
        for ind, obj in zip(nds['X'], nds['F']):
            print(f"Solution: {ind}, Objectives: {obj}")

    else:
        raise ValueError("Invalid algorithm specified.")
