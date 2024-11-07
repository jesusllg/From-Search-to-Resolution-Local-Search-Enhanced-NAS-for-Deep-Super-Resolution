# local_search.py

import numpy as np
from evaluation import evaluate_model
from utils import soft_pareto_dominates
from config import LOCAL_SEARCH_CONFIG

class LocalSearchBase:
    """
    Base class for local search algorithms.
    """
    def __init__(self, problem, population):
        self.problem = problem
        self.population = population
        self.max_evaluations = LOCAL_SEARCH_CONFIG['MAX_EVALUATIONS']
        self.evaluation_counter = 0
        self.current_solution = None
        self.current_obj = None
        self.best_solution = None
        self.best_obj = None

    def select_random_individual(self):
        """
        Select a random individual from the population for refinement.
        """
        self.current_solution = self.population[np.random.randint(len(self.population))]
        self.current_obj = self.problem.evaluate(self.current_solution, self.evaluation_counter)
        self.evaluation_counter += 1  # Increment evaluation counter
        self.best_solution = self.current_solution.copy()
        self.best_obj = self.current_obj.copy()

    def mutate(self, solution):
        """
        Mutate the solution by flipping bits, altering no more than 25% of the bits.
        """
        num_bits = len(solution)
        max_mutations = int(0.25 * num_bits)
        num_mutations = np.random.randint(1, max_mutations + 1)
        mutation_indices = np.random.choice(num_bits, num_mutations, replace=False)
        neighbor = solution.copy()
        neighbor[mutation_indices] = 1 - neighbor[mutation_indices]
        return neighbor

    def optimize(self):
        """
        Perform the local search optimization.
        """
        evaluations_per_individual = 1000
        while self.evaluation_counter < self.max_evaluations:
            # Every 1000 evaluations, select a new random individual
            if self.evaluation_counter % evaluations_per_individual == 0 or self.current_solution is None:
                self.select_random_individual()

            # Perform the search steps
            self.search()

    def search(self):
        """
        Perform the search steps. To be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

class HillClimbing(LocalSearchBase):
    """
    Hill Climbing local search algorithm for multi-objective optimization.
    """
    def __init__(self, problem, population):
        super().__init__(problem, population)
        self.assessment_interval = 10  # Assess every 10 steps

    def search(self):
        """
        Perform hill climbing search steps.
        """
        steps = 0
        while steps < 1000 and self.evaluation_counter < self.max_evaluations:
            neighbor = self.mutate(self.current_solution)
            neighbor_obj = self.problem.evaluate(neighbor, self.evaluation_counter)
            self.evaluation_counter += 1
            steps += 1

            # Use Pareto optimality to measure individual improvement
            if soft_pareto_dominates(neighbor_obj, self.current_obj):
                self.current_solution = neighbor
                self.current_obj = neighbor_obj

            # Assessments conducted every 10 steps
            if steps % self.assessment_interval == 0:
                # Evaluate the current solution
                current_eval_obj = self.current_obj

                # Compare with the best solution found so far
                if soft_pareto_dominates(current_eval_obj, self.best_obj):
                    self.best_solution = self.current_solution.copy()
                    self.best_obj = current_eval_obj.copy()

                # Optionally, record or log the evaluation
                print(f"Evaluation {self.evaluation_counter}: Best Objectives = {self.best_obj}")

class TabuSearch(LocalSearchBase):
    """
    Tabu Search local search algorithm for multi-objective optimization.
    """
    def __init__(self, problem, population):
        super().__init__(problem, population)
        self.tabu_tenure = LOCAL_SEARCH_CONFIG['TABU_TENURE']
        self.tabu_list = {}
        self.assessment_interval = 10  # Assess every 10 steps

    def search(self):
        """
        Perform tabu search steps.
        """
        steps = 0
        while steps < 1000 and self.evaluation_counter < self.max_evaluations:
            neighbor = self.mutate(self.current_solution)
            # Generate a key for the neighbor (e.g., hash of the solution)
            neighbor_key = tuple(neighbor.tolist())
            if neighbor_key in self.tabu_list:
                # Skip this neighbor if it's in the tabu list
                steps += 1
                continue

            neighbor_obj = self.problem.evaluate(neighbor, self.evaluation_counter)
            self.evaluation_counter += 1
            steps += 1

            if soft_pareto_dominates(neighbor_obj, self.current_obj):
                self.current_solution = neighbor
                self.current_obj = neighbor_obj
                # Add to tabu list
                self.tabu_list[neighbor_key] = self.tabu_tenure
            else:
                # Add to tabu list even if not improved
                self.tabu_list[neighbor_key] = self.tabu_tenure

            # Decrease tabu tenure
            self.tabu_list = {k: v - 1 for k, v in self.tabu_list.items() if v > 1}

            # Assessments conducted every 10 steps
            if steps % self.assessment_interval == 0:
                # Evaluate the current solution
                current_eval_obj = self.current_obj

                # Compare with the best solution found so far
                if soft_pareto_dominates(current_eval_obj, self.best_obj):
                    self.best_solution = self.current_solution.copy()
                    self.best_obj = current_eval_obj.copy()

                # Optionally, record or log the evaluation
                print(f"Evaluation {self.evaluation_counter}: Best Objectives = {self.best_obj}")

class SimulatedAnnealing(LocalSearchBase):
    """
    Simulated Annealing local search algorithm for multi-objective optimization.
    """
    def __init__(self, problem, population):
        super().__init__(problem, population)
        self.temperature = LOCAL_SEARCH_CONFIG['INITIAL_TEMP']
        self.cooling_rate = LOCAL_SEARCH_CONFIG['COOLING_RATE']
        self.assessment_interval = 10  # Assess every 10 steps

    def acceptance_probability(self, delta):
        """
        Calculate the acceptance probability for worse solutions.
        """
        return np.exp(-delta / self.temperature)

    def search(self):
        """
        Perform simulated annealing search steps.
        """
        steps = 0
        while steps < 1000 and self.evaluation_counter < self.max_evaluations:
            neighbor = self.mutate(self.current_solution)
            neighbor_obj = self.problem.evaluate(neighbor, self.evaluation_counter)
            self.evaluation_counter += 1
            steps += 1

            # Compute the difference in objectives (aggregate difference)
            delta = sum([n - c for n, c in zip(neighbor_obj, self.current_obj)])

            if delta < 0 or np.random.rand() < self.acceptance_probability(delta):
                self.current_solution = neighbor
                self.current_obj = neighbor_obj

            # Update temperature
            self.temperature *= self.cooling_rate
            if self.temperature < 1e-3:
                self.temperature = LOCAL_SEARCH_CONFIG['INITIAL_TEMP']  # Reset temperature

            # Assessments conducted every 10 steps
            if steps % self.assessment_interval == 0:
                # Evaluate the current solution
                current_eval_obj = self.current_obj

                # Compare with the best solution found so far
                if soft_pareto_dominates(current_eval_obj, self.best_obj):
                    self.best_solution = self.current_solution.copy()
                    self.best_obj = current_eval_obj.copy()

                # Optionally, record or log the evaluation
                print(f"Evaluation {self.evaluation_counter}: Best Objectives = {self.best_obj}")
