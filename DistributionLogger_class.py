
# While running the evolutionary algorithm (NSGA2 in our case), we need to record kktpm 
# and GD values to be able to analyze the results. This is done by the class DistributionLogger.
# The __call__ method is usually called six time per one evolutionary algorithm.



import pandas
import numpy as np

from apbi_proximity_measure import apbi_kktpm
from pbi_proximity_measure import pbi_kktpm


def generational_distance(approximation_set, pareto_front, p=2):
        """
        Compute the Generational Distance (GD) between an approximation set
        and the true Pareto front.

        Parameters
        ----------
        approximation_set : np.ndarray, shape (n_points, n_objectives)
        pareto_front : np.ndarray, shape (m_points, n_objectives)
        p : int, default=2
            Power for Minkowski distance (p=2 => Euclidean).

        Returns
        -------
        float
            The Generational Distance.
        """
        distances = []
        for point in approximation_set:
            # distance to the closest point in the Pareto front
            min_dist = np.min(np.linalg.norm(pareto_front - point, axis=1))
            distances.append(min_dist ** p)
        return (np.sum(distances) / len(approximation_set)) ** (1 / p)


        
        
class DistributionLogger:    
    """
    Custom callback to log and store intermediate results every 'part_interval' steps.
    """
    def __init__(self, part_interval, problem_instance, problem, phi_func, g_functions):
        self.data = []            # List of tuples (iteration, f1_list, f2_list)
        self.KKTPM_measures = []  # List of tuples (iteration, [kktpm_values...])
        self.pbi_kktpm_measures = []  # List of tuples (iteration, [kktpm_values...])
        self.GD_measures = []     # List of tuples (iteration, gd_value)
        self.part_interval = part_interval
        self.problem_instance = problem_instance
        self.gradients = []
        self.problem = problem
        self.phi_func = phi_func
        self.g_functions = g_functions

    def __call__(self, algorithm):

        # Check if we should log at this generation
        current_evals = algorithm.evaluator.n_eval
        if (current_evals / len(algorithm.pop)) % self.part_interval == 0:
            iteration = algorithm.n_gen
            population = algorithm.pop.get("X")  # Decision vectors
            population_values = algorithm.pop.get("F")  # Objective values
            # Attempt to get the Pareto front
            pf = self.problem_instance.pareto_front()
            print("Pareto front is: ", pf)
            # For plotting: if 1D objective, treat second as 0
            if self.problem_instance.n_obj == 1:
                f1 = [val[0] for val in population_values]
                f2 = [0] * len(population_values)
            else:
                f1 = [val[0] for val in population_values]
                f2 = [val[1] for val in population_values]
            self.data.append((iteration, f1, f2))
            # Print number of objectives for debugging
            print(f"Number of objectives of problem: {self.problem_instance.n_obj}")
            # There are more ways of defininf optimal point
            # 1. minimal point in current population
            minimal_point = np.min(pf, axis=0)
            # Vector of zeros
            minimal_point_zeros = np.zeros(self.problem.n_obj)
            # There are also more ways to define direction vector
            # average point in current population
            avg_point = np.mean(population_values, axis=0)
            # direction vector from average to minimal
            direction_vector_1 = minimal_point - avg_point - np.ones(self.problem.n_obj) * 1e-2
            direction_vector_2 = np.ones(self.problem.n_obj)/self.problem.n_obj
            #Defining sigma
            sigma = 1

            # Compute KKTPM for each point
            offset = 1e-2  # Small offset to avoid division by zero, etc.
            kktpm_values = []
            for sol in population:


                kktpm_val = apbi_kktpm(
                    sol,
                    lambda xx: self.phi_func(xx, direction_vector_2, minimal_point_zeros, sigma),
                    self.g_functions,
                    implementation="complicate_linear_system"
                )
                if kktpm_val is not None:
                    kktpm_values.append(kktpm_val)
            pbi_kktpm_values = []
            for sol in population:
                # Slight offset to minimal_point to avoid division by zero, etc.
                pbi_kktpm_val = pbi_kktpm(
                    sol,
                    lambda xx: self.phi_func(xx, direction_vector_2, minimal_point_zeros, sigma),
                    self.g_functions
                )
                if pbi_kktpm_val is not None:
                    pbi_kktpm_values.append(pbi_kktpm_val)
            #Compute gradient of d1 and d2
            # Compute Generational Distance
            gd_val = generational_distance(population_values, pf)
            self.GD_measures.append([iteration, gd_val])
            self.KKTPM_measures.append([iteration, kktpm_values])
            self.pbi_kktpm_measures.append([iteration, pbi_kktpm_values])
