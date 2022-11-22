"""
Module containing implementation of standard evolution algorithm
"""

from typing import Callable
import numpy as np


class EvolutionAlgorithm:
    """
    Class implementing evolution algorithm.
    """

    def __init__(
        self,
        value_function: Callable[[np.ndarray], np.ndarray],
        population_size: int,
        mutation_strength: float,
        upper_bound: float,
        dimensionality: int = 2
    ) -> None:
        self.value_function: Callable = value_function
        self.population_size: int = population_size
        self.mutation_strength: float = mutation_strength
        self.dimensionality: int = dimensionality
        self.epoch: int = 0
        self.population: np.ndarray = np.random.uniform(
            low=-upper_bound, high=upper_bound, size=(population_size, dimensionality)
        )
        self.previous_population: np.ndarray = np.zeros(shape=(population_size, dimensionality))
        self.evaluations: np.ndarray = self.value_function(self.population)
        self.previous_evaluations: np.ndarray = np.zeros(shape=(population_size, dimensionality))
        self.best_evaluation: float = float("-inf")
        self.best: np.ndarray | None = None
        self._find_best()

    def next_generation(
        self, crossing_type: Callable[[np.ndarray, float], None], crossing_probability: float
    ) -> tuple[float, float]:
        """
        Generates next generation
        """
        self.previous_population = self.population
        self.previous_evaluations = self.evaluations
        self._reproduce()
        self._genetic_operations(crossing_type, crossing_probability)
        self.evaluations = self.value_function(self.population)
        self._find_best()
        self._succession()
        self.epoch += 1
        return self._calculate_success_rate(), self._calculate_mean_distance()

    def _find_best(self) -> None:
        """
        Method finds best specimen
        """
        best_index = self.evaluations.argmax()
        if self.best_evaluation < self.evaluations[best_index]:
            self.best = self.population[best_index]
            self.best_evaluation = self.evaluations[best_index]

    def _reproduce(self) -> None:
        """
        Selects next next population from current population. Chosen reproduction type:
        tournament reproduction
        """
        reproduced = np.zeros((self.population_size, self.dimensionality))
        for i in range(len(self.population)):
            pair = np.random.randint(self.population_size, size=2)
            if self.evaluations[pair[0]] < self.evaluations[pair[1]]:
                reproduced[i] = self.population[pair[1]]
            else:
                reproduced[i] = self.population[pair[0]]
        self.population = reproduced


    def _genetic_operations(
        self, crossing_func: Callable[[np.ndarray, float], None], crossing_probability: float
    ) -> None:
        """
        Performs genetic operations (in our case crossing) on current population

        :param crossing_func: function that will be used to perform crossing
        :param crossing_probability: probability of crossing
        """
        crossing_func(self.population, crossing_probability)
        self.population += (
            np.random.normal(size=(self.population_size, self.dimensionality))
            * self.mutation_strength
        )

    def _succession(self) -> None:
        """
        Selects population next generation. Chosen succession: succession with elite 1
        """
        worst_idx = self.evaluations.argmin()
        prev_best_idx = self.previous_evaluations.argmax()
        if self.evaluations[worst_idx] < self.previous_evaluations[prev_best_idx]:
            self.population[worst_idx] = self.previous_population[prev_best_idx]
            self.evaluations[worst_idx] = self.previous_evaluations[prev_best_idx]

    def _calculate_success_rate(self) -> float:
        """
        Calculates success rate by calculating mean percent of specimens from previous population
        worse than specimen from current population

        :return: success rate
        """
        successes = 0
        for evaluation in self.evaluations:
            successes += len(self.previous_evaluations < evaluation)
        return successes / (self.population_size * self.population_size)

    def _calculate_mean_distance(self) -> float:
        """
        Calculates mean euclidean distance between specimens of current population

        :return: mean distance
        """
        distance_sum = 0
        for i in range(self.population_size):
            distance_sum += np.sum(np.linalg.norm(self.population[i:] - self.population[i]))
        return distance_sum / self.population_size



def one_point_crossing(population: np.ndarray, probability: float) -> None:
    """
    Performs crossing by choosing randomly (uniform distribution) one crossing point p where
    p < dimensionality and from 2 parents produces 2 offsprings one with features 0 to p of
    parent 1 and p to dimensionality of parent 2 and one with the other features

    :param population: specimens to cross
    :param probability: probability of crossing
    """
    do_cross = np.random.uniform(size=len(population)//2)
    num_of_feat = population.shape[1]
    cross_points = np.random.randint(low=0, high=num_of_feat, size=len(population)//2)
    for i in range(0, len(population), 2):
        if do_cross[i//2] < probability:
            population[i][:cross_points[i//2]], population[i+1][:cross_points[i//2]] = (
                population[i+1][:cross_points[i//2]], population[i][:cross_points[i//2]]
            )


def average_crossing(population: np.ndarray, probability: float) -> None:
    """
    Performs crossing by averaging parents

    :param population: specimens to cross
    :param probability: probability of crossing
    """
    do_cross = np.random.uniform(size=len(population)//2)
    for i in range(0, len(population), 2):
        if do_cross[i//2] < probability:
            population[i] += population[i+1]
            population[i+1] += population[i]
            population[i] /= 2
            population[i+1] /= 2


def uniform_crossing(population: np.ndarray, probability: float) -> None:
    """
    Performs uniform crossing for each feature if randomly generated number is
    < than probability of crossing then features are swapped

    :param population: specimens to cross
    """
    do_cross = np.random.uniform(size=len(population)//2)
    num_of_feat = population.shape[1]
    for i in range(0, len(population), 2):
        if do_cross[i//2] < probability:
            for f in range(num_of_feat):
                if np.random.random() < 0.5:
                    population[i][f], population[i+1][f] = (
                        population[i+1][f], population[i][f]
                    )
