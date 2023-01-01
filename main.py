"""
Main programme for experiments
"""

import numpy as np
from evolution_algorithm import EvolutionAlgorithm, one_point_crossing, average_crossing, uniform_crossing
from qlearning_evolution import QLearningEvolution
from cec2017.functions import f1


if __name__ == "__main__":
    evolution = EvolutionAlgorithm(
        value_function=lambda x: np.apply_along_axis(func1d=f1, axis=1, arr=x),
        population_size=128,
        mutation_strength=0.3,
        upper_bound=100,
        dimensionality=2
    )
    dist_bins = [1000., 10000., 100000., 500000., 1000000.]
    succ_bins = [0.2, 0.4, 0.6, 0.8]
    cross_prob = [0, 0.3, 0.5, 0.8, 0.9, 0.95]
    cross_funcs = [one_point_crossing, average_crossing, uniform_crossing]
    qevolution =QLearningEvolution(
        evolution_algorithm=evolution,
        mean_distance_bins=dist_bins,
        success_rate_bins=succ_bins,
        crossing_functions=cross_funcs,
        crossing_probabilities=cross_prob,
        epsilon=0.4,
        learning_rate=0.9,
        discount_factor=0.5,
        reward_function=lambda x,y: x+y
    )
    qevolution.run(100)
