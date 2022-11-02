"""
Main programme for experiments
"""

import numpy as np
from evolution_algorithm import EvolutionAlgorithm, one_point_crossing
from cec2017.functions import f1


if __name__ == "__main__":
    evolution = EvolutionAlgorithm(
        value_function=lambda x: np.apply_along_axis(func1d=f1, axis=1, arr=x),
        population_size=128,
        mutation_strength=0.3,
        upper_bound=100,
        dimensionality=2
    )
    for epoch in range(10000):
        evolution.next_generation(one_point_crossing, 0.3)
        print(f"Best: {evolution.best}, evaluation {evolution.best_evaluation}")
