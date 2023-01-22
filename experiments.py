import csv
from typing import Any
import numpy as np

from cec2017.functions import f1, f8, f15, f24
from qlearning_evolution import QLearningEvolution
from evolution_algorithm import EvolutionAlgorithm
from evolution_algorithm import one_point_crossing, average_crossing, uniform_crossing


# constant hyperparams
EPOCH_MAX = 1000
EPOCHS = [100*n-1 for n in range(1, 11)]
MEAN_DISTANCE_BINS = [10**n for n in range(2, 6)]
SUCCESS_RATE_BINS = [0.1*n for n in range(1, 10)]
CROSSING_FUNCTIONS = [one_point_crossing, average_crossing, uniform_crossing]
CROSSING_PROBABILITIES = [0.1, 0.3, 0.6, 0.8, 1]
EPSILON = 0.5
LEARNING_RATE = 0.7
DISCOUNT_FACTOR = 0.99

# hyperparams to tweak
POPULATION_SIZES = [10, 100, 1000]
MUTATION_STRENGTHS = [0, 0.5, 1, 3]

# experiment params
DIMENSIONALITIES = [2, 10, 20]
VALUE_FUNCTIONS = [f1, f8, f15, f24]
REWARD_FUNCTIONS = {
    "linear success rate": lambda x,y: x,
    "square success rate": lambda x,y: x^2,
    "const-linear succes rate": lambda x,y: x if x < 0.2 else 0.2,
    "specific succes rate": lambda x,y: x if x < 0.2 else 0.2 - (x - 0.2)
}

# files
OUTPUT_FILE = 'results.csv'
columns = [
    'algorithm',
    'value_function',
    'dimensionalities',
    'epochs',
    'population_size',
    'mutation_strength',
    'mean_distances',
    'success_rate_bins',
    'epsilon',
    'learning_rate',
    'discount_factor',
    'run_number',
    'best_score'
]
columns.extend([f'{f.__name__} uses' for f in CROSSING_FUNCTIONS])
columns.extend([f'{p} crossing probability uses' for p in CROSSING_PROBABILITIES])

if __name__ == '__main__':
    with open(OUTPUT_FILE, 'w') as f:
        writer = csv.DictWriter(f, columns)
        writer.writeheader()

    for pop_size in POPULATION_SIZES:
        for mut_strength in MUTATION_STRENGTHS:
            for dim in DIMENSIONALITIES:
                for func in VALUE_FUNCTIONS:
                    for i in range(25):
                        ea = EvolutionAlgorithm(value_function=lambda x: np.apply_along_axis(func1d=lambda x: func(x) * -1, axis=1, arr=x),
                                                population_size=pop_size,
                                                mutation_strength=mut_strength,
                                                upper_bound=100,
                                                dimensionality=dim)

                        for reward_func_name in REWARD_FUNCTIONS.keys():
                            qle = QLearningEvolution(evolution_algorithm=ea,
                                                     mean_distance_bins=MEAN_DISTANCE_BINS,
                                                     success_rate_bins=SUCCESS_RATE_BINS,
                                                     crossing_functions=CROSSING_FUNCTIONS,
                                                     crossing_probabilities=CROSSING_PROBABILITIES,
                                                     epsilon=EPSILON,
                                                     learning_rate=LEARNING_RATE,
                                                     discount_factor=DISCOUNT_FACTOR,
                                                     reward_function=REWARD_FUNCTIONS[reward_func_name])
                            records = qle.run(EPOCH_MAX)
                            result: dict[str, Any] = {}
                            result['algorithm'] = 'Q-Learning Evolution'
                            result['value_function'] = func.__name__
                            result['dimensionalities'] = dim
                            result['population_size'] = pop_size
                            result['mutation_strength'] = mut_strength
                            result['mean_distances'] = MEAN_DISTANCE_BINS
                            result['success_rate_bins'] = SUCCESS_RATE_BINS
                            result['epsilon'] = EPSILON
                            result['learning_rate'] = LEARNING_RATE
                            result['discount_factor'] = DISCOUNT_FACTOR
                            result['run_number'] = i
                            for epoch in EPOCHS:
                                result['epochs'] = epoch
                                result['best_score'] = records[epoch].best_evaluation
                                for f in CROSSING_FUNCTIONS:
                                    result[f'{f.__name__} uses'] = 0
                                for p in CROSSING_PROBABILITIES:
                                    result[f'{p} crossing probability uses'] = 0
                                for record in records:
                                    result[f'{record.crossing_func} uses'] += 1
                                    result[f'{record.crossing_probability} crossing probability uses'] += 1
                                with open(OUTPUT_FILE, 'a') as f:
                                    writer = csv.DictWriter(f, columns)
                                    writer.writerow(result)
                        for cross_func in CROSSING_FUNCTIONS:
                            for cross_prob in CROSSING_PROBABILITIES:
                                history = ea.run(EPOCH_MAX, cross_func, cross_prob)
                                result: dict[str, Any] = {}
                                result['algorithm'] = 'Evolution algorithm'
                                result['value_function'] = func.__name__
                                result['dimensionalities'] = dim
                                result['population_size'] = pop_size
                                result['mutation_strength'] = mut_strength
                                result['mean_distances'] = None
                                result['success_rate_bins'] = None
                                result['epsilon'] = None
                                result['learning_rate'] = None
                                result['discount_factor'] = None
                                result['run_number'] = i
                                for epoch in EPOCHS:
                                    result['epochs'] = epoch
                                    result['best_score'] = history[epoch]['evaluation']
                                    for f in CROSSING_FUNCTIONS:
                                        result[f'{f.__name__} uses'] = None
                                    for p in CROSSING_PROBABILITIES:
                                        result[f'{p} crossing probability uses'] = None
                                    with open(OUTPUT_FILE, 'a') as f:
                                        writer = csv.DictWriter(f, columns)
                                        writer.writerow(result)