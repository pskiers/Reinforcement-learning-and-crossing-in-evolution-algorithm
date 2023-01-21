import numpy as np
import pandas as pd

from cec2017.functions import f21, f22, f23, f24
from qlearning_evolution import QLearningEvolution, Record
from evolution_algorithm import EvolutionAlgorithm
from evolution_algorithm import one_point_crossing, average_crossing, uniform_crossing


# hyperparams to tweak
POPULATION_SIZES = [10, 100, 1000]
MUTATION_STRENGTHS = [0, 0.1, 0.3, 0.5]
DIMENSIONALITIES = [2, 10, 20]
VALUE_FUNCTIONS = {'f21': f21, 'f22': f22, 'f23': f23, 'f24': f24}

# constant hyperparams
EPOCHS = 0
MEAN_DISTANCE_BINS = [10**n for n in [2, 3, 4, 5, 6]]
SUCCESS_RATE_BINS = np.arange(0.0, 1.0, 0.1)
CROSSING_FUNCTIONS = {'one_point_crossing': one_point_crossing,
                      'average_crossing': average_crossing,
                      'uniform_crossing': uniform_crossing}
CROSSING_PROBABILITIES = np.arange(0.1, 1.0, 0.1)
EPSILON = 0.2
LEARNING_RATE = 0.9
DISCOUNT_FACTOR = 0.5

# ???
REWARD_FUNCTION = lambda x,y: x

# files
OUTPUT_FILE = 'results.csv'


if __name__ == '__main__':
    df = pd.DataFrame(columns = [
                      'population_size',
                      'mutation_strength',
                      'dimensionalities',
                      'value_function',
                      'epochs',
                      'mean_distances',
                      'success_rate_bins',
                      'crossing_functions',
                      'crossing_probabilities',
                      'epsilon',
                      'learning_rate',
                      'discount_factor'])

    for pop_size in POPULATION_SIZES:
        for mut_strength in MUTATION_STRENGTHS:
            for dim in DIMENSIONALITIES:
                for func_name in VALUE_FUNCTIONS.keys():
                    ea = EvolutionAlgorithm(value_function=VALUE_FUNCTIONS[func_name],
                                            population_size=pop_size,
                                            mutation_strength=mut_strength,
                                            upper_bound=100,
                                            dimensionality=dim)
                    qle = QLearningEvolution(evolution_algorithm=ea,
                                             mean_distance_bins=MEAN_DISTANCE_BINS,
                                             success_rate_bins=SUCCESS_RATE_BINS,
                                             crossing_functions=CROSSING_FUNCTIONS.values(),
                                             crossing_probabilities=CROSSING_PROBABILITIES,
                                             epsilon=EPSILON,
                                             learning_rate=LEARNING_RATE,
                                             discount_factor=DISCOUNT_FACTOR,
                                             reward_function=REWARD_FUNCTION)
                    # run a copy of ea 25 times
                    # collect results

                    # run QLE 25 times
                    # collect records
                    # save results to dataframe
                    pass

    df.to_csv(OUTPUT_FILE)
