import pandas as pd

from cec2017.functions import f21, f22, f23, f24
from qlearning_evolution import QLearningEvolution
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
SUCCESS_RATE_BINS = [n/10 for n in float(range(10))]
CROSSING_FUNCTIONS = {'one_point_crossing': one_point_crossing,
                      'average_crossing': average_crossing,
                      'uniform_crossing': uniform_crossing}
CROSSING_PROBABILITIES = [n/10 for n in float(range(1, 10))]
EPSILON = 0.2
LEARNING_RATE = 0.9
DISCOUNT_FACTOR = 0.5

# ???
REWARD_FUNCTION = None

# files
OUTPUT_FILE = 'results.csv'


if __name__ == '__main__':
    df = pd.DataFrame(['population_size',
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
                for func in VALUE_FUNCTIONS.keys():
                    # run QLE
                    # save results to dataframe
                    pass

    df.to_csv(OUTPUT_FILE)
