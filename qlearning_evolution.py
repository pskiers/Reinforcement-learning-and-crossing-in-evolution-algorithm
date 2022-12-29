"""
Module containing implementation of qlearning algorithm
"""
import copy

import numpy as np

from evolution_algorithm import EvolutionAlgorithm


class QLearningEvolution:
    def __init__(self, evolution_algorithm: EvolutionAlgorithm, mean_distance_bins: list, success_rate_bins: list, crossing_functions: list, crossing_probabilities: list, epsilon: float, learning_rate: float):
        # evolution algorithm to optimize
        # will be deepcopied before every run
        self.evolution_algorithm = evolution_algorithm
        
        # bins for state discretization
        self.mean_distance_bins = mean_distance_bins
        self.success_rate_bins = success_rate_bins

        # actions lists
        self.crossing_functions = crossing_functions
        self.crossing_probabilities = crossing_probabilities

        # state and actions shapes
        self.state_shape = [len(self.mean_distance_bins) + 1, len(self.success_rate_bins) + 1]
        self.action_shape = [len(crossing_functions), len(crossing_probabilities)]

        # q-learning parameters and qtable
        self.q_table = np.zeros([*self.state_shape, *self.action_shape])
        self.epsilon = epsilon
        self.learning_rate = learning_rate

    def discretize_state(self, mean_distance: float, success_rate: float) -> tuple:
        '''
        Discreticizes given state (mean_distance and success rate of evolution algorithm) using np.digitize and bins declared in init.

        :param mean_distance: mean euclidean distance between specimens of current population
        :param success_rate: mean percent of specimens from previous population worse than specimen from current population

        :return: bin indexes for both values respectively, expressed as a tuple of two integers
        '''
        return np.digitize(mean_distance, self.mean_distance_bins), np.digitize(success_rate, self.success_rate_bins)

    def choose_action(self, mean_distance: float, success_rate: float) -> tuple:
        '''
        Choose next action based on qtable and current mean distance and success rate using epsilon-greedy policy

        :param mean_distance: mean euclidean distance between specimens of current population
        :param success_rate: mean percent of specimens from previous population worse than specimen from current population

        :return: next action
        '''
        dist_bin, success_bin = self.discretize_state(mean_distance, success_rate)

        if np.random.rand() < self.epsilon:
            # choose random action
            return tuple([np.random.randint(0, max_val) for max_val in self.action_shape])
        else:
            # choose action with the highest q_table value
            return np.argmax(self.q_table[dist_bin, success_bin, :, :])
    
    def reset_qtable(self):
        '''
        Sets Qtable to all zeros
        '''
        self.q_table = np.zeros([*self.state_shape, *self.action_shape])
