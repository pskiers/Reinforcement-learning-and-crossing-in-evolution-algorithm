"""
Module containing implementation of qlearning algorithm
"""
import copy
from typing import Callable
from dataclasses import dataclass
import numpy as np
from evolution_algorithm import EvolutionAlgorithm

@dataclass
class Record:
    """
    Class represents single history record
    """
    best_point: np.ndarray
    best_evaluation: float
    avg_distance: float
    distance_bin: int
    success_rate: float
    success_bin: int
    crossing_func: str
    crossing_probability: float
    reward: float



class QLearningEvolution:
    """
    Class implementing Q-learning evolution algorithm
    """
    def __init__(
            self,
            evolution_algorithm: EvolutionAlgorithm,
            mean_distance_bins: list[float],
            success_rate_bins: list[float],
            crossing_functions: list[Callable[[np.ndarray, float], None]],
            crossing_probabilities: list[float],
            epsilon: float,
            learning_rate: float,
            discount_factor: float,
            reward_function: Callable[[float, float], float]
        ) -> None:
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
        self.discount_factor = discount_factor
        self.reward_function = reward_function

    def discretize_state(self, mean_distance: float, success_rate: float) -> tuple[int, int]:
        """
        Discreticizes given state (mean_distance and success rate of evolution algorithm) using
        np.digitize and bins declared in init.

        :param mean_distance: mean euclidean distance between specimens of current population
        :param success_rate: mean percent of specimens from previous population worse than specimen
            from current population

        :return: bin indexes for both values respectively, expressed as a tuple of two integers
        """
        return (
            int(np.digitize(mean_distance, self.mean_distance_bins)),
            int(np.digitize(success_rate, self.success_rate_bins))
        )

    def choose_action(self, mean_distance: float, success_rate: float) -> tuple:
        """
        Choose next action based on qtable and current mean distance and success rate using
        epsilon-greedy policy

        :param mean_distance: mean euclidean distance between specimens of current population
        :param success_rate: mean percent of specimens from previous population worse than specimen
            from current population

        :return: next action
        """
        dist_bin, success_bin = self.discretize_state(
            mean_distance, success_rate
        )

        if np.random.rand() < self.epsilon:
            # choose random action
            return tuple([np.random.randint(0, max_val) for max_val in self.action_shape])
        else:
            # choose action with the highest q_table value
            relevant = self.q_table[dist_bin, success_bin, :, :]
            return np.unravel_index(np.argmax(relevant), relevant.shape)
    def reset_qtable(self) -> None:
        """
        Sets q-table to all zeros
        """
        self.q_table = np.zeros([*self.state_shape, *self.action_shape])

    def update_qtable(self, state: tuple, action: tuple, next_state: tuple, reward: float) -> None:
        """
        Update q-table after receiving reward

        :param state: tuple representing current EA state - population mean distance and
            evolution success rate
        :param action: tuple representing action that gave the reward - crossing function
            and crossing probability
        :param next_state: tuple representing EA state after action - population mean
            distance and evolution success rate
        :param reward: reward received
        """
        self.q_table[state[0], state[1], action[0], action[1]] = (1 - self.learning_rate) * \
            self.q_table[state[0], state[1], action[0], action[1]] + self.learning_rate * \
            (reward + self.discount_factor * np.max(self.q_table[next_state, :]))

    def run(self, iterations: int, verbose: bool = True) -> list[Record]:
        """
        Runs the algorithm

        :param iterations: number of iterations to run
        :param verbose: if set to false the method runs silently

        :return: history of the run
        """
        self.reset_qtable()
        evolution = copy.deepcopy(self.evolution_algorithm)

        history = []
        # get starting state
        state = self.discretize_state(self.evolution_algorithm.calculate_mean_distance(), 1)

        for _ in range(iterations):
            # choose action
            action = self.choose_action(*state)
            cross_func = self.crossing_functions[action[0]]
            cross_prob = self.crossing_probabilities[action[1]]

            # perform genetic operations
            # get new state
            mean_dist, succ_rate = evolution.next_generation(cross_func, cross_prob)

            state_next = self.discretize_state(mean_dist, succ_rate)

            # get reward
            reward = self.reward_function(succ_rate, mean_dist)

            # update qtable
            self.update_qtable(state, action, state_next, reward)

            # next iteration
            state = state_next

            new_record = Record(
                best_point=evolution.best,  # type: ignore
                best_evaluation=evolution.best_evaluation,
                avg_distance=mean_dist,
                distance_bin=state[0],
                success_rate=succ_rate,
                success_bin=state[1],
                crossing_func=cross_func.__name__,
                crossing_probability=cross_prob,
                reward=reward
            )
            history.append(new_record)
            if verbose is True:
                msg = f"Best: {evolution.best}, evaluation {evolution.best_evaluation}, "
                msg += f"avg distance {state[0]}, success rate {state[1]}, "
                msg += f"crossing {cross_func.__name__}, crossing probability {cross_prob}"
                print(msg)
        return history
