"""
Module containing implementation of qlearning algorithm
"""
import numpy as np

from evolution_algorithm import EvolutionAlgorithm, one_point_crossing, average_crossing, uniform_crossing


crossing_functions = [one_point_crossing, average_crossing, uniform_crossing]
crossing_probabilities = [(x*10 + 5)/100 for x in range(10)]

# mean distance and success rate will be discreticised using np.digitize
distance_bins = [10**x for x in range(6)]
success_bins = [(x*10 + 5)/100 for x in range(10)]


class QLearningEvolution:
    def __init__(self):
        pass


def QLearningEvolution(evolution_algorithm, learning_rate, discount, epsilon, max_iter, success_rate):
    # Q ← zainicjalizuj

    # o ← ocena( q, P0 ) # ewolucyjny

    # xbest, obest ← znajdź najlepszego( P0, o ) # ewolucyjny

    # śr_d0 ← oblicz średnią odległość ( P0 )
    mean_distance = evolution_algorithm._calculate_mean_distance()

    for _ in range(max_iter):
        # st ← pst, śr_dt
        state = (evolution_algorithm._calculate_mean_distance,
                 evolution_algorithm._calculate_success_rate)

        # pc, k ← wybierz akcję ( st, Q,  )
        # wybierz funkcję krzyżowania
        # wybierz prawdopodobieństwo krzyżowania

        # R ← reprodukcja( Pt, o, μ ) # ewolucyjny start
        evolution_algorithm._reproduce()

        # C ← krzyżowanie( R, pc  )
        # M ← mutacja( C, σ  )
        evolution_algorithm._genetic_operations(
            crossing_func, crossing_probability)

        # om ← ocena( q, M )

        # xbest , obest ← znajdź najlepszego( M, om,  xbest, obest )

        # Pt+1, o ← sukcesja( Pt, M, o, om ) # ewolucyjny stop
        evolution_algorithm._succession()

        # śr_dt+1 ← oblicz średnią odległość ( Pt+1 )
        # pst+1 ← oblicz procent sukcesów ( Pt+1, Pt )
        # st+1 ← pst+1, śr_dt+1
        state_next = (evolution_algorithm._calculate_mean_distance,
                      evolution_algorithm._calculate_success_rate)

        # rt ← wyznacz nagrodę ( st+1 )

        # Q ( st, a ) ← Q ( st, a ) +  ( rt +  maxaQ ( st+1, a ) - Q ( st, a ) )
