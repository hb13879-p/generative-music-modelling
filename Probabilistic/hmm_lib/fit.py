from seqlearn.hmm import MultinomialHMM
from keras.utils.np_utils import to_categorical
import numpy as np
from seqlearn._decode import viterbi
from collections import Counter


obs = ["Rainy", "Rainy", "Sunny", "Rainy", "Cloudy", "Rainy"]
obs = [[0, 0], [1, 0], [2, 0, 1]]
hidden_states_strings = ["walk", "shop", "clean"]
hidden_state_values = [0, 1, 2]
hidden_states = [[1, 2], [1, 2], [1, 2, 1]]
n_hidden_states = 3
n_observation_types = 3


def init_pi(hidden_states, hidden_state_values):
    pi = np.zeros((1, len(hidden_state_values)))
    first_state = [elem[0] for elem in hidden_states]
    total = 0
    for i in hidden_state_values:
        x = first_state.count(i)
        pi[0, i] = x
        total += x
    pi = pi / total
    return pi


pi = init_pi(hidden_states, hidden_state_values)
# print(pi)

# alpha is ixj where alpha(i,j) is prob of transition from state i to state j
# each row should add to 1
def init_alpha(samples, hidden_state_values):
    n_hidden_states = len(hidden_state_values)
    alpha = np.zeros((n_hidden_states, n_hidden_states))
    transition_count = {}
    firsts_count = {}
    for sample in samples:
        pairs = list(zip(sample, sample[1:]))
        for key in pairs:
            alpha[key] += 1
    row_sums = alpha.sum(axis=1).reshape(n_hidden_states, 1)
    return np.divide(alpha, row_sums, where=row_sums != 0)


alpha = init_alpha(hidden_states, hidden_state_values)
# print(alpha)


def init_beta(obs, hidden_states, n_hidden_states, n_observation_types):
    beta = np.zeros((n_hidden_states, n_observation_types))
    hidden_state_obs_pairs = []
    for i, x in enumerate(hidden_states):
        hidden_state_obs_pairs.append(list(zip(x, obs[i])))
    hidden_state_obs_pairs = [
        item for sublist in hidden_state_obs_pairs for item in sublist
    ]
    for key in hidden_state_obs_pairs:
        beta[key] += 1
    row_sums = beta.sum(axis=1).reshape(n_hidden_states, 1)
    beta = np.divide(beta, row_sums, where=row_sums != 0)
    return beta


beta = init_beta(obs, hidden_states, n_hidden_states, n_observation_types)
print(beta)
