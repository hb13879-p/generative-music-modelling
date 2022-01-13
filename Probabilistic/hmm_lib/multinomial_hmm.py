import numpy as np


class MultinomialHMM(object):
    def __init__(self, state_space, obs_space):
        self.state_space = state_space
        self.obs_space = obs_space
        self.n_states = len(state_space)
        self.n_obs_types = len(obs_space)

    def initialise_params(self, pi, alpha, beta):
        self.pi = pi
        self.alpha = alpha
        self.beta = beta

    def fit(self, hidden_state_samples, obs_seq):
        self.__init_pi(hidden_state_samples, self.state_space)
        self.__init_alpha(hidden_state_samples, self.state_space)
        self.__init_beta(obs_seq, hidden_state_samples)

    def __init_pi(self, hidden_states, hidden_state_values):
        pi = np.zeros((1, self.n_states))
        first_state = [elem[0] for elem in hidden_states]
        total = 0
        for i in hidden_state_values:
            x = first_state.count(i)
            pi[0, i] = x
            total += x
        pi = pi / total
        self.pi = pi.T

    def __init_alpha(self, hidden_states, hidden_state_values):
        n_hidden_states = len(hidden_state_values)
        alpha = np.zeros((n_hidden_states, n_hidden_states))
        transition_count = {}
        firsts_count = {}
        for sample in hidden_states:
            pairs = list(zip(sample, sample[1:]))
            for key in pairs:
                alpha[key] += 1
        row_sums = alpha.sum(axis=1).reshape(n_hidden_states, 1)
        self.alpha = np.divide(alpha, row_sums, where=row_sums != 0)

    def __init_beta(self, obs, hidden_states):
        beta = np.zeros((self.n_states, self.n_obs_types))
        hidden_state_obs_pairs = []
        for i, x in enumerate(hidden_states):
            hidden_state_obs_pairs.append(list(zip(x, obs[i])))
        hidden_state_obs_pairs = [
            item for sublist in hidden_state_obs_pairs for item in sublist
        ]
        for key in hidden_state_obs_pairs:
            beta[key] += 1
        row_sums = beta.sum(axis=1).reshape(self.n_states, 1)
        self.beta = np.divide(beta, row_sums, where=row_sums != 0)

    def viterbi(self, y):
        # T1: each element i,j stores probability of most likely path so far X = (x1...xj) with xj = si that generates observations Y = (y1...yT)
        # T2: each element i,j stores xj-1 of most likely state so far
        T1 = np.zeros((self.n_states, len(y)))
        T2 = np.zeros((self.n_states, len(y)))
        X = np.negative(np.ones((np.shape(y))))
        Z = np.negative(np.ones((np.shape(y))))
        for i, state in enumerate(self.state_space):
            T1[i, 0] = self.pi[i] * self.beta[i, y[0]]
            T2[i, 0] = 0
        for i, obs in enumerate(y[1:], 1):
            for j, state in enumerate(self.state_space):
                max, argmax = self.__get_maxk(i, j, T1, y)
                T1[j, i] = max
                T2[j, i] = argmax
        # get most likely final state of most likely path so far (and its probability)
        max_prob, argmax = self.__get_final_state(T1)
        Z[-1] = argmax
        X[-1] = self.state_space[int(Z[-1])]
        for i, t in reversed(list(enumerate(y[1:], 1))):
            Z[i - 1] = T2[int(Z[i]), i]
            X[i - 1] = self.state_space[int(Z[i - 1])]
        self.most_likely_path = X
        self.most_likely_path_prob = max_prob
        return X, max_prob

    def __get_final_state(self, T1):
        max = 0
        argmax = -1
        for k, state in enumerate(self.state_space):
            x = T1[k, -1]
            if x > max:
                max = x
                argmax = k
        return max, argmax

    def __get_maxk(self, i, j, T1, y):
        max = 0
        argmax = -1
        for k, state in enumerate(self.state_space):
            x = T1[k, i - 1] * self.alpha[k, j] * self.beta[j, y[i]]
            if x > max:
                max = x
                argmax = k
        return max, argmax
