import numpy as np


class MultinomialHMM(object):
    def __init__(self, state_space, obs_space, n_obs_seq=1):
        # obs seq is n_obs_seq lists of observation sequence samples - could be 3d
        # eg obs_seq = [[[0,0],[1,0],[2,0,1]],[[0,1],[3,2],[3,2,1]]]
        # ie dim [n_obs_seq,n_sample_sequences,n_timesteps_in_each_seq]
        self.__check_init_dims(obs_space)
        self.n_obs_seq = n_obs_seq
        self.state_space = state_space
        # obs_space is list of n_obs_seq lists
        # eg [[0,1,2],[0,1,2,3]]
        self.obs_space = obs_space
        self.n_states = len(state_space)
        self.n_obs_types = []
        for i in range(n_obs_seq):
            self.n_obs_types.append(len(obs_space[i]))

    def __check_init_dims(self, obs_space):
        assert any(
            isinstance(el, list) for el in obs_space
        ), "Please wrap 1d list in another list"

    def __check_fit_dims(self, obs_seq):
        # check obs_seq 3dims
        assert any(
            isinstance(el, list) for el in obs_seq[0]
        ), "Please make sure obs_seq is a 3d list of shape (n_obs_seq,n_sequence_samples,sample_seq_len)"
        assert (
            len(obs_seq) == self.n_obs_seq
        ), "n_obs_seq doesn't match number of sequences in obs_seq"
        midlen = len(obs_seq[0])
        for i in obs_seq:
            assert (
                len(i) == midlen
            ), "Different numbers of observation sequences in obs_seq"
        lens = np.zeros((len(obs_seq), len(obs_seq[0])))
        for i, sublist in enumerate(obs_seq):
            for j, subsublist in enumerate(sublist):
                lens[i, j] = len(subsublist)
        find_differences = lens == lens[0, :]
        np.savetxt("find_difference.csv", lens, delimiter=",")
        assert np.all(
            find_differences
        ), "Some observation sequences are different lengths across observation types"

    def __check_initialise_params_dims(self, beta):
        # check beta 3dims
        assert (
            beta[0].ndim == 2
        ), "Please make sure beta is a list of 2d emission matrices shape (n_hidden_states,len(obs_space))"
        for i in range(self.n_obs_seq):
            assert len(beta[0][i]) == len(
                self.obs_space[i]
            ), "Please make sure beta is a 3d list of shape (n_obs_seq,n_hidden_states,len(obs_space))"

    def initialise_params(self, pi, alpha, beta):
        self.__check_initialise_params_dims(beta)
        self.pi = pi
        self.alpha = alpha
        # beta is list of emission matrices, per observation sequence
        self.beta = beta
        self.__remove_alpha_nans()

    def fit(self, hidden_state_samples, obs_seq):
        self.__check_fit_dims(obs_seq)
        self.__init_pi(hidden_state_samples, self.state_space)
        self.__init_alpha(hidden_state_samples, self.state_space)
        self.__init_beta(obs_seq, hidden_state_samples)
        self.__remove_alpha_nans()
        self.__smooth_alpha_beta()

    def __remove_alpha_nans(self):
        self.alpha = np.nan_to_num(self.alpha)

    # simple additive smoothing
    def __smooth_alpha_beta(self):
        e = 0.01
        # add e to each zero term
        self.alpha[self.alpha == 0] += e
        self.alpha = self.__normalise_array(self.alpha)
        for i in range(len(self.beta)):
            self.beta[i][self.beta[i] == 0] += e
            self.beta[i] = self.__normalise_array(self.beta[i])

    def __normalise_array(self, a):
        row_sums = a.sum(axis=1)
        return a / row_sums[:, np.newaxis]

    def __init_pi(self, hidden_states, hidden_state_values):
        pi = np.zeros((1, self.n_states))
        first_state = [elem[0] for elem in hidden_states[0:-1]]
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
        self.beta = []
        for i in range(self.n_obs_seq):
            obs_seq = obs[i]
            beta = np.zeros((self.n_states, self.n_obs_types[i]))
            hidden_state_obs_pairs = []
            for j, x in enumerate(hidden_states):
                hidden_state_obs_pairs.append(list(zip(x, obs_seq[j])))
            hidden_state_obs_pairs = [
                item for sublist in hidden_state_obs_pairs for item in sublist
            ]
            for key in hidden_state_obs_pairs:
                beta[key] += 1
            row_sums = beta.sum(axis=1).reshape(self.n_states, 1)
            self.beta.append(np.divide(beta, row_sums, where=row_sums != 0))

    def viterbi(self, y):
        # y should be list of n_obs_seq lists
        # T1: each element i,j stores probability of most likely path so far X = (x1...xj) with xj = si that generates observations Y = (y1...yT)
        # T2: each element i,j stores xj-1 of most likely state so far
        self.__check_vit_dims(y)
        T1 = np.zeros((self.n_states, len(y[0])))
        T2 = np.zeros((self.n_states, len(y[0])))
        X = np.negative(np.ones((np.shape(y[0]))))
        Z = np.negative(np.ones((np.shape(y[0]))))
        for i, state in enumerate(self.state_space):
            betaterm = self.__get_initial_betaterm(i, y)
            T1[i, 0] = self.pi[i] * betaterm
            T2[i, 0] = 0
        for i, obs in enumerate(y[0][1:], 1):
            for j, state in enumerate(self.state_space):
                max, argmax = self.__get_maxk(i, j, T1, y)
                T1[j, i] = max
                T2[j, i] = argmax
        # get most likely final state of most likely path so far (and its probability)
        max_prob, argmax = self.__get_final_state(T1)
        Z[-1] = argmax
        X[-1] = self.state_space[int(Z[-1])]
        for i, t in reversed(list(enumerate(y[0][1:], 1))):
            Z[i - 1] = T2[int(Z[i]), i]
            X[i - 1] = self.state_space[int(Z[i - 1])]
        self.most_likely_path = X
        self.most_likely_path_prob = max_prob
        return X, max_prob

    def __check_vit_dims(self, y):
        assert any(
            isinstance(el, list) for el in y
        ), "Please wrap 1d list in another list"

    def __get_initial_betaterm(self, i, y):
        tot = 1
        for j, seq in enumerate(y):
            if np.size(y[j]) == 0:
                tot *= 1
            else:
                tot *= self.beta[j][i, y[j][0]]
        return tot

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
            betaterm = self.__get_mid_betaterm(i, j, y)
            x = T1[k, i - 1] * self.alpha[k, j] * betaterm
            if x > max:
                max = x
                argmax = k
        return max, argmax

    def __get_mid_betaterm(self, i, j, y):
        tot = 1
        for k, seq in enumerate(y):
            tot *= self.beta[k][j, y[k][i]]
        return tot

    def approximate_joint_dist(self, y):
        self.__calculate_psi(y)
        self.__calculate_sequence_likelihood_eq1(len(y[0]))

    def __calculate_psi(self, y):  # only one psi per model
        self.psi = np.zeros((len(y[0]), len(self.state_space)))
        for i, state in enumerate(self.state_space):
            for t, timestep in enumerate(y[0]):
                ans = 1
                for j in range(self.n_obs_seq):
                    ans *= self.beta[j][i, y[j][t]]
                self.psi[t, i] = ans

    def __calculate_sequence_likelihood_eq1(self, T):
        Psi = self.alpha
        alpha = np.ones((T, self.n_states))
        beta = np.ones((T, self.n_states))
        for t in range(1, T):
            alpha[t, :] = np.dot(Psi.T, (alpha[t - 1, :] * self.psi[t - 1, :]))
        for t in reversed(range(T - 1)):
            beta[t, :] = np.dot(Psi, (beta[t + 1, :] * self.psi[t + 1, :]))
        Ptilde = alpha * beta * self.psi
        # sum Ptilde over rows
        partition = np.sum(Ptilde, axis=1, keepdims=True)
        self.PA = np.divide(
            Ptilde, partition, out=np.zeros_like(Ptilde), where=partition != 0
        )
