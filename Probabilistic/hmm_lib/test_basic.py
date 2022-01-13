import unittest
import numpy as np
import numpy.testing as npt
from multinomial_hmm import MultinomialHMM


class BasicTestCase(unittest.TestCase):
    def test_fit(self):
        obs_strings = ["Rainy", "Rainy", "Sunny", "Rainy", "Cloudy", "Rainy"]
        obs = [[0, 0], [1, 0], [2, 0, 1]]
        hidden_states_strings = ["walk", "shop", "clean"]
        state_space = [0, 1, 2]
        obs_space = [0, 1, 2]
        hidden_states = [[1, 2], [2, 2], [1, 2, 1]]
        n_hidden_states = 3
        n_observation_types = 3
        model = MultinomialHMM(state_space, obs_space)
        model.fit(hidden_states, obs)
        npt.assert_almost_equal(model.pi, [[0], [0.67], [0.33]], decimal=2)
        npt.assert_almost_equal(
            model.alpha, [[0, 0, 0], [0, 0, 1], [0, 0.5, 0.5]], decimal=2
        )
        npt.assert_almost_equal(
            model.beta, [[0, 0, 0], [0.33, 0.33, 0.33], [0.75, 0.25, 0]], decimal=2
        )
        beta = model.beta
        beta = beta.T
        print(np.shape(beta))
        T = 8
        psi = np.tile(np.prod(beta, axis=0, keepdims=True), (T, 1))
        print(psi)

    def test_viterbi(self):
        obs = [0, 1, 2]
        states = [0, 1]
        pi = np.array([[0.6], [0.4]])
        alpha = np.array([[0.7, 0.3], [0.4, 0.6]])
        beta = np.array([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]])
        test_viterbi = MultinomialHMM(states, obs)
        test_viterbi.initialise_params(pi, alpha, beta)
        y = [0, 1, 2]
        path, prob = test_viterbi.viterbi(y)
        self.assertEqual(prob, 0.01512)
        self.assertListEqual(list(path), [0.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
