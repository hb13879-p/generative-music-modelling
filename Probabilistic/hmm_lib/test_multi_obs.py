import unittest
import numpy as np
import numpy.testing as npt
from multinomial_hmm_mult_obs import MultinomialHMM


class MultiObsTestCase(unittest.TestCase):

    def test_fit(self):
        obs_spaces = [[0,1,2],[0,1,2,3]]
        states = [0,1]
        model = MultinomialHMM(states,obs_spaces,n_obs_seq=2)
        obs_seq = [[[0,0],[1,0],[2,0,1]],[[0,1],[3,2],[3,2,1]]]
        hidden_state_seq = [[0,1],[0,0],[1,1,0]]
        model.fit(hidden_state_seq,obs_seq)
        npt.assert_almost_equal(model.beta[0],[[0.5,0.5,0],[0.67,0,0.33]],decimal=2)
        npt.assert_almost_equal(model.beta[1],[[0.25,0.25,0.25,0.25],[0,0.33,0.33,0.33]],decimal=2)
        y = [[0,1,2],[0,3,2,1]]

    def test_viterbi_single_obs_init_params(self):
        obs = [[0,1,2]]
        states = [0,1]
        pi = np.array([[0.6],[0.4]])
        alpha = np.array([[0.7,0.3],[0.4,0.6]])
        beta = np.array([[[0.5,0.4,0.1],[0.1,0.3,0.6]]])
        model = MultinomialHMM(states,obs)
        model.initialise_params(pi,alpha,beta)
        y = [[0,1,2]]
        path,prob = model.viterbi(y)
        self.assertEqual(prob,0.01512)
        self.assertListEqual(list(path),[0.,0.,1.])

    def test_viterbi_dual_obs_init_params(self):
        obs_space = [[0,1,2],[0,1]]
        states = [0,1,2,3,4]
        pi = np.array([[0.1],[0.1],[0.3],[0.3],[0.2]])
        alpha = np.array([[0.1,0.1,0.1,0.1,0.5],[0.1,0,0.1,0.4,0.4],[0.6,0.1,0.1,0.1,0.1],[0.2,0.2,0.2,0.2,0.2],[0.3,0.1,0.1,0.2,0.3]])
        beta1 = np.array([[0.3,0.3,0.4],[0.2,0.3,0.5],[0.3,0.2,0.5],[0.1,0.5,0.4],[0.5,0.3,0.2]])
        beta2 = np.array([[0.5,0.5],[0.3,0.7],[0.7,0.3],[0.4,0.6],[0.9,0.1]])
        beta = [beta1,beta2]
        #print(beta)
        model = MultinomialHMM(states,obs_space)
        model.initialise_params(pi,alpha,beta)
        y = [[0,2,1,2,2,1,0],[0,1,1,1,0,0,0]]
        path,prob = model.viterbi(y)
        #print(path)
        #print(prob)

    def test_fwd_bckwd_single_obs(self):
        obs_spaces = [[0,1,2]]
        states = [0,1]
        model = MultinomialHMM(states,obs_spaces,n_obs_seq=1)
        obs_seq = [[[0,0],[1,0],[2,0,1]]]
        hidden_state_seq = [[0,1],[0,0],[0,1,0]]
        model.fit(hidden_state_seq,obs_seq)
        print(model.alpha)
        print(model.beta)
        npt.assert_almost_equal(model.beta[0],[[0.4,0.4,0.2],[0.98,0.01,0.01]],decimal=2)
        y = [0,1,2,0,1,1,0,1]
        #print(model.calculate_sequence_likelihood_eq1(y))

if __name__ == "__main__":
    unittest.main()
