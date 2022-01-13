import unittest
import numpy as np
import numpy.testing as npt
from hmm_lib.multinomial_hmm_mult_obs import MultinomialHMM


class MultiObsTestCase(unittest.TestCase):
    def test_fwd_bckwd_single_obs(self):
        obs_spaces = [[0, 1, 2]]
        states = [0, 1]
        model = MultinomialHMM(states, obs_spaces, n_obs_seq=1)
        obs_seq = [[[0, 0], [1, 0], [2, 0, 1]]]
        hidden_state_seq = [[0, 1], [0, 0], [0, 1, 0]]
        model.fit(hidden_state_seq, obs_seq)
        npt.assert_almost_equal(model.beta[0], [[0.4, 0.4, 0.2], [1, 0, 0]], decimal=2)
        y = [0, 1, 2, 0, 1, 1, 0, 1]
        print(model.approximate_joint_dist(y))


if __name__ == "__main__":
    unittest.main()
