import numpy as np
from multinomial_hmm_mult_obs import MultinomialHMM
import numpy.testing as npt

def main():
    test_fit()

def test_fit():
    obs_spaces = [[0,1,2],[0,1,2,3]]
    states = [0,1]
    model = MultinomialHMM(states,obs_spaces,n_obs_seq=2)
    obs_seq = [[[0,0],[1,0],[2,0,1]],[[0,1],[3,2],[3,2,1]]]
    hidden_state_seq = [[0,1],[0,0],[1,1,0]]
    model.fit(hidden_state_seq,obs_seq)
    npt.assert_almost_equal(model.beta[0],[[0.5,0.5,0],[0.67,0,0.33]],decimal=2)
    npt.assert_almost_equal(model.beta[1],[[0.25,0.25,0.25,0.25],[0,0.33,0.33,0.33]],decimal=2)
    y = [[0,1,2,0],[0,3,2,1]]
    #print(model.beta)
    model.approximate_joint_dist(y)


if __name__ == "__main__":
    main()
