import numpy as np
from multinomial_hmm import MultinomialHMM

def main():
    '''
    obs = ["Rainy","Rainy","Sunny","Rainy","Cloudy","Rainy"]
    obs = [[0,0],[1,0],[2,0,1]]
    hidden_states_strings = ["walk","shop","clean"]
    state_space = [0,1,2]
    obs_space = [0,1,2]
    hidden_states = [[1,2],[1,2],[1,2,1]]
    n_hidden_states = 3
    n_observation_types = 3
    model = MultinomialHMM(state_space,obs_space)
    model.fit(hidden_states,obs)
    y = [1,2]
    path,prob = model.viterbi(y)
'''
    obs = [0,1,2]
    states = [0,1]
    pi = np.array([[0.6],[0.4]])
    alpha = np.array([[0.7,0.3],[0.4,0.6]])
    beta = np.array([[0.5,0.4,0.1],[0.1,0.3,0.6]])
    test_viterbi = MultinomialHMM(states,obs)
    test_viterbi.initialise_params(pi,alpha,beta)
    y = [0,1,2]
    path,prob = test_viterbi.viterbi(y)
    print(path)
    print(prob)




if __name__ == '__main__':
    main()
