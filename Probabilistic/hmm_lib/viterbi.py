import numpy as np

y = [0, 1, 2, 2, 0, 1, 2, 2, 2, 1, 1, 2, 0, 0, 1, 2]  # observation sequence
obs_space = [0, 1, 2]
state_space = [0, 1]
pi = [0.4, 0.6]
alpha = np.array([[0.45, 0.55], [0.4, 0.6]])
beta = np.array([[0.4, 0.2, 0.4], [0.3, 0.4, 0.3]])


def viterbi(obs_space, state_space, pi, y, alpha, beta):
    # T1: each element i,j stores probability of most likely path so far X = (x1...xj) with xj = si that generates observations Y = (y1...yT)
    # T2: each element i,j stores xj-1 of most likely state so far
    T1 = np.zeros((len(state_space), len(y)))
    T2 = np.zeros((len(state_space), len(y)))
    X = np.negative(np.ones((np.shape(y))))
    Z = np.negative(np.ones((np.shape(y))))
    for i, state in enumerate(state_space):
        T1[i, 0] = pi[i] * beta[i, y[0]]
        T2[i, 0] = 0
    for i, obs in enumerate(y[1:], 1):
        for j, state in enumerate(state_space):
            max, argmax = get_maxk(i, j, T1, state_space, alpha, beta)
            T1[j, i] = max
            T2[j, i] = argmax
    # get most likely final state of most likely path so far (and its probability)
    max_prob, argmax = get_final_state(state_space, T1)
    Z[-1] = argmax
    print(Z[-1])
    X[-1] = state_space[int(Z[-1])]
    for i, t in reversed(list(enumerate(y[1:], 1))):
        Z[i - 1] = T2[int(Z[i]), i]
        X[i - 1] = state_space[int(Z[i - 1])]
    return X, max_prob


def get_final_state(state_space, T1):
    max = 0
    argmax = -1
    for k, state in enumerate(state_space):
        x = T1[k, -1]
        if x > max:
            max = x
            argmax = k
    return max, argmax


def get_maxk(i, j, T1, state_space, alpha, beta):
    max = 0
    argmax = -1
    for k, state in enumerate(state_space):
        x = T1[k, i - 1] * alpha[k, j] * beta[j, y[i]]
        if x > max:
            max = x
            argmax = k
    return max, argmax


def viterbi2(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    for st in states:
        V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}
    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})
        for st in states:
            max_tr_prob = max(
                V[t - 1][prev_st]["prob"] * trans_p[prev_st][st] for prev_st in states
            )
            for prev_st in states:
                if V[t - 1][prev_st]["prob"] * trans_p[prev_st][st] == max_tr_prob:
                    max_prob = max_tr_prob * emit_p[st][obs[t]]
                    V[t][st] = {"prob": max_prob, "prev": prev_st}
                    break
    for line in dptable(V):
        print(line)
    opt = []
    # The highest probability
    max_prob = max(value["prob"] for value in V[-1].values())
    previous = None
    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["prob"] == max_prob:
            opt.append(st)
            previous = st
            break
    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

    return opt, max_prob


def dptable(V):
    # Print a table of steps from dictionary
    yield " ".join(("%12d" % i) for i in range(len(V)))
    for state in V[0]:
        yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)


print(str(y))
X, T1 = viterbi(obs_space, state_space, pi, y, alpha, beta)
X2, max_prob = viterbi2(y, state_space, pi, alpha, beta)
print(X)
print(T1)
print(X2)
print(max_prob)
