import numpy as np

T = 8  # seq_len
S = 3  # no states


def marg(psi, Psi):
    alpha = np.ones((T, S))
    beta = np.ones((T, S))
    for t in range(1, T):
        alpha[t, :] = np.dot(Psi.T, (alpha[t - 1, :] * psi[t - 1, :]))
    for t in reversed(range(T - 1)):
        beta[t, :] = np.dot(Psi, (beta[t + 1, :] * psi[t + 1, :]))
    Ptilde = alpha * beta * psi
    # sum Ptilde over rows
    partition = np.sum(Ptilde, axis=1, keepdims=True)
    PY2 = np.divide(Ptilde, partition, out=np.zeros_like(Ptilde), where=partition != 0)
    PY = Ptilde / partition
    print(PY)
    print(PY2)
    return alpha, beta, PY


Psi_a = np.random.rand(S, S)  # transition matrix
psi_a = np.zeros((T, S))  # emission matrix
psi_a[4] = [0.3, 0.3, 0.4]
print(psi_a)
alpha_a, beta_a, PA = marg(psi_a, Psi_a)
"""
Psi_b = np.random.rand(S,S) #transition matrix
psi_b = np.random.rand(T,S) #emission matrix
alpha_b,beta_b,PB = marg(psi_b,Psi_b)


def probAgivenC(PA,PB,psi_a,psi_b):
     sumb = np.sum(PB * psi_b,axis=1,keepdims=True)
     PAC = PA * psi_a * sumb
     row_sums = PAC.sum(axis=1)
     print(row_sums)
     norm_PAC = PAC/row_sums[:,np.newaxis]
     return norm_PAC



PA = np.array([[0.2,0.2,0.6],
            [0.2,0.2,0.6],
            [0.2,0.2,0.6],
            [0.2,0.2,0.6],
            [0.2,0.2,0.6],
            [0.2,0.2,0.6],
            [0.1,0.4,0.5],
            [0.2,0.2,0.6]])
PB = np.array([[0.5,0.5],
            [0.5,0.5],
            [0.5,0.5],
            [0.5,0.5],
            [0.5,0.5],
            [0.5,0.5],
            [0.5,0.5],
            [0.5,0.5]])
psi_a = np.array([[0.036,0.03,0.02],
                [0.036,0.03,0.02],
                [0.036,0.03,0.02],
                [0.036,0.03,0.02],
                [0.036,0.03,0.02],
                [0.036,0.03,0.02],
                [0.036,0.03,0.02],
                [0.036,0.03,0.02]])
psi_b = np.array([[0.00390625,0.0024],
                [0.00390625,0.0024],
                [0.00390625,0.0024],
                [0.00390625,0.0024],
                [0.00390625,0.0024],
                [0.00390625,0.0024],
                [0.00390625,0.0024],
                [0.00390625,0.0024]])
print(probAgivenC(PA,PB,psi_a,psi_b))
"""
