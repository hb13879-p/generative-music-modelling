import numpy as np

obs_seq = [[[0, 0], [1, 0], [2, 0, 1]], [[0, 1], [3, 2], [3, 2, 1]]]


def check(obs_seq):
    midlen = len(obs_seq[0])
    for i in obs_seq:
        assert len(i) == midlen, "Different numbers of observation sequences in obs_seq"

    lens = np.zeros((len(obs_seq), len(obs_seq[0])))
    for i, sublist in enumerate(obs_seq):
        for j, subsublist in enumerate(sublist):
            lens[i, j] = len(subsublist)
    find_differences = lens == lens[0, :]
    assert np.all(
        find_differences
    ), "Some observation sequences are different lengths across observation types"


check(obs_seq)
