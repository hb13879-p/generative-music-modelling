from base.base_data_loader import BaseDataLoader
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
import pickle


class HmmMelodyDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(HmmMelodyDataLoader, self).__init__(config)
        self.obs_seq = [
            [[1, 2, 0, 0], [0, 1, 2, 2], [1, 0, 1, 2]],
            [[0, 1, 1, 0], [0, 1, 1, 1], [1, 1, 0, 0]],
        ]
        self.bn_seq = [
            np.array([1, 2, 0, 2]),
            np.array([1, 0, 0, 2]),
            np.array([2, 0, 1, 1]),
        ]
        self.tn_seq = [
            np.array([0, 1, 1, 0]),
            np.array([1, 0, 1, 0]),
            np.array([1, 1, 0, 1]),
        ]
        self.mn_seq = [
            np.array([0, 1, 1, 1]),
            np.array([0, 1, 1, 0]),
            np.array([1, 1, 0, 1]),
        ]
        # list of np arrays (each of lenght of seq) for each of 5 state sequences
        self.state_seq = [self.bn_seq, self.mn_seq, self.tn_seq]
        self.obs1_state_space = [0, 1, 2]
        self.obs2_state_space = [0, 1]
        self.bn_state_space = [0, 1, 2]
        self.mn_state_space = [0, 1]
        self.tn_state_space = [0, 1]

    def get_obs_seq(self):
        return self.obs_seq

    def get_state_seq(self):
        return self.state_seq
