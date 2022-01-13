from base.base_data_loader import BaseDataLoader
import os
import pickle


class HmmMelodyDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(HmmMelodyDataLoader, self).__init__(config)
        self.obs_seq = pickle.load(
            open(
                os.path.abspath(
                    r"C:\Users\User\Documents\Project\Models/Probabilistic/data_loaders/stored_data/simple_obs_seq.p"
                ),
                "rb",
            )
        )
        self.state_seq = pickle.load(
            open(
                os.path.abspath(
                    r"C:\Users\User\Documents\Project\Models/Probabilistic/data_loaders/stored_data/simple_state_seq.p"
                ),
                "rb",
            )
        )
        self.test_set_size = config.data.test_set_size
        self.train_test_split()
        self.calculate_state_spaces()

    def train_test_split(self):
        self.obs_train_seq = []
        self.obs_test_seq = []
        for i, seq in enumerate(self.obs_seq):
            print(i)
            self.obs_train_seq.append(seq[0 : -self.test_set_size])
            self.obs_test_seq.append(seq[-self.test_set_size :])
        self.state_train_seq = []
        self.state_test_seq = []
        for seq in self.state_seq:
            self.state_train_seq.append(seq[0 : -self.test_set_size])
            self.state_test_seq.append(seq[-self.test_set_size :])

    def calculate_state_spaces(self):
        self.state_spaces = []
        for seq in self.state_seq:
            state_space = list(set(x for l in seq for x in l))
            print(state_space)
            self.state_spaces.append(state_space)

    def get_state_spaces(self):
        return self.state_spaces

    def get_obs_train_seq(self):
        return self.obs_train_seq

    def get_obs_test_seq(self):
        return self.obs_test_seq

    def get_state_train_seq(self):
        return self.state_train_seq

    def get_state_test_seq(self):
        return self.state_test_seq
