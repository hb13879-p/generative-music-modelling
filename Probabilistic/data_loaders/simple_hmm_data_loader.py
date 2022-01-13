from base.base_data_loader import BaseDataLoader
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
import pickle


class SimpleHmmDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(SimpleHmmDataLoader, self).__init__(config)
        self.top_note = []
        self.bottom_note = []
        self.note_one = []
        self.note_two = []
        self.note_three = []
        self.__load_data()
        self.encode_qualities()
        self.populate_single_note_lists(self.note_generator())
        self.create_label_encoders_and_encode_data()
        self.create_lengths_array(config.data.seq_length)
        self.save_data()

    def save_data(self):
        pickle.dump(
            self.get_obs_seq(),
            open(
                os.path.abspath(
                    r"C:\Users\User\Documents\Project\Models/Probabilistic/data_loaders/stored_data/simple_obs_seq.p"
                ),
                "wb",
            ),
        )
        pickle.dump(
            self.get_state_seq(),
            open(
                os.path.abspath(
                    r"C:\Users\User\Documents\Project\Models/Probabilistic/data_loaders/stored_data/simple_state_seq.p"
                ),
                "wb",
            ),
        )
        pickle.dump(
            self.get_label_encoders(),
            open(
                os.path.abspath(
                    r"C:\Users\User\Documents\Project\Models/Probabilistic/data_loaders/stored_data/simple_label_encoders.p"
                ),
                "wb",
            ),
        )

    def __load_data(self):
        self.voicings = pickle.load(
            open(
                os.path.abspath(r"C:\Users\User\Documents\Project\Dataset/voicings.p"),
                "rb",
            )
        )
        self.roots = pickle.load(
            open(
                os.path.abspath(r"C:\Users\User\Documents\Project\Dataset/roots.p"),
                "rb",
            )
        )
        self.qualities = pickle.load(
            open(
                os.path.abspath(r"C:\Users\User\Documents\Project\Dataset/qualities.p"),
                "rb",
            )
        )

    def encode_qualities(self):
        self.encoded_quals = []
        qual_encode = {
            "diminished-seventh": 0,
            "diminished": 0,
            "dominant": 1,
            "dominantsharp9": 1,
            "dominant-ninth": 1,
            "dominantflat9": 1,
            "dominant-13th": 1,
            "dominant-ninthsus4": 1,
            "dominant-11th": 1,
            "dominantalt": 1,
            "dominant-ninthflat5": 1,
            "augmented7#9": 1,
            "dominant-13thflat9": 1,
            "dominantsus4": 1,
            "dominantflat5": 1,
            "half-diminished": 2,
            "major": 3,
            "major-seventh": 3,
            "major-sixth": 3,
            "major-seventhsharp5": 3,
            "major-sixth9": 3,
            "major-ninth": 3,
            "major-seventhsharp11": 3,
            "minor-seventh": 4,
            "minor-11th": 4,
            "minor": 4,
            "minor-ninth": 4,
            "minor-sixth": 5,
            "minor-sixth9": 5,
            "major-minor": 5,
            "augmented-seventh": 6,
            "augmented": 6,
            "dominantsharp5": 6,
            "dominantsharp11": 7,
            "dominant-13thsharp11": 7,
            "dominant-ninthsharp11": 7,
        }
        self.n_qualities = len(set(elem for elem in qual_encode.values()))
        for c in self.qualities:
            self.encoded_quals.append(qual_encode[c])

    def note_generator(self):
        for chord in self.voicings:
            # get bottom note
            yield chord[0]
            chord = chord[1:]
            # get top note
            yield chord[-1]
            chord = chord[0:-1]
            # check if there are other notes and return them bottom to top
            for i in range(3):
                if not chord:
                    yield 0
                else:
                    yield chord[0]
                    chord = chord[1:]

    def populate_single_note_lists(self, gen):
        for chord in self.voicings:
            self.bottom_note.append(next(gen))
            self.top_note.append(next(gen))
            self.note_one.append(next(gen))
            self.note_two.append(next(gen))
            self.note_three.append(next(gen))

    def create_label_encoders_and_encode_data(self):
        self.le_bottom = LabelEncoder()
        self.le_n1 = LabelEncoder()
        self.le_n2 = LabelEncoder()
        self.le_n3 = LabelEncoder()
        self.le_top = LabelEncoder()
        self.encoded_bottom_note = self.le_bottom.fit_transform(self.bottom_note)
        self.encoded_note_one = self.le_n1.fit_transform(self.note_one)
        self.encoded_note_two = self.le_n2.fit_transform(self.note_two)
        self.encoded_note_three = self.le_n3.fit_transform(self.note_three)
        self.encoded_top_note = self.le_top.fit_transform(self.top_note)
        self.bn_state_space = sorted(set(self.encoded_bottom_note))
        self.n1_state_space = sorted(set(self.encoded_note_one))
        self.n2_state_space = sorted(set(self.encoded_note_two))
        self.n3_state_space = sorted(set(self.encoded_note_three))
        self.tn_state_space = sorted(set(self.encoded_top_note))
        self.qual_state_space = range(self.n_qualities)
        self.root_state_space = range(12)

    def get_label_encoders(self):
        return [self.le_bottom, self.le_n1, self.le_n2, self.le_n3, self.le_top]

    def get_state_seq(self):
        bn_seq = self.__split_list_into_seq(self.encoded_bottom_note, self.lengths)
        n1_seq = self.__split_list_into_seq(self.encoded_note_one, self.lengths)
        n2_seq = self.__split_list_into_seq(self.encoded_note_two, self.lengths)
        n3_seq = self.__split_list_into_seq(self.encoded_note_three, self.lengths)
        tn_seq = self.__split_list_into_seq(self.encoded_top_note, self.lengths)
        return [bn_seq, n1_seq, n2_seq, n3_seq, tn_seq]

    def get_obs_seq(self):
        roots = self.roots
        quals = self.encoded_quals
        data = [roots, quals]
        obs_seq = []
        assert len(data) == self.config.data.n_obs_seq, "Incorrect number of obs seq"
        for i in data:
            obs_seq.append(self.__split_list_into_seq(i, self.lengths))
        return obs_seq

    def __split_list_into_seq(self, data, lengths):
        obs_seq = []
        i = 0
        for leng in lengths:
            obs_seq.append(data[i : i + leng])
            i += leng
        return obs_seq

    def create_lengths_array(self, seq_length):
        n_full_sequences = int(len(self.roots) / seq_length)
        len_partial_sequence = len(self.roots) % seq_length
        assert n_full_sequences * seq_length + len_partial_sequence == len(self.roots)
        self.lengths = [seq_length] * n_full_sequences
        self.lengths.append(len_partial_sequence)

    def get_lengths(self):
        return self.lengths
