from base.base_data_loader import BaseDataLoader
from keras.datasets import mnist
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

class LSTMMelodyDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(LSTMMelodyDataLoader, self).__init__(config)
        self.__load_data(config.data.seq_length)

    def __load_data(self,seq_length):
        X = np.genfromtxt(os.path.abspath("C:/Users/User/Documents\Project/Dataset/CSV_Outputs/labels.csv"), delimiter=',')
        Y = np.genfromtxt(os.path.abspath("C:/Users/User/Documents/Project/Dataset/CSV_Outputs/Y_concat.csv"), delimiter=',')
        melody = pickle.load( open(os.path.abspath(r"C:\Users\User\Documents\Project\Dataset/melody.p"), "rb" ) )
        chord_rhythm = list(map(int,pickle.load( open(os.path.abspath(r"C:\Users\User\Documents\Project\Dataset/chord_rhythm.p"), "rb" ) )))
        self.__encode_melody(melody,chord_rhythm)
        X = np.concatenate((X,self.encoded_melody),axis=1)
        Y = Y.T
        cutoff = np.shape(X)[0] % seq_length
        X = X[:-cutoff]
        Y = Y[:-cutoff]
        m = int(np.shape(X)[0] / seq_length)
        X = X.reshape(m,seq_length,np.shape(X)[1])
        Y = Y.reshape(m,seq_length,np.shape(Y)[1])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,Y,test_size=0.18,shuffle=False)

    def __encode_melody(self,melody,chord_rhythm):
        i = 0
        encoded_melody = []
        for chord in chord_rhythm:
            mel = melody[i:chord]
            encoded_melody.append(self.__create_12d_vector_and_extract_melody(mel))
            i += len(mel)
        self.encoded_melody = encoded_melody

    def __create_12d_vector_and_extract_melody(self,melody):
        if len(melody) == 0:
            return np.zeros(12)
        res = np.zeros(12)
        melody = [(x + 3) % 12 for x in melody]
        for note in melody:
            res[note] += 1
        res /= len(melody)
        assert np.sum(res) == 1
        return res

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test
