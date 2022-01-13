from base.base_data_loader import BaseDataLoader
from keras.datasets import mnist
import os
import numpy as np
from data_loader.data_augmenter import DataAugmenter
from sklearn.model_selection import train_test_split

class SimpleLSTMDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(SimpleLSTMDataLoader, self).__init__(config)
        self.__load_data(config)

    def __load_data(self,config):
        X = np.genfromtxt(os.path.abspath("C:/Users/User/Documents\Project/Dataset/CSV_Outputs/labels.csv"), delimiter=',')
        Y = np.genfromtxt(os.path.abspath("C:/Users/User/Documents/Project/Dataset/CSV_Outputs/Y_concat.csv"), delimiter=',')
        Y = Y.T
        seq_length = config.data.seq_length
        cutoff = np.shape(X)[0] % seq_length
        X = X[:-cutoff]
        Y = Y[:-cutoff]
        m = int(np.shape(X)[0] / seq_length)
        X = X.reshape(m,seq_length,np.shape(X)[1])
        Y = Y.reshape(m,seq_length,np.shape(Y)[1])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,Y,test_size=0.18,shuffle=False)

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test
