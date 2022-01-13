from base.base_data_loader import BaseDataLoader
import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split

class SimpleLSTMDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(SimpleLSTMDataLoader, self).__init__(config)
        self.__load_data(config)

    def __load_data(self,config):
        X = pickle.load( open( config.data.dataset_X_filename, "rb" ) )
        Y = pickle.load( open( config.data.dataset_Y_filename, "rb" ) )
        print(np.shape(X))
        print(X[0])
        print(Y[0])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,Y,test_size=0.05,shuffle=False)

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test
