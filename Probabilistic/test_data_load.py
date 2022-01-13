import pickle
import numpy as np
import os

X = np.genfromtxt(
    os.path.abspath("C:/Users/User/Documents\Project/Dataset/CSV_Outputs/labels.csv"),
    delimiter=",",
)
print(np.shape(X))
print(X)
