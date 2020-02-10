import numpy as np
from preprocessings import load_from_pickle

y_test = load_from_pickle(filename="y-test.pkl")
print(np.bincount(y_test))