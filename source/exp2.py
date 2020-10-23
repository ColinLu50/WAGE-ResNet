import pickle
import numpy as np

with open('../model/w_np_list_ResNet20_1.pickle', 'rb') as f1:
    np_list = pickle.load(f1)
    print(len(np_list))

with open('../model/scale_list_ResNet20_1.pickle', 'rb') as f2:
    np_list = pickle.load(f2)
    print(len(np_list))