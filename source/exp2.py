import pickle
import numpy as np

with open('../model/w_np_list_test.pickle', 'rb') as f1:
    np_list = pickle.load(f1)
    print(len(np_list))

with open('../model/scale_list_test.pickle', 'rb') as f2:
    np_list = pickle.load(f2)
    print(len(np_list))
    
for weights_np in np_list:
    weights_np = weights_np.reshape([-1])
    print(weights_np[:10])