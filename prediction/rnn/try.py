import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error,r2_score


import os
import numpy as np
import torch
task = 'tg'
rep = [1,2,3,4,5,6,7,8,9,10]
for r in rep:
    tok_file = "./data_pyg/prediction/{}/{}_raw_{}/{}_raw_120_tk.pkl".format(task,task,r,task)
    test_idx = pd.read_csv('./rnn/test_idx.csv', header = None).values.T[0]
    with open(tok_file, 'rb') as file:
        test_df =pickle.load(file)
    test_set = test_df['tk'].iloc[test_idx].values.tolist()
    # padding test set
    input_len = 1587
    for t in test_set:
        t += [0] * (input_len-len(t))
    test_tk_file = "./data_pyg/prediction/{}/{}_raw_{}/tg_test_tk.npy".format(task,task,r)
    test_set2 = np.load(test_tk_file)
    if test_set[0] == test_set2[0]:
        print("***************")

    break