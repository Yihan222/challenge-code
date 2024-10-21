import pandas as pd
import numpy as np
import pickle
import os
import os.path as osp
from data_aug import gsplit,rsplit,csvcatg,csvcatr
from dataset_produce import SmilesRepeat
import heapq
_use_ck = True
_use_param = True
if __name__=='__main__':
    m = 'rnn'
    task = 'tg'
    # train_rep can be a list of ints/lists, 
    # int numbers means use single dataset, 
    # list means use concatenate datasets of different repeating times
    train_rep = [10]
    test_rep = [11,12,13,14,15,16,17,18,19,20]

    test_num = len(test_rep)
    
    for r in train_rep:
        if isinstance(r,list):
            _use_concat_train = True
            # concatenate datasets
            csvcatr(task,r)
            r = int('{}{}'.format(r[0],r[-1]))
        else:
            _use_concat_train = False
        if not _use_concat_train:
            data_path = "./data_pyg/prediction/{}/{}".format(task,r)
            ck_path = './checkpoints/tg/rnn/{}'.format(r)
            res_path = "./rnn_res/{}".format(r)
            df_save_name = "./rnn_res/repeat_{}.csv".format(r)
            res_csv_name = "./rnn_res/result.csv"
        else:
            data_path = "./data_pyg/prediction/{}/concat/{}".format(task,r)
            ck_path = './checkpoints/tg/rnn/concat/{}'.format(r)
            res_path = "./rnn_res/concat/{}".format(r)
            df_save_name = "./rnn_res/concat/repeat_{}.csv".format(r)
            res_csv_name = "./rnn_res/concat/result.csv"

        if not osp.exists(ck_path):
            os.makedirs(ck_path)   
        if not osp.exists(res_path):
            os.makedirs(res_path)   

        # load train/val datasets
        train_file = osp.join(data_path,'train','train.pkl')
        valid_file = osp.join(data_path,'valid','valid.pkl')

        with open(train_file, 'rb') as file:
            train_polymer_df =pickle.load(file)
        X_train = train_polymer_df['tk'].values.tolist()
        y_train = train_polymer_df['tg'].values.tolist()
        mt = len(X_train[0])
        print(mt)
    