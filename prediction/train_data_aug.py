import pickle
import pandas as pd


import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score,root_mean_squared_error
import torch

def csvcat():
    ori = "./data_pyg/prediction/{}/{}_raw_1/{}_raw_120_tk.pkl".format(task,task,task)
    with open(ori, 'rb') as file:
        ori_polymer_df =pickle.load(file)
    des = './data_pyg/prediction/tg/oneToten_raw_120_tk.pkl'

    train_idx = pd.read_csv('./rnn/train_idx.csv', header = None).values.T[0]

    x_train_list = ori_polymer_df['tk'].iloc[train_idx].values.tolist()
    y_train = ori_polymer_df['tg'].iloc[train_idx]
    #print(y_train)
    # padding training set to maximum length
    input_len = 1587
    for x in x_train_list:
        x += [0] * (input_len-len(x))
    #print(x_train_list)
    df = pd.DataFrame({'tk': x_train_list,'tg':y_train})
    #print(train_df)
    #train_df = x_train_df.join(y_train)

    for tr in [2,3,4,5,6,7,8,9,10]:

        file = "./data_pyg/prediction/{}/{}_raw_{}/{}_raw_120_tk.pkl".format(task,task,tr,task)
        # import the ori test dataset
        with open(file, 'rb') as file:
            polymer_df =pickle.load(file)

        # obtain original training tokens
        cur_list = polymer_df['tk'].iloc[train_idx].values.tolist()
        #print(y_train)
        # padding training set to maximum length
        for x in cur_list:
            x += [0] * (input_len-len(x))
        cur_df = pd.DataFrame({'tk': cur_list,'tg':y_train})
        df = pd.concat([df,cur_df],axis = 0,ignore_index=True)
            #print(train_df)
        print(len(df))
    with open(des,'wb') as file:
        pickle.dump(df,file)
    print("********************************* Tokens Saved! *********************************")

def csvcatg():
    des_train_file =  "./data_pyg/prediction/tg/tg_raw_11/tg_raw.csv"
    #train_idx = pd.read_csv('./data_pyg/prediction/tg/tg_raw_1/split/random/train.csv.gz', compression='gzip', header = None).values.T[0]
    #valid_idx = pd.read_csv('./data_pyg/prediction/tg/tg_raw_1/split/random/valid.csv.gz', compression='gzip', header = None).values.T[0]
    all_data = []
    for r in rep:
        cur_raw_file = 'data_pyg/prediction/{}/{}_raw_{}/{}_raw.csv'.format(task,task,r,task)
        df = pd.read_csv(cur_raw_file)
        #df_t = df.iloc[train_idx]
        #df_v = df.iloc[valid_idx]
        all_data.append(df)


    result = pd.concat(all_data, axis=0,ignore_index=False)
    #print(len(result))
    # 将结果保存为新的CSV文件
    result.to_csv(des_train_file, index=False) 

def gnnidx():
    train_idx = pd.read_csv('./data_pyg/prediction/tg/tg_raw_1/split/random/train.csv.gz', compression='gzip', header = None).values.T[0].tolist()
    valid_idx = pd.read_csv('./data_pyg/prediction/tg/tg_raw_1/split/random/valid.csv.gz', compression='gzip', header = None).values.T[0].tolist()
    rest = train_idx
    resv = valid_idx
    tadd = [[],[],[],[],[],[],[],[],[]]
    vadd = [[],[],[],[],[],[],[],[],[]]
    for i in train_idx:
        for r in range(1,10):
            tadd[r-1].append(7174*r+i)
    for v in valid_idx:
        for r in range(1,10):
            vadd[r-1].append(7174*r+v)
    for r in range(1,10):
        rest += tadd[r-1]
        #print(len(rest))
        resv += vadd[r-1]
        #print(len(resv))
    #print(rest[0],rest[4303])
    #print(df.iloc[rest[0]],df.iloc[rest[4303]])
    # train 4303/ valid 718
    #print(dff)
    rest = np.array(rest)
    resv = np.array(resv)
    df_train = pd.DataFrame({'train': rest})
    df_valid = pd.DataFrame({'valid': resv})
    #df_test = pd.DataFrame({'test': test_idx})
    #df_test_all = pd.DataFrame({'smiles':test_df['SMILES'],'tg_groundtruth':test_df['tg']})
    df_train.to_csv('./data_pyg/prediction/tg/tg_raw_11/split/random/train.csv.gz', index=False, header=False, compression="gzip")
    df_valid.to_csv('./data_pyg/prediction/tg/tg_raw_11/split/random/valid.csv.gz', index=False, header=False, compression="gzip")

if __name__=='__main__':
    task = 'tg'
    rep = [1,2,3,4,5,6,7,8,9,10]
    #csvcat()
    df = pd.read_csv('./data_pyg/prediction/tg/tg_raw_11/tg_raw.csv')
    #gnnidx()
    '''
    train_idx = pd.read_csv('./data_pyg/prediction/tg/tg_raw_11/split/random/train.csv.gz', compression='gzip', header = None).values.T[0].tolist()
    print(len(train_idx))
    valid_idx = pd.read_csv('./data_pyg/prediction/tg/tg_raw_11/split/random/valid.csv.gz', compression='gzip', header = None).values.T[0].tolist()
    print(len(valid_idx))'''
    test_1 = './data_pyg/prediction/tg/tg_raw_1/tg_test_tk.npy'
    t1 = np.load(test_1)
    test_2 = './data_pyg/prediction/tg/tg_raw_2/tg_test_tk.npy'
    t2 = np.load(test_2)
    print(test_1[0],test_2[0])
