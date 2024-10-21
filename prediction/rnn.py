import pickle
import pandas as pd


import os
import os.path as osp
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, LSTM, Embedding, Bidirectional, TimeDistributed, Reshape
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model
from sklearn.metrics import mean_absolute_error, r2_score,root_mean_squared_error
import torch
import math
from data_aug import csvcatr

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


#tf.debugging.set_log_device_placement(True)



def getRNNmodel(LSTMunits, out_dim, input_dim):

	RNNmodel = Sequential()
	RNNmodel.add(Embedding(input_dim, out_dim))
	RNNmodel.add(Bidirectional(LSTM(LSTMunits, return_sequences=True)))
	RNNmodel.add(Bidirectional(LSTM(LSTMunits, return_sequences=True)))
	RNNmodel.add(TimeDistributed(Dense(int(LSTMunits/2), activation="relu")))
	RNNmodel.add(Reshape((int(LSTMunits/2*input_len),)))
	RNNmodel.add(Dense(1))

	return RNNmodel

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true,y_pred)
    rmse = root_mean_squared_error(y_true,y_pred)
    r2 = r2_score(y_true, y_pred)
    y1,y2 = [],[]
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            y1.append(0)
            y2.append(0)
        else:
            if y_pred[i] == 0:
                y1.append(0)
                y2.append(math.log(abs(y_true[i])))
            elif y_true[i] == 0:
                y1.append(math.log(abs(y_pred[i])))
                y2.append(0)
            else:
                if y_pred[i]/y_true[i] > 0:
                    y1.append(math.log(abs(y_pred[i])))
                else:
                    y1.append(0)
                y2.append(math.log(abs(y_true[i])))

    lgmae = mean_absolute_error(y1,y2)
    return mae,rmse,lgmae,r2

_use_checkpoint = False
if __name__=='__main__':
    m = 'rnn'
    task = 'tg'
    # train_rep can be a list of ints/lists, 
    # int numbers means use single dataset, 
    # list means use concatenate datasets of different repeating times
    train_rep = [[1,2,3,4,5,6,7,8,9,10]]
    test_rep = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
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
            df_save_name = "./rnn_res/repeat_{}.csv".format(r)
            res_csv_name = "./rnn_res/result.csv"
        else:
            data_path = "./data_pyg/prediction/{}/concat/{}".format(task,r)
            ck_path = './checkpoints/tg/rnn/concat/{}'.format(r)
            df_save_name = "./rnn_res/concat/repeat_{}.csv".format(r)
            res_csv_name = "./rnn_res/concat/result.csv"

        if not osp.exists(ck_path):
            os.makedirs(ck_path)   
 
        
        # load train/val datasets
        train_file = osp.join(data_path,'train','train.pkl')
        valid_file = osp.join(data_path,'valid','valid.pkl')

        with open(train_file, 'rb') as file:
            train_polymer_df =pickle.load(file)
        X_train = train_polymer_df['tk'].values.tolist()
        y_train = train_polymer_df['tg'].values.tolist()
        mt = len(X_train[0])

        with open(valid_file, 'rb') as file:
            valid_polymer_df =pickle.load(file)
        X_valid = valid_polymer_df['tk'].values.tolist()
        y_valid = valid_polymer_df['tg'].values.tolist()
        
        # all data should be padded to maximum length
        max_test_file = "./data_pyg/prediction/{}/{}/test/test.pkl".format(task,test_rep[-1])
        with open(max_test_file, 'rb') as file:
            max_test_df =pickle.load(file)
            mte = len(max_test_df['tk'].values.tolist()[0])
            #print(max_test_df)
        input_len = max(mt,mte)
        if input_len != mt:
            for x in X_train:
                x += [0] * (input_len-len(x))
            for x in X_valid:
                x += [0] * (input_len-len(x))
        '''
        full_idx = list(range(len(train_polymer_df)))
        # split test sets in not done this before
        if not osp.exists('./rnn/train_idx.csv'):
            train_idx, test_idx, _, test_df = train_test_split(full_idx, train_polymer_df, test_size=0.1, random_state=42)
            df_train = pd.DataFrame({'train': train_idx})
            df_test = pd.DataFrame({'test': test_idx})
            df_train.to_csv('./rnn/train_idx.csv', index=False, header=False)
            df_test.to_csv('./rnn/test_idx.csv', index=False, header=False)
        '''
        

        input_dim = 47
        out_dim = 15
        LSTMunits = 60
        results = {
            "model": [],
            "train_repeat_time":[],
            "test_repeat_time":[],
            "task": [],
            "test_mae":[],
            "test_rmse":[],
            "test_lgmae":[],
            "test_r2":[],
        }
        df = pd.DataFrame(results)
        # for different repeating times, calculate mean+-std on 10 models as final output
        dfs = [df] * test_num

        for i in range(10):

            RNNmodel = getRNNmodel(LSTMunits, out_dim, input_dim)            
            RNNmodel.compile(loss='mse', optimizer='adam', metrics=['mean_absolute_error'])
            
            # fit model
            # add callbacks
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=20)
            # learn: use models  trained on different dataset
            model_name = osp.join(ck_path,'{}_{}_{}_{}.hdf5'.format(LSTMunits,out_dim,r,i))
            mc = ModelCheckpoint(model_name, monitor='val_loss', mode='min', verbose=0, save_best_only=True)
            if not _use_checkpoint:
                history = RNNmodel.fit(np.array(X_train), np.array(y_train), validation_data=(np.array(X_valid), np.array(y_valid)), epochs=500, batch_size=128, callbacks=[es, mc])
                # save the history
                history_df = pd.DataFrame.from_dict(history.history)
                his_name = "history_{}.csv".format(r)
                his_path = osp.join(ck_path,his_name)
                history_df.to_csv(his_path, index=False)
            else:
                RNNmodel = load_model(model_name)
            # each model test for different repeating times
            print("Start testing...")
            for j in range(len(test_rep)):
                tr = test_rep[j]
                cur_df = dfs[j]
                test_file = "./data_pyg/prediction/{}/{}/test/test.pkl".format(task,tr)
                with open(test_file, 'rb') as file:
                    test_polymer_df =pickle.load(file)
                X_test = test_polymer_df['tk'].values.tolist()
                for x in X_test:
                    x += [0] * (input_len-len(x))
                y_test = np.array(test_polymer_df['tg'].values.tolist())


                y_pred = RNNmodel.predict(np.array(X_test))
                mae,rmse,lgmae,r2 = evaluate(y_test,y_pred)
                new_results = {
                    "model": m,
                    "train_repeat_time":r,
                    "test_repeat_time":tr,
                    "task": task,
                    "test_mae":mae,
                    "test_rmse":rmse,
                    "test_lgmae":lgmae,
                    "test_r2":r2,
                }
                new_df = pd.DataFrame([new_results])
                '''
                if osp.exists(df_save_name):
                    new_df.to_csv(df_save_name, mode="a", header=False, index=False)
                else:
                    new_df.to_csv(df_save_name, index=False)
                '''
                cur_df = pd.concat([cur_df, new_df], ignore_index=True)
                dfs[j] = cur_df

        for df in dfs:
            # Calculate mean and std, and format them as "mean±std".
            summary_cols = ["model", "train_repeat_time","test_repeat_time", "task"]
            df_mean = df.groupby(summary_cols).mean().round(4)
            df_std = df.groupby(summary_cols).std().round(4)

            df_mean = df_mean.reset_index()
            df_std = df_std.reset_index()
            df_summary = df_mean[summary_cols].copy()
            # Format 'train', 'valid' columns as "mean±std".
            for metric in ['r2','rmse','mae','lgmae']:
                col_name = 'test_'+metric
                df_summary[col_name] = df_mean[col_name].astype(str) + "±" + df_std[col_name].astype(str)

            # Save and print the summary DataFrame.
            if osp.exists(res_csv_name):
                df_summary.to_csv(res_csv_name, mode="a", header=False, index=False)
            else:
                df_summary.to_csv(res_csv_name, index=False)
            print(df_summary)
                    

