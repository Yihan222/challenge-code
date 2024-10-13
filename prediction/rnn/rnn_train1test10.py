import pickle
import pandas as pd


import os
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

def evaluate(y):
    try:
        mae = mean_absolute_error(y_test,y)
    except:
        mae = np.nan
    try: 
        rmse = root_mean_squared_error(y_test,y)
    except:
        rmse = np.nan
    y1,y2 = [],[]

    for i in range(len(y_test)):
        if y_test[i] != np.nan and y[i] and y_test[i]/y[i]>0:
            y1.append(abs(y_test[i]))
            y2.append(abs(y[i]))
    try:
        lgmae = mean_absolute_error(np.log(y1),np.log(y2))
    except:
        lgmae = np.nan
    try:
        r2 = r2_score(y_test, y)
    except:
        r2=np.nan
    return mae, rmse,lgmae,r2

if __name__=='__main__':
    models = ['rnn']
    task = 'tg'
    rep = [1,2,3,4,5,6,7,8,9,10]
    for m in models:
        #ori_file = "./data_pyg/prediction/{}/{}_raw_{}/{}_raw_120_tk.pkl".format(task,task,m,task)
        train_file = "./data_pyg/prediction/tg/oneToten_raw_120_tk.pkl"


        # import the ori test dataset
        with open(train_file, 'rb') as file:
            train_polymer_df =pickle.load(file)
        full_idx = list(range(len(train_polymer_df)))
        # split test sets in not done this before
        if not os.path.exists('./rnn/train_idx.csv'):
            train_idx, test_idx, _, test_df = train_test_split(full_idx, train_polymer_df, test_size=0.1, random_state=42)
            df_train = pd.DataFrame({'train': train_idx})
            df_test = pd.DataFrame({'test': test_idx})
            df_train.to_csv('./rnn/train_idx.csv', index=False, header=False)
            df_test.to_csv('./rnn/test_idx.csv', index=False, header=False)

        # train/test set ids are fixed for different rnns
        #train_idx = pd.read_csv('./rnn/train_idx.csv', header = None).values.T[0]
        test_idx = pd.read_csv('./rnn/test_idx.csv', header = None).values.T[0]
        # obtain original training tokens
        input_len = 1587
        #X_train_list = train_polymer_df['tk'].iloc[train_idx].values.tolist()

        X_train = np.array(train_polymer_df['tk'].values.tolist())
        y_train = np.array(train_polymer_df['tg'].values.tolist())
        # get maximum tokens length
        
        '''
        max_l_file = "./data_pyg/prediction/{}/{}_raw_10/{}_raw_120_tk.pkl".format(task,task,task)

        with open(max_l_file, 'rb') as file:
            max_l_polymer_df =pickle.load(file)
        max_l_train = max_l_polymer_df['tk'].iloc[train_idx].values.tolist()
        input_len = len(max_l_train[0])
        
        # padding training set to maximum length
        
        for x in X_train:
            x += [0] * (input_len-len(x))
        X_train = np.array(X_train)
        '''
        y_test = np.load('./rnn/tg_test.npy')

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
        dfs = [df] * 10

        for i in range(10):
            path = "./rnn/models/inf_models/"
            RNNmodel = getRNNmodel(LSTMunits, out_dim, input_dim)
            #use_gpu = torch.cuda.is_available()
            
            RNNmodel.compile(loss='mse', optimizer='adam', metrics=['mean_absolute_error'])
            
            # fit model
            # add callbacks
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=20)
            # learn: use models  trained on different dataset
            model_name = path + "best_model_" + str(LSTMunits) + "_" + str(out_dim) +"1to10.hdf5"


            mc = ModelCheckpoint(model_name, monitor='val_loss', mode='min', verbose=0, save_best_only=True)

            history = RNNmodel.fit(X_train, y_train, validation_split=0.2, epochs=500, batch_size=128, callbacks=[es, mc])

            # save the history
            history_df = pd.DataFrame.from_dict(history.history)
            name = "history_" + str(LSTMunits) + "_" + str(out_dim) + "1to10.csv"
            history_df.to_csv(path+name, index=True)
            # each model test for different repeating times
            for r in rep:
                cur_df = dfs[r-1]
                dir_name = "./rnn/res/inf_res_1to10"
                res_name = "res_{}{}.npy".format(r,i)
                if not os.path.exists(dir_name):
                    os.mkdir(dir_name)
                res_file = os.path.join(dir_name,res_name)
                # pad test set to fixed length for different numbers of repeating units
                '''
                tok_file = "./data_pyg/prediction/{}/{}_raw_{}/{}_raw_120_tk.pkl".format(task,task,r,task)
                test_idx = pd.read_csv('./rnn/test_idx.csv', header = None).values.T[0]
                with open(tok_file, 'rb') as file:
                    test_df =pickle.load(file)
                test_set = test_df['tk'].iloc[test_idx].values.tolist()
                # padding test set
                input_len = 1587
                for t in test_set:
                    t += [0] * (input_len-len(t))
                save_name =  "./data_pyg/prediction/{}/{}_raw_{}/{}_test_tk.npy".format(task,task,r,task)
                np.save(save_name,test_set)
                '''
                test_tk_file = "./data_pyg/prediction/{}/{}_raw_{}/tg_test_tk.npy".format(task,task,r)
                test_set = np.load(test_tk_file)
                preds = RNNmodel.predict(np.array(test_set))
                np.save(res_file,preds)
                mae,rmse,lgmae,r2 = evaluate(preds)
                new_results = {
                    "model": m,
                    "train_repeat_time":11,
                    "test_repeat_time":r,
                    "task": task,
                    "test_mae":mae,
                    "test_rmse":rmse,
                    "test_lgmae":lgmae,
                    "test_r2":r2,
                }
                cur_df = pd.concat([cur_df, pd.DataFrame([new_results])], ignore_index=True)
                dfs[r-1] = cur_df
        for df in dfs:
            # Calculate mean and std, and format them as "mean±std".
            summary_cols = ["model", "train_repeat_time","test_repeat_time", "task"]
            df_mean = df.groupby(summary_cols).mean().round(4)
            df_std = df.groupby(summary_cols).std().round(4)

            df_mean = df_mean.reset_index()
            df_std = df_std.reset_index()
            df_summary = df_mean[summary_cols].copy()
            # Format 'train', 'valid' columns as "mean±std".
            for metric in ['r2','mae','rmse','lgmae']:
                col_name = 'test_'+metric
                df_summary[col_name] = df_mean[col_name].astype(str) + "±" + df_std[col_name].astype(str)

            # Save and print the summary DataFrame.
            res_csv_name = "./rnn/result_rnn_tg.csv"      
            if os.path.exists(res_csv_name):
                df_summary.to_csv(res_csv_name, mode="a", header=False, index=False)
            else:
                df_summary.to_csv(res_csv_name, index=False)
            print(df_summary)

            
                    

