import pickle
import pandas as pd
import sys
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
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')
max_length_dic = config.get('data','max_length_dic')
input_dim = int(config.get('rnn','input_dim'))
out_dim = int(config.get('rnn','out_dim'))
LSTMunits = int(config.get('rnn','LSTMunits'))

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


#tf.debugging.set_log_device_placement(True)

class MainRNN():
    def __init__(self, train_rep, task_name = 'tg', test_rep = [1], seed_num = 10, pad_size = None, _use_ck = False):
        # pad_size is actually the second number in the ck directory, which specifies the test set length you use in your training
        self.r = train_rep
        self.task = task_name
        self.model_type = 'rnn'
        self._use_ck = _use_ck
        if isinstance(train_rep,list):
            self.train_rep = train_rep
        else:
            self.train_rep = [train_rep]
        if isinstance(test_rep,list):
            self.test_rep = test_rep
        else:
            self.test_rep = [test_rep]

        if self._use_ck:
            if not pad_size:
                print("************ You need to specify the padding size of your rnn! ************")
                sys.exit()
        self.seed_num = seed_num
        if isinstance(train_rep,list):
            self._use_concat_train = True
            csvcatr(task_name, train_rep)
            max_r = train_rep[-1]
            self.r = int('{}{}'.format(train_rep[0],max_r))
            self.data_path = './data_pyg/prediction/tg/concat2/{}'.format(self.r)
            self.ck_path_first = './checkpoints/tg/rnn/concat2/'
            self.df_save_name = "./res/rnn_res/concat2/repeat_{}.csv".format(self.r)
            self.res_csv_name = "./res/rnn_res/concat2/result.csv"
        else:
            self._use_concat_train = False            
            max_r = train_rep
            self.r = train_rep
            self.data_path = './data_pyg/prediction/tg/{}'.format(self.r)
            self.ck_path_first = './checkpoints/tg/rnn/'
            self.df_save_name = "./res/rnn_res/repeat_{}.csv".format(train_rep)
            self.res_csv_name = "./res/rnn_res/result.csv"
        
        if self._use_ck:
            self.ck_path = osp.join(self.ck_path_first,'{}_{}'.format(self.r,pad_size))
            if not osp.exists(self.ck_path):
                print("************ No such rnn exist! ************")
                sys.exit()
        else:
            self.ck_path = osp.join(self.ck_path_first,'{}_{}'.format(self.r,self.test_rep[-1]))
            if not osp.exists(self.ck_path):
                os.makedirs(self.ck_path) 
    
        # figure out the maximum input length for training
        # consider training & testing needed, e.g., train on single dataset 10 but test on 15
        try:
            mldf = pd.read_csv(max_length_dic)
            self.input_len = max(int(mldf[mldf['repeat_time'] == max_r].max_token_len.iloc[0]), int(mldf[mldf['repeat_time'] == test_rep[-1]].max_token_len.iloc[0]))
        except Exception as e:
            print(f'An error occurred: {e}')
        

    def getRNNmodel(self):

        RNNmodel = Sequential()
        RNNmodel.add(Embedding(input_dim, out_dim))
        RNNmodel.add(Bidirectional(LSTM(LSTMunits, return_sequences=True)))
        RNNmodel.add(Bidirectional(LSTM(LSTMunits, return_sequences=True)))
        RNNmodel.add(TimeDistributed(Dense(int(LSTMunits/2), activation="relu")))
        RNNmodel.add(Reshape((int(LSTMunits/2*self.input_len),)))
        RNNmodel.add(Dense(1))

        return RNNmodel

    def evaluate(self,y_true, y_pred):
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

    def model_seed(self,i,X_train=None,y_train=None,X_valid=None,y_valid=None):
        
        RNNmodel = self.getRNNmodel()            
        RNNmodel.compile(loss='mse', optimizer='adam', metrics=['mean_absolute_error'])
        model_name = osp.join(self.ck_path,'{}_{}_{}_{}.hdf5'.format(LSTMunits,out_dim,self.r,i))
        if not self._use_ck:
            # add callbacks
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=20)
            mc = ModelCheckpoint(model_name, monitor='val_loss', mode='min', verbose=0, save_best_only=True)
            
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
            # fit model
            history = RNNmodel.fit(np.array(X_train), np.array(y_train), validation_data=(np.array(X_valid), np.array(y_valid)), epochs=500, batch_size=128, callbacks=[es, mc])
            # save the history
            history_df = pd.DataFrame.from_dict(history.history)
            his_name = "history_{}.csv".format(self.r)
            his_path = osp.join(self.ck_path,his_name)
            history_df.to_csv(his_path, index=False)
        else:
            RNNmodel = load_model(model_name)
        return RNNmodel
            
    def main(self): 
        # load train/val datasets      
        if not self._use_ck:
            train_file = osp.join(self.data_path,'train','train.pkl')
            with open(train_file, 'rb') as file:
                train_polymer_df =pickle.load(file)
            X_train = train_polymer_df['tk'].values.tolist()
            y_train = train_polymer_df['tg'].values.tolist()

            valid_file = osp.join(self.data_path,'valid','valid.pkl')
            with open(valid_file, 'rb') as file:
                valid_polymer_df =pickle.load(file)
            X_valid = valid_polymer_df['tk'].values.tolist()
            y_valid = valid_polymer_df['tg'].values.tolist()    

            if self.input_len != len(X_train[0]):
                for x in X_train:
                    x += [0] * (self.input_len-len(x))
                for x in X_valid:
                    x += [0] * (self.input_len-len(x))
                  
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
        test_num = len(self.test_rep)   
        df = pd.DataFrame(results)
        # for different repeating times, calculate mean+-std on 10 models as final output
        dfs = [df] * test_num
        for i in range(self.seed_num):
            if self._use_ck:
                RNNmodel = self.model_seed(i)
            else:
                RNNmodel = self.model_seed(i,X_train,y_train,X_valid,y_valid)

            # each model test for different repeating times
            print("Start testing...")
            for j in range(test_num):
                tr = self.test_rep[j]
                cur_df = dfs[j]
                test_file = "./data_pyg/prediction/{}/{}/test/test.pkl".format(self.task,tr)
                with open(test_file, 'rb') as file:
                    test_polymer_df =pickle.load(file)
                X_test = test_polymer_df['tk'].values.tolist()
                if self.input_len != len(X_test[0]):
                    for x in X_test:
                        x += [0] * (self.input_len-len(x))
                y_test = np.array(test_polymer_df['tg'].values.tolist())
                y_pred = RNNmodel.predict(np.array(X_test))
                mae,rmse,lgmae,r2 = self.evaluate(y_test,y_pred)
                new_results = {
                    "model": self.model_type,
                    "train_repeat_time":self.r,
                    "test_repeat_time":tr,
                    "task": self.task,
                    "test_mae":mae,
                    "test_rmse":rmse,
                    "test_lgmae":lgmae,
                    "test_r2":r2,
                }
                new_df = pd.DataFrame([new_results])
                '''
                if osp.exists(self.df_save_name):
                    new_df.to_csv(self.df_save_name, mode="a", header=False, index=False)
                else:
                    new_df.to_csv(self.df_save_name, index=False)
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
            if osp.exists(self.res_csv_name):
                df_summary.to_csv(self.res_csv_name, mode="a", header=False, index=False)
            else:
                df_summary.to_csv(self.res_csv_name, index=False)
            print(df_summary)
                        


if __name__=='__main__':
    pass