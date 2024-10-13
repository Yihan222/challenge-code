import pickle
import pandas as pd
import csv
import torch
from dataset_produce import SmilesRepeat

import os
import os.path as osp
from opc import PygPolymerDataset
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, LSTM, Embedding, Bidirectional, TimeDistributed, Reshape
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model
from sklearn.metrics import mean_absolute_error, r2_score

os.environ["CUDA_VISIBLE_DEVICES"]="0" 
tf.config.experimental.set_visible_devices([],'GPU')
TF_ENABLE_ONEDNN_OPTS=0
char_lexicon = ['#','%','(',')','*','+','-','0','1','2','3','4','5','6','7','8','9','=','B','C','F','G','H','I','K','L','N','O','P','S','T','Z','[',']','a','b','c','d','e','i','l','n','o','r','s','/','\\']
#char_dic = {k: v for v, k in enumerate(char_lexicon)}
char_dic = {'#': 0, '%': 1, '(': 2, ')': 3, '*': 4, '+': 5, '-': 6, '0': 7, '1': 8, '2': 9, '3': 10, '4': 11, '5': 12, '6': 13, '7': 14, '8': 15, '9': 16, '=': 17, 'B': 18, 'C': 19, 'F': 20, 'G': 21, 'H': 22, 'I': 23, 'K': 24, 'L': 25, 'N': 26, 'O': 27, 'P': 28, 'S': 29, 'T': 30, 'Z': 31, '[': 32, ']': 33, 'a': 34, 'b': 35, 'c': 36, 'd': 37, 'e': 38, 'i': 39, 'l': 40, 'n': 41, 'o': 42, 'r': 43, 's': 44,'/':45,'\\':46}

def tok(poly_str):

    res = [char_dic[char] for char in poly_str]
    #for c in char_tokens:
    #   res.append(char_dic[c])
    return res



if __name__=='__main__':
    task = 'tg'
    rep = [1,2,3,4,5,6,7,8,9,10]
    max_ls = []
    with tf.device('/gpu:0'):
        for r in rep:
        
            path = "./rnn/{}_{}/".format(task,r)
            tok_file = "./data_pyg/prediction/{}/{}_raw_{}/{}_raw_120_tk.pkl".format(task,task,r,task)

            open_file = "./data_pyg/prediction/{}/{}_raw_{}/{}_raw_120.csv".format(task,task,r,task)
            res_file = "./rnn/res_{}.npy".format(r)
            
            if not os.path.exists(open_file):
                dataproduce = SmilesRepeat(r, task, root='data_pyg/prediction/')
                dataproduce.repeat()
            if not os.path.exists(tok_file):
                polymer_df = pd.read_csv(open_file)
                max_l = polymer_df['SMILES'].str.len().max()
                #max_l = 120
                toks = []
                for index, row in polymer_df.iterrows():
                    poly = row['SMILES']
                    res = tok(poly)
                    res += [0] * (max_l-len(res))
                    #print(res)
                    toks.append(res)
                    #print(encode(poly)
                max_ls.append(max_l)
                df = pd.DataFrame({'tk':toks,'tg':polymer_df['tg']})
                
                with open(tok_file,'wb') as file:
                    pickle.dump(df,file)
                print("********************************* Tokens Saved! *********************************")
