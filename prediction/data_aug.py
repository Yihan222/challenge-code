import pandas as pd
import os
import torch
import pickle
import numpy as np
import gzip
import os.path as osp
from dataset_produce import SmilesRepeat
from configparser import ConfigParser

char_lexicon = ['%','(',')','*','+','-','0','1','2','3','4','5','6','7','8','9','=','B','C','F','G','H','I','K','L','N','O','P','S','T','Z','[',']','a','b','c','d','e','i','l','n','o','r','s','/','\\','#']
char_dic = {'#': 0, '%': 1, '(': 2, ')': 3, '*': 4, '+': 5, '-': 6, '0': 7, '1': 8, '2': 9, '3': 10, '4': 11, '5': 12, '6': 13, '7': 14, '8': 15, '9': 16, '=': 17, 'B': 18, 'C': 19, 'F': 20, 'G': 21, 'H': 22, 'I': 23, 'K': 24, 'L': 25, 'N': 26, 'O': 27, 'P': 28, 'S': 29, 'T': 30, 'Z': 31, '[': 32, ']': 33, 'a': 34, 'b': 35, 'c': 36, 'd': 37, 'e': 38, 'i': 39, 'l': 40, 'n': 41, 'o': 42, 'r': 43, 's': 44,'/':45,'\\':46}


config = ConfigParser()
config.read('config.ini')
data_root = config.get('data','data_root')
test_file_path = config.get('data','test_file_path')
valid_file_path = config.get('data','valid_file_path')
train_file_path = config.get('data','train_file_path')
max_length_dic = config.get('data','max_length_dic')

def tok(poly_str):
    res = [char_dic[char] for char in poly_str]
    return res

def rnntokenize(task, rep):
    max_ls = []
    # store the dictionary of max token length
    for r in rep:
        dir_name = osp.join(data_root,task,str(r))
        tok_file = osp.join(dir_name,"raw.pkl")
        open_file = osp.join(dir_name,"raw.csv")   
        if not osp.exists(open_file):
            dataproduce = SmilesRepeat(r, task, root=data_root)
            dataproduce.repeat()
        if not osp.exists(tok_file):
            polymer_df = pd.read_csv(open_file)
            max_l = polymer_df['SMILES'].str.len().max()
            #max_l = 120
            toks = []
            for index, row in polymer_df.iterrows():
                poly = row['SMILES']
                res = tok(poly)
                res += [0] * (max_l-len(res))
                toks.append(res)
            max_ls.append([r,max_l])
            df = pd.DataFrame({'tk':toks,'tg':polymer_df['tg']})
            
            with open(tok_file,'wb') as file:
                pickle.dump(df,file)
            print("********************************* Tokens Saved! *********************************")
        else:
            print("********************************* Token File Existed! *********************************")
    '''
    max_ls_df = pd.DataFrame(columns=['repeat_time','max_token_len'],data=max_ls)
    if osp.exists(max_length_dic):
        max_ls_df.to_csv(max_length_dic, mode="a", header=False, index=False)
    else:
        max_ls_df.to_csv(max_length_dic, index=False)
    '''

def rsplit(task, rep):
    # Read the CSV files
    test_idx = read_gzipped_csv(test_file_path).values.T[0]
    valid_idx = read_gzipped_csv(valid_file_path).values.T[0]
    train_idx = read_gzipped_csv(train_file_path).values.T[0]
    if not isinstance(rep,list):
        rep = [rep]
    for r in rep:
        print(r)
        dir_name = osp.join(data_root,task,str(r))
        ori_tk_name = osp.join(dir_name,'raw.pkl')
        with open(ori_tk_name, 'rb') as file:
            oridf =pickle.load(file)
        for s in ['test','train','valid']:
            sub_dir_name = osp.join(dir_name,s)
            if not osp.exists(sub_dir_name):
                os.makedirs(sub_dir_name)
            tok_file = osp.join(sub_dir_name,'{}.pkl'.format(s))
            if not osp.exists(tok_file):
                if s == 'test':
                    df = oridf.iloc[test_idx]
                elif s == 'train':
                    df = oridf.iloc[train_idx]
                else:
                    df = oridf.iloc[valid_idx]
                with open(tok_file,'wb') as file:
                    pickle.dump(df,file)
                print("*********************** {} Saved! ************************".format(tok_file))
            else:
                print("*********************** {} Existed! ************************".format(tok_file))


def csvcatr(task, rep):
    # concatenate train or valid set
    r_name = '{}{}'.format(rep[0],rep[-1])
    for s in ['train','valid']:
        dir_name = osp.join(data_root,task,'concat2',r_name,s)
        if not osp.exists(dir_name):
            os.makedirs(dir_name)
        des = osp.join(dir_name,'{}.pkl'.format(s))
        if not osp.exists(des):
            ori = osp.join(data_root,task,str(rep[-1]),s,'{}.pkl'.format(s))
            with open(ori, 'rb') as file:
                ori_polymer_df =pickle.load(file)

            #train_idx = pd.read_csv('./rnn/train_idx.csv', header = None).values.T[0]
            x_list = ori_polymer_df['tk'].values.tolist()
            y = ori_polymer_df['tg']
            # padding training set to maximum length
            input_len = len(x_list[0])
            for x in x_list:
                x += [0] * (input_len-len(x))
            df = pd.DataFrame({'tk': x_list,'tg':y})
            for tr in rep[:-1]:
                file = osp.join(data_root,task,str(tr),s,'{}.pkl'.format(s))
                with open(file, 'rb') as file:
                    polymer_df =pickle.load(file)

                # obtain original training tokens
                cur_list = polymer_df['tk'].values.tolist()
                # padding training set to maximum length
                for x in cur_list:
                    x += [0] * (input_len-len(x))
                cur_df = pd.DataFrame({'tk': cur_list,'tg':y})
                df = pd.concat([df,cur_df],axis = 0,ignore_index=True)
            with open(des,'wb') as file:
                pickle.dump(df,file)
            print("********************************* Tokens Saved! *********************************")
        else:
            print("*********************** {} Existed! ************************".format(des))

# Function to read a gzipped CSV file
def read_gzipped_csv(file_path):
    with gzip.open(file_path, 'rt') as f:
        return pd.read_csv(f, header=None)

def gsplit(task, rep):
    # Read the CSV files
    test_idx = read_gzipped_csv(test_file_path).values.T[0]
    valid_idx = read_gzipped_csv(valid_file_path).values.T[0]
    train_idx = read_gzipped_csv(train_file_path).values.T[0]
    if not isinstance(rep,list):
        rep = [rep]
    for r in rep:
        dir_name = osp.join(data_root,task,str(r))
        ori_csv_name = osp.join(dir_name,'raw.csv')
        oridf = pd.read_csv(ori_csv_name)
        for s in ['test','train','valid']:
            sub_dir_name = osp.join(dir_name,s)
            if not osp.exists(sub_dir_name):
                os.makedirs(sub_dir_name)
            des = osp.join(sub_dir_name,'{}.csv'.format(s))
            if not osp.exists(des):
                if s == 'test':
                    df = oridf.iloc[test_idx]
                elif s == 'train':
                    df = oridf.iloc[train_idx]
                else:
                    df = oridf.iloc[valid_idx]

                df.to_csv(des, index=False,header=True) 
            else:
                print("*********************** {} Existed! ************************".format(des))



def csvcatg(task, rep):
    r_name = '{}{}'.format(rep[0],rep[-1])
    for s in ['train','valid']:
        dir_name = osp.join(data_root,task,'concat2',r_name,s)
        if not osp.exists(dir_name):
            os.makedirs(dir_name)
        des = osp.join(dir_name,'{}.csv'.format(s))
        if not osp.exists(des):

            all_data = []
            for r in rep:
                cur_raw_file = osp.join(data_root,task,str(r),s,'{}.csv'.format(s))
                df = pd.read_csv(cur_raw_file)
                all_data.append(df)
            result = pd.concat(all_data, axis=0,ignore_index=False)
            # SAVE CSV
            result.to_csv(des, index=False) 
        else:
            print("*********************** {} Existed! ************************".format(des))

 
def testmerge(a,b):
    '''
    target: test the intersection numbers of 2 arrays
    usage: testmerge(df1,df2)
    '''
    intersection = pd.merge(a, b, how='inner')
    # Count the number of intersecting lines
    intersection_count = intersection.shape[0]
    return intersection_count    

if __name__ == '__main__':
    pass
    