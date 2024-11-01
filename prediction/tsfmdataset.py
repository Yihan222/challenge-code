import torch
import sys
import pandas as pd
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset

from torch.nn.utils.rnn import pad_sequence
from dataset_produce import SmilesRepeat
from data_aug import gsplit

char_lexicon = ['<pad>','%','(',')','*','+','-','0','1','2','3','4','5','6','7','8','9','=','B','C','F','G','H','I','K','L','N','O','P','S','T','Z','[',']','a','b','c','d','e','i','l','n','o','r','s','/','\\','#']
char_dic = {'<pad>': 0, '%': 1, '(': 2, ')': 3, '*': 4, '+': 5, '-': 6, '0': 7, '1': 8, '2': 9, '3': 10, '4': 11, '5': 12, '6': 13, '7': 14, '8': 15, '9': 16, '=': 17, 'B': 18, 'C': 19, 'F': 20, 'G': 21, 'H': 22, 'I': 23, 'K': 24, 'L': 25, 'N': 26, 'O': 27, 'P': 28, 'S': 29, 'T': 30, 'Z': 31, '[': 32, ']': 33, 'a': 34, 'b': 35, 'c': 36, 'd': 37, 'e': 38, 'i': 39, 'l': 40, 'n': 41, 'o': 42, 'r': 43, 's': 44,'/':45,'\\':46,'#':47}

class TransPolyDataset(Dataset):
    def __init__(self,data_root,rep,set_name,task='tg'):
        self.data_root = data_root
        self.rep = rep
        self.set_name = set_name
        self.task = task
        self.data, self.prop = self.getdata()

    def tok(self, poly_str):
        res = [char_dic[char] for char in poly_str]
        return torch.tensor(res)

    def getdata(self):
        toks = []
        prop_c = False
        self.first_line = True
        for r in self.rep:
            ori_dataset_path = osp.join(self.data_root,str(r))
            if not osp.exists(ori_dataset_path):
                # no dataset for this specific repeat time, create dataset
                dataproduce = SmilesRepeat(r)
                dataproduce.repeat()

            dir_name = osp.join(self.data_root,str(r),self.set_name)
            if not osp.exists(dir_name):
                # no split files for this dataset, splitting...
                gsplit(r,self.task)
                
            file_name = osp.join(dir_name,"{}.csv".format(self.set_name))   
            polymer_df = pd.read_csv(file_name)
            for index, row in polymer_df.iterrows():
                poly = row['SMILES']
                res = self.tok(poly)
                toks.append(res)
            if not prop_c:
                props = torch.tensor(polymer_df['tg'].values)
                prop_c = True
            else:
                props = torch.cat([props, torch.tensor(polymer_df['tg'].values)], 0)
        #padded_toks = pad_sequence(toks, batch_first=True, padding_value=0)
        return toks, props
    
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        return self.data[idx], self.prop[idx]

def _collate_fn(batch):
    props = []
    datas = []
    for i in range(len(batch)):
        datas.append(batch[i][0])
        props.append(batch[i][1])
    datas = pad_sequence(datas, batch_first=True, padding_value=0)
    datas = torch.stack(list(datas), dim=0)
    props = torch.stack(props)
    return datas, props

class TransPolyDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(TransPolyDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

if __name__=='__main__':
    pass