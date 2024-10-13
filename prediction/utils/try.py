import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error,r2_score
import argparse
from tqdm.auto import tqdm
import os
import glob
import math
from opc import PygPolymerDataset
import numpy as np
from model import GNN
from dataset_produce import SmilesRepeat    
import os
import ast
import json
import os.path as osp
import pandas as pd

import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import to_networkx

from sklearn.model_selection import train_test_split

from opc.utils.mol import smiles2graph
from opc.utils.split import scaffold_split, similarity_split
#from opc.utils.features import task_properties
def get_idx_split(split_type="random"):
        task_name = 'density'
        path = '/scratch365/yzhu25/challenge-code/prediction/data_pyg/prediction/{}/{}_raw_1'.format(task_name,task_name)

        raw_file_name = "{}_raw.csv".format(task_name)
        csv_file = osp.join(path,raw_file_name)
        data_df = pd.read_csv(csv_file)
        #print(len(data_df))
        #print('Splitting with random seed 42 and ratio 0.6/0.1/0.3')

        full_idx = list(range(len(data_df)))
        train_ratio, valid_ratio, test_ratio = 0.6, 0.1, 0.3
        train_idx, test_idx, train_df, test_df = train_test_split(full_idx, data_df, test_size=test_ratio, random_state=42)
        train_idx, valid_idx, _, _ = train_test_split(train_idx, train_idx, test_size=valid_ratio/(valid_ratio+train_ratio), random_state=42)
        df_train = pd.DataFrame({'train': train_idx})
        df_valid = pd.DataFrame({'valid': valid_idx})
        df_test = pd.DataFrame({'test': test_idx})
        #df_test.to_csv(path+'/results.csv',index=False)
        df_test_all = pd.DataFrame({'smiles':test_df['SMILES'],'tg_groundtruth':test_df['density']})
        df_train.to_csv(osp.join(path, 'train.csv.gz'), index=False, header=False, compression="gzip")
        df_valid.to_csv(osp.join(path, 'valid.csv.gz'), index=False, header=False, compression="gzip")
        df_test.to_csv(osp.join(path, 'test.csv.gz'), index=False, header=False, compression="gzip")
        df_test_all.to_csv('test_density.csv')

def im():
        df1 = pd.read_csv("test.csv")
    

        print(df1.shape[0])
if __name__=='__main__':
    get_idx_split()

