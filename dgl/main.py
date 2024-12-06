import numpy as np
import os
import os.path as osp
import sys
import pandas as pd
import socket
import random
import glob
import argparse, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from class_baseline.nets.load_net import gnn_model # import GNNs
from class_baseline.utils import seed_torch, print_info
#from data.data import LoadData # import dataset
from opc.dataset_pyg import PygPolymerDataset
from pygtodgl import DGLDatasetFromPyG,collate_gnn,collate_dense_gnn


def view_model_param(MODEL_NAME, net_params):
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    #print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param

def train_val_pipeline(MODEL_NAME, DATASET_NAME, params, net_params, dirs):
    
    root_ckpt_dir, train_info_name, test_info_name = dirs
    avg_convergence_epochs = []

    train_dataset = PygPolymerDataset(name="prediction", root="data_pyg", set_name="train",task_name=DATASET_NAME,repeat_times=params["train_repeat_time"],_use_concat_train =params["use_concat_train"])
    valid_dataset = PygPolymerDataset(name="prediction", root="data_pyg", set_name="valid",task_name=DATASET_NAME,repeat_times=params["train_repeat_time"],_use_concat_train =params["use_concat_train"])
    test_dataset = PygPolymerDataset(name="prediction", root="data_pyg", set_name="test",task_name=DATASET_NAME,repeat_times=1,_use_concat_train = False)
    dgl_train_dataset = DGLDatasetFromPyG(train_dataset)
    dgl_valid_dataset = DGLDatasetFromPyG(valid_dataset)
    dgl_test_dataset = DGLDatasetFromPyG(test_dataset)
    '''
    if MODEL_NAME in ['GCN', 'GAT']:
        if net_params['self_loop']:
            print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            dataset._add_self_loops()
    '''
    
    # Write the network and optimization hyper-parameters in folder config/
    info = {
        "model": MODEL_NAME,
        "params":params,
        "net_params":net_params,
        "valid_rmse":[],
        "test_rmse":[],
        "test_r2":[],
        "avg_convergence_epochs":[]
    }

    # At any point you can hit Ctrl + C to break out of training early.
    results = {'valid_rmse': [], 'test_rmse': [], 'test_r2':[]}
    for i in range(params['seed']):
        seed_torch(i)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = gnn_model(MODEL_NAME, net_params).to(device)
        optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])

        # batching exception for Diffpool
        drop_last = True if MODEL_NAME == 'DiffPool' else False

        if MODEL_NAME in ['RingGNN', '3WLGNN']:
            # import train functions specific for WL-GNNs
            from class_baseline.utils import training_dense as training, validate_dense as validate

            train_loader = DataLoader(dgl_train_dataset, shuffle=True, collate_fn=collate_dense_gnn)
            valid_loader = DataLoader(dgl_valid_dataset, shuffle=False, collate_fn=collate_dense_gnn)
            test_loader = DataLoader(dgl_test_dataset, shuffle=False, collate_fn=collate_dense_gnn)

        else:
            # import train functions for all other GCNs
            from class_baseline.utils import training_sparse as training, validate_sparse as validate

            train_loader = DataLoader(dgl_train_dataset, batch_size=params['batch_size'], shuffle=True, drop_last=drop_last, collate_fn=collate_gnn)
            valid_loader = DataLoader(dgl_valid_dataset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=collate_gnn)
            test_loader = DataLoader(dgl_test_dataset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=collate_gnn)

        best_epoch = 0
        print("Start training...")

        for epoch in range(params['epochs']):
            training(model, train_loader, optimizer, device)
            valid_perf = validate(model, valid_loader,device)
            if epoch == 0 or valid_perf['rmse'] <  best_valid['rmse']:
                best_valid = valid_perf
                best_train = validate(model, train_loader, device)
                best_test = validate(model, test_loader, device)

                best_epoch = epoch
                state = {
                    "model": model.state_dict(),
                }
                # Saving checkpoint
                ck_name= os.path.join(root_ckpt_dir, f"{i}.pkl")
                torch.save(state, ck_name)
            else:   
                # save checkpoints
                if epoch - best_epoch > params['patience']:
                    break
        results['test_r2'].append(best_test['r2'])
        results['test_rmse'].append(best_test['rmse'])
        results['valid_rmse'].append(best_valid['rmse'])
        avg_convergence_epochs.append(best_epoch)
        print('Finished training of {}th model! Best validation results from epoch {}.'.format(i,best_epoch))
        print('Model saved as {}.'.format(ck_name))
        print_info('train', best_train)
        print_info('valid', best_valid)
    ace = np.mean(np.array(avg_convergence_epochs))
    print("AVG CONVERGENCE Time (Epochs): {:.4f}".format(ace))
    for mode, nums in results.items():
        mean,std = round(np.mean(results[mode]),4), round(np.std(results[mode]),4)
        info[mode] = f'{mean}±{std}'
    info["avg_convergence_epochs"] = ace
    info_df = pd.DataFrame([info])
    if osp.exists(train_info_name):
        info_df.to_csv(train_info_name, mode="a", header=False, index=False,float_format='%.4f')
    else:
        info_df.to_csv(train_info_name, index=False, float_format='%.4f')


def test_pipeline(MODEL_NAME, DATASET_NAME, params, net_params, dirs, test_rep, datasetidx=None):
    # batching exception for Diffpool
    drop_last = True if MODEL_NAME == 'DiffPool' else False
    
    root_ckpt_dir, train_info_name, test_info_name = dirs
    results = {
        "model": MODEL_NAME,
        "train_repeat_time":params["train_repeat_time"],
        "test_repeat_time":[],
        "test_mae":[],
        "test_rmse":[],
        "test_r2":[],
    }
    df = pd.DataFrame(results)
    num_t = len(test_rep)
    dfs = [df]*num_t
    for i in range(params['seed']):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ck_name= os.path.join(root_ckpt_dir, f"{i}.pkl")
        model = gnn_model(MODEL_NAME, net_params).to(device)
        if not osp.exists(ck_name):
            print("************************** No model found! **************************")
        else:
            state = torch.load(ck_name)
            model.load_state_dict(state['model'])
        test_res = []
        for i in range(len(test_rep)):
            r = test_rep[i]
            test_dataset = PygPolymerDataset(name="prediction", root="data_pyg", set_name="test",task_name=DATASET_NAME,repeat_times=r,_use_concat_train = False)
            if datasetidx:
                test_dataset = test_dataset[datasetidx[0]:datasetidx[1]]
            dgl_test_dataset = DGLDatasetFromPyG(test_dataset)
            #print(len(dgl_test_dataset[0].ndata['feat']))

            '''
            if MODEL_NAME in ['GCN', 'GAT']:
                if net_params['self_loop']:
                    print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
                    dataset._add_self_loops()
            '''
            if MODEL_NAME in ['RingGNN', '3WLGNN']:
                # import train functions specific for WL-GNNs
                from class_baseline.utils import training_dense as training, validate_dense as validate
                test_loader = DataLoader(dgl_test_dataset, shuffle=False, collate_fn=collate_dense_gnn)

            else:
                # import train functions for all other GCNs
                from class_baseline.utils import training_sparse as training, validate_sparse as validate
                test_loader = DataLoader(dgl_test_dataset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=collate_gnn)

            test_perf = validate(model,test_loader,device)
            cur_df = dfs[i]
            new_results = {
                "model": MODEL_NAME,
                "train_repeat_time":params["train_repeat_time"],
                "test_repeat_time":r,
                "test_mae":test_perf["mae"],
                "test_rmse":test_perf["rmse"],
                "test_r2":test_perf["r2"],
            }
            new_df = pd.DataFrame([new_results])
            cur_df = pd.concat([cur_df, new_df], ignore_index=True)
            dfs[i] = cur_df

    for df in dfs:
        # Calculate mean and std, and format them as "mean±std".
        summary_cols = ["model", "train_repeat_time","test_repeat_time"]
        df_mean = df.groupby(summary_cols).mean().round(4)
        df_std = df.groupby(summary_cols).std().round(4)

        df_mean = df_mean.reset_index()
        df_std = df_std.reset_index()
        df_summary = df_mean[summary_cols].copy()
        for metric in ['r2','rmse','mae']:
            col_name = 'test_'+metric
            df_summary[col_name] = df_mean[col_name].astype(str) + "±" + df_std[col_name].astype(str)

        # Save and print the summary DataFrame.    
        if osp.exists(test_info_name):
            df_summary.to_csv(test_info_name, mode="a", header=False, index=False)
        else:
            df_summary.to_csv(test_info_name, index=False)
        print(df_summary)


def main(config_path, test_rep = None, _use_ck = False):
    with open(config_path) as f:
        config = json.load(f)

    # model, dataset, out_dir:
    MODEL_NAME = config['model']
    DATASET_NAME = config['dataset']
    out_dir = config['out_dir']
    # parameters
    params = config['params']

    # network parameters
    net_params = config['net_params']
    net_params['batch_size'] = params['batch_size']

    dataset = PygPolymerDataset(name="prediction", root="data_pyg", set_name="train",task_name=DATASET_NAME,repeat_times=1,_use_concat_train = False)

    if MODEL_NAME == 'DiffPool':
        # calculate assignment dimension: pool_ratio * largest graph's maximum
        # number of nodes  in the dataset
        num_nodes = [dataset.all[i][0].number_of_nodes() for i in range(len(dataset.all))]
        max_num_node = max(num_nodes)
        net_params['assign_dim'] = int(max_num_node * net_params['pool_ratio']) * net_params['batch_size']

    if MODEL_NAME == 'RingGNN':
        num_nodes = [dataset.all[i][0].number_of_nodes() for i in range(len(dataset.all))]
        net_params['avg_node_num'] = int(np.ceil(np.mean(num_nodes)))
    
    if params["use_concat_train"]:
        root_ckpt_dir = osp.join(out_dir, 'checkpoints', DATASET_NAME, MODEL_NAME,'concat',str(params["train_repeat_time"]))
    else:
        root_ckpt_dir = osp.join(out_dir, 'checkpoints', DATASET_NAME, MODEL_NAME,'single',str(params["train_repeat_time"]))
    if not osp.exists(root_ckpt_dir):
        os.makedirs(root_ckpt_dir)
    root_res_dir = osp.join(out_dir, 'res')
    if not osp.exists(root_res_dir):
        os.makedirs(root_res_dir)
    train_info_name = osp.join(root_res_dir,f'{DATASET_NAME}_train.csv')
    test_info_name = osp.join(root_res_dir,f'{DATASET_NAME}_test.csv')
    dirs = root_ckpt_dir, train_info_name, test_info_name
    
    if not _use_ck:
        net_params['total_param'] = view_model_param(MODEL_NAME, net_params)
        train_val_pipeline(MODEL_NAME, DATASET_NAME, params, net_params, dirs)
    else:
        if not test_rep:
            raise ValueError("Please define the test repeating times.")
        test_pipeline(MODEL_NAME, DATASET_NAME, params, net_params, dirs, test_rep)
    

if __name__ == "__main__":
    """
    USER CONTROLS
    """
    task = 'o2'
    model_name = 'gcn'
    config_path = f'class_baseline/configs/regression/{task}_{model_name}.json'
    test_rep = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    #test_rep = [20]
    main(config_path=config_path,test_rep = test_rep,_use_ck = True)
        