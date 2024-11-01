import pickle
import pandas as pd
import sys
import os
import os.path as osp
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error,root_mean_squared_error,r2_score
import torch
import torch.optim as optim
import math
from transformer import Transformer
from tsfmdataset import TransPolyDataset, TransPolyDataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')

criterion = torch.nn.L1Loss()
lr = float(config.get('transformer','lr'))
epochs = int(config.get('transformer','epochs'))
batch_size = int(config.get('transformer','batch_size'))
max_length_dic = config.get('data','max_length_dic')
patience = int(config.get('transformer','patience'))

def seed_torch(seed=0):
    print("Seed", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class MainTransformer():
    def __init__(self, train_rep, task_name = 'tg', test_rep = [1], seed_num = 10, _use_ck = False):
        self.task = task_name
        self.model_type = 'transformer'
        if isinstance(train_rep,list):
            self.train_rep = train_rep
        else:
            self.train_rep = [train_rep]
        if isinstance(test_rep,list):
            self.test_rep = test_rep
        else:
            self.test_rep = [test_rep]
        self._use_ck = _use_ck
        self.data_root = "./data_pyg/prediction/{}".format(self.task)
        self.seed_num = seed_num
        if isinstance(train_rep,list):
            self._use_concat_train = True
            self.r = int('{}{}'.format(train_rep[0],train_rep[-1]))
            self.ck_path = './checkpoints/tg/transformer/concat/{}'.format(self.r)
            self.df_save_name = "./res/tran_res/concat/repeat_{}.csv".format(self.r)
            self.res_csv_name = "./res/tran_res/concat/result.csv"
        else:
            self._use_concat_train = False            
            self.r = train_rep
            self.ck_path = './checkpoints/tg/transformer/{}'.format(self.r)
            self.df_save_name = "./res/tran_res/repeat_{}.csv".format(train_rep)
            self.res_csv_name = "./res/tran_res/result.csv"
        
        if self._use_ck:
            if not osp.exists(self.ck_path):
                print("************ No such transformer exist! ************")
                sys.exit()
        else:
            if not osp.exists(self.ck_path):
                os.makedirs(self.ck_path) 

    def training(self, model, loader, optimizer, device):
        model.train()

        # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        for step, (batch_x,batch_y) in enumerate(loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            if batch_x.shape[0] == 0:
                del batch_x
                del batch_y
                pass
            else:
                pred = model(batch_x).squeeze()
                optimizer.zero_grad()
                is_valid = ~torch.isnan(batch_y)
                loss = criterion(
                    pred.to(torch.float32)[is_valid], batch_y.to(torch.float32)[is_valid]
                )
                loss.backward()
                optimizer.step()
                del batch_x,batch_y,pred,loss
            torch.cuda.empty_cache()


    def validate(self, model, loader, device):
        model.eval()
        y_true = []
        y_pred = []

        for step, (batch_x,batch_y) in enumerate(loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            if batch_x.shape[0] == 0: 
                pass
            else:
                with torch.no_grad():
                    pred = model(batch_x).squeeze()
                y_true.append(batch_y.detach().cpu())
                y_pred.append(pred.detach().cpu())
            torch.cuda.empty_cache()
        if len(batch_x) > 1:
            y_true = torch.cat(y_true, dim=0).numpy().reshape(-1)
            y_pred = torch.cat(y_pred, dim=0).numpy().reshape(-1)
        else:
            y_true, y_pred = torch.tensor(y_true).numpy().reshape(-1), torch.tensor(y_pred).numpy().reshape(-1)
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
    
        perf ={'mae': mae, 'rmse':rmse,'lgmae':lgmae,'r2': r2}
        return perf

    def test(self, model, device, datasetidx=None):
        # test on different test sets
        test_res = []
        for r in self.test_rep:
            test_dataset  = TransPolyDataset(self.data_root,[r],'test')
            if datasetidx:
                test_dataset = test_dataset[datasetidx[0]:datasetidx[1]]
            test_loader = TransPolyDataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
            )
            test_perf = self.validate(model,test_loader, device)
            self.print_info('Test result of test set with {} repeating units'.format(r), test_perf)
            test_res.append(test_perf)

        print('Finished testing!')
        #print(test_res)
        return (
            test_res
        )

    def print_info(self, set_name, perf):
        output_str = '{}\t\t'.format(set_name)
        for metric_name in perf.keys():
            output_str += '{}: {:<10.4f} \t'.format(metric_name, perf[metric_name])
        print(output_str)    

    def model_seed(self,i,train_loader=None,valid_loader=None):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = Transformer().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)            
        #ck_name represents seed
        #default lr=0.0001,hidden_size=256,loss function=l1, hidden layers=6
        ck_name = osp.join(self.ck_path,'{}.pt'.format(i))
        best_train, best_valid, best_params = None, None, None
        if self._use_ck:
            # checkpoint
            if not osp.exists(ck_name):
                print("************************** No model found! **************************")
                sys.exit()            
            else:
                state = torch.load(ck_name)
                model.load_state_dict(state['model'])
                optimizer.load_state_dict(state['optimizer'])
        else:
            if not train_loader or not valid_loader:
                print("************************** Please load training/validating dataset! **************************")
                sys.exit()             
            # Training settings
            best_epoch = 0
            print("Start training...")

            for epoch in range(epochs):
                self.training(model, train_loader, optimizer, device)
                valid_perf = self.validate(model, valid_loader, device)
                if epoch == 0 or valid_perf['mae'] <  best_valid['mae']:
                    train_perf = self.validate(model, train_loader, device)
                    best_params = parameters_to_vector(model.parameters())

                    best_valid = valid_perf
                    best_train = train_perf

                    best_epoch = epoch
                else:   
                    # save checkpoints
                    if epoch - best_epoch > patience:
                        break
            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(state, ck_name)

            print('Finished training of {}th model of {} repeat times! Best validation results from epoch {}.'.format(i,self.r,best_epoch))
            print('Model saved as {}.'.format(ck_name))
            self.print_info('train', best_train)
            self.print_info('valid', best_valid)

            vector_to_parameters(best_params, model.parameters())
        
        return (
            model,
            device,
            best_train,
            best_valid
        )
    
            
    def main(self): 
        # train a new model
        if not self._use_ck:
            # load train/val datasets
            train_dataset = TransPolyDataset(self.data_root,self.train_rep,'train')
            valid_dataset = TransPolyDataset(self.data_root,self.train_rep,'valid')

            train_loader = TransPolyDataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
            )
            valid_loader = TransPolyDataLoader(
                valid_dataset,
                batch_size=batch_size,
                shuffle=False,
            )
        # test a existed model
        else:  
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
            if not self._use_ck:
                model,device,best_train,best_valid = self.model_seed(i, train_loader, valid_loader)
            else:
                model,device,_,_ = self.model_seed(i)
                print("Start testing...")
                test_res = self.test(model,device)
                for j in range(len(test_res)):
                    test_perf = test_res[j]
                    cur_df = dfs[j]
                    new_results = {
                        "model": 'transformer',
                        "train_repeat_time":self.r,
                        "test_repeat_time":self.test_rep[j],
                        "task": self.task,
                        "test_mae":test_perf["mae"],
                        "test_rmse":test_perf["rmse"],
                        "test_lgmae":test_perf["lgmae"],
                        "test_r2":test_perf["r2"],
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
        if not self._use_ck:
            print('Finished training for all {} models of {} repeat times!'.format(self.seed_num, self.r))
        else:
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
            print('Finished testing for all {} models of {} repeat times!'.format(self.seed_num, self.r))

        
if __name__ == "__main__":
    pass