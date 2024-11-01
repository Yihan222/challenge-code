import argparse
from tqdm.auto import tqdm
import os
import os.path as osp
import numpy as np
import sys
import glob
import math
import pandas as pd
import optuna 

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from sklearn.metrics import mean_squared_error, mean_absolute_error,root_mean_squared_error,r2_score
from opc import PygPolymerDataset
from opc.utils.features import task_properties
from model import GNN
from dataset_produce import SmilesRepeat
from data_aug import csvcatg
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')

criterion = torch.nn.L1Loss()

def seed_torch(seed=0):
    print("Seed", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class MainGNN():
    def __init__(self, train_rep, task_name = 'tg', model_type = 'gin-virtual', param_csv_name = "hyperparameters_summary.csv", test_rep = [1],seed_num = 10,_use_ck = True, _use_param = True):
        self.r = train_rep
        self.task_name = task_name
        self.model_type = model_type
        self.param_csv_name = param_csv_name #path for saving the results of searching best hyperparameters
        self.test_rep = test_rep
        self._use_ck = _use_ck
        self._use_param = _use_param
        self.seed_num = seed_num
        # single dataset or merged dataset
        if isinstance(self.r,list):
            self._use_concat_train = True
            csvcatg(self.task_name,self.r)
            self.r = int('{}{}'.format(self.r[0],self.r[-1]))
            self.ck_path = osp.join('checkpoints/',str(self.task_name),self.model_type,'concat2',str(self.r))
            # path for saving results for each model
            self.df_save_name = "./res/gnn_res/{}/{}/concat2/repeat_{}.csv".format(self.task_name,self.model_type,self.r)
            # path for saving results for calculating std&mean for 10 models
            self.res_csv_name = "./res/gnn_res/{}/{}/concat2/result.csv".format(self.task_name,self.model_type)
        else:
            self._use_concat_train = False
            self.ck_path = osp.join('checkpoints/',str(self.task_name),self.model_type,str(self.r))
            self.df_save_name = "./res/gnn_res/{}/{}/repeat_{}.csv".format(self.task_name,self.model_type,self.r)
            self.res_csv_name = "./res/gnn_res/{}/{}/result.csv".format(self.task_name,self.model_type) 

        if not osp.exists(self.ck_path):
            if not self._use_ck:
                os.makedirs(self.ck_path)
            else:
                print("************ No such {} model exist! ************".format(self.model_type))
        print("************************** Work on {} task use {} model trained on {} repeat times **************************".format(self.task_name,self.model_type,self.r))

    def training(self, model, loader, optimizer,device):
        model.train()

        # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        for step, batch in enumerate(loader):
            batch = batch.to(device)

            if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
                del batch
                pass
            else:
                pred = model(batch)
                optimizer.zero_grad()
                is_valid = ~torch.isnan(batch.y)
                loss = criterion(
                    pred.to(torch.float32)[is_valid], batch.y.to(torch.float32)[is_valid]
                )
                loss.backward()
                optimizer.step()
                del batch,pred,loss
                torch.cuda.empty_cache()

    def validate(self, model, loader,device):
        model.eval()
        y_true = []
        y_pred = []

        for step, batch in enumerate(loader):
            batch = batch.to(device)
            if batch.x.shape[0] == 1:                
                pass
            else:
                with torch.no_grad():
                    pred = model(batch)
                y_true.append(batch.y.detach().cpu())
                y_pred.append(pred.detach().cpu())

        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()

        y_true, y_pred = y_true.reshape(-1), y_pred.reshape(-1)
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
            dataset = PygPolymerDataset(name="prediction", root="data_pyg", set_name = "test", repeat_times=r,task_name=self.task_name)
            if datasetidx:
                dataset = dataset[datasetidx[0]:datasetidx[1]]
            test_loader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.args.num_workers,
            )
            test_perf = self.validate(model,test_loader,device)
            self.print_info('Test result of test set with {} repeating units'.format(r), test_perf)
            test_res.append(test_perf)

        print('Finished testing!')

        return (
            test_res
        )

    def save_results(self, model, loader,device, filename="result.csv"):

        task_properties = task_properties["prediction"]
        model.eval()
        y_pred = []

        for step, batch in enumerate(loader):
            batch = batch.to(device)
            if batch.x.shape[0] == 1:
                pass
            else:
                with torch.no_grad():
                    pred = model(batch)
                y_pred.append(pred.detach().cpu())
        y_pred = torch.cat(y_pred, dim=0).numpy()
        df_pred = pd.DataFrame(y_pred, columns=task_properties)
        df_pred.to_csv(filename, index=False, header=False)
        print(f"Predictions saved to {filename}")

    # obtain the oldest_checkpoint to replace
    def oldest_checkpoint(self, path):
        names = glob.glob(osp.join(path, "*.pt"))

        if not names:
            return None

        oldest_counter = 10000000
        checkpoint_name = names[0]

        for name in names:
            counter = name.rstrip(".pt").split("-")[-1]

            if not counter.isdigit():
                continue
            else:
                counter = int(counter)

            if counter < oldest_counter:
                checkpoint_name = name
                oldest_counter = counter

        return checkpoint_name


    def latest_checkpoint(self, path):
        names = glob.glob(osp.join(path, "*.pt"))

        if not names:
            return None

        latest_counter = 0
        checkpoint_name = names[0]

        for name in names:
            counter = name.rstrip(".pt").split("-")[-1]

            if not counter.isdigit():
                continue
            else:
                counter = int(counter)

            if counter > latest_counter:
                checkpoint_name = name
                latest_counter = counter

        return checkpoint_name

    #find a suitable path for saving checkpoint
    def find_checkname(self, path, max_to_keep=1):

        checkpoints = glob.glob(osp.join(path, "*.pt"))

        if max_to_keep and len(checkpoints) >= max_to_keep:
            checkpoint = self.oldest_checkpoint(path)
            os.remove(checkpoint)
        else:
            if not checkpoints:
                counter = 1
            else:
                checkpoint = self.latest_checkpoint(path)
                counter = int(checkpoint.rstrip(".pt").split("-")[-1]) + 1

            checkpoint = osp.join(path, "model-%d.pt" % counter)
        return checkpoint

    def print_info(self, set_name, perf):
        output_str = '{}\t\t'.format(set_name)
        for metric_name in perf.keys():
            output_str += '{}: {:<10.4f} \t'.format(metric_name, perf[metric_name])
        print(output_str)
    
    # hyperparameter tuning
    def hyper(self,drop_ratio,num_layer,lr,batch_size):
        r = self.r
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        train_dataset = PygPolymerDataset(name="prediction", root="data_pyg", set_name="train",task_name=self.task_name,repeat_times=r,_use_concat_train = self._use_concat_train)
        valid_dataset = PygPolymerDataset(name="prediction", root="data_pyg", set_name="valid",task_name=self.task_name,repeat_times=r,_use_concat_train = self._use_concat_train)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        # initiate model
        if self.model_type == "gin":
            model = GNN(
                gnn_type="gin",
                repeat_time = train_dataset.repeat_times,
                num_task=train_dataset.num_tasks,
                num_layer=num_layer,
                drop_ratio=drop_ratio,
                virtual_node=False,
            ).to(device)
        elif self.model_type == "gin-virtual":
            model = GNN(
                gnn_type="gin",
                repeat_time = train_dataset.repeat_times,
                num_task=train_dataset.num_tasks,
                num_layer=num_layer,
                drop_ratio=drop_ratio,
                virtual_node=True,
            ).to(device)
        elif self.model_type == "gcn":
            model = GNN(
                gnn_type="gcn",
                repeat_time = train_dataset.repeat_times,
                num_task=train_dataset.num_tasks,
                num_layer=num_layer,
                drop_ratio=drop_ratio,
                virtual_node=False,
            ).to(device)
        elif self.model_type == "gcn-virtual":
            model = GNN(
                gnn_type="gcn",
                repeat_time = train_dataset.repeat_times,
                num_task=train_dataset.num_tasks,
                num_layer=num_layer,
                drop_ratio=drop_ratio,
                virtual_node=True,
            ).to(device)
        else:
            raise ValueError("Invalid GNN type")
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_train, best_valid, best_params = None, None, None
        best_epoch = 0
        #print("Start training...")
        for epoch in range(300):
            self.training(model, train_loader, optimizer, device)
            valid_perf = self.validate(model, valid_loader,device)
            if epoch == 0 or valid_perf['mae'] <  best_valid['mae']:
                train_perf = validate(model, train_loader,device)
                best_params = parameters_to_vector(model.parameters())
                best_valid = valid_perf
                best_train = train_perf
                best_epoch = epoch
            elif epoch - best_epoch > args.patience:
                break

        vector_to_parameters(best_params, model.parameters())

        return np.mean(best_valid['r2'])

    # obtain hyperparameters
    def getparam(self):
        set_up_param = False
        def objective(trial):
            drop_ratio=trial.suggest_float("drop_ratio",0.1,0.5,step=0.1)
            num_layer=trial.suggest_int("num_layer",5,10,step=1)
            lr=trial.suggest_float("learning_rate",1e-3,1e-2,log=True)
            batch_size=trial.suggest_int("batch_size",256,1024,step=256)
            return self.hyper(drop_ratio,num_layer,lr,batch_size)
        if not self._use_param:
            parameters = {
                "model_type":[],
                "task":[],
                "drop_ratio":[],
                "num_layer":[],
                "lr":[],
                "batch_size":[],
            }
            df_p = pd.DataFrame(parameters)
            # each kind of model&task only try search for parametrs once
            # use optuna to find the best hyperparameters
            if not set_up_param:
                study = optuna.create_study(direction="maximize")
                study.optimize(objective,n_trials=20)
                drop_ratio = study.best_params['drop_ratio']
                num_layer = study.best_params['num_layer']
                lr = study.best_params['learning_rate']
                batch_size = study.best_params['batch_size']
                parameters_new = {
                    "model_type":self.model_type,
                    "task":self.task_name,
                    "drop_ratio":drop_ratio,
                    "num_layer":num_layer,
                    "lr":lr,
                    "batch_size":batch_size,
                }   
                # save best parameters
                df_p= pd.concat([df_p, pd.DataFrame([parameters_new])], ignore_index=True)
                if osp.exists(self.param_csv_name):
                    df_p.to_csv(self.param_csv_name, mode="a", header=False, index=False)
                else:
                    df_p.to_csv(self.param_csv_name, index=False)
                set_up_param = True 
        # use config params     
        else:
            drop_ratio = float(config.get(self.model_type,'drop_ratio'))
            num_layer = int(config.get(self.model_type,'num_layer'))
            lr = float(config.get(self.model_type,'lr'))
            batch_size = int(config.get(self.model_type,'batch_size'))
        return drop_ratio, num_layer, lr, batch_size

    def add_arg(self):
        drop_ratio, num_layer, lr, batch_size = self.getparam()
        parser = argparse.ArgumentParser(
            description="GNN baselines for polymer property prediction"
        )
        parser.add_argument(
            "--gnn",
            type=str,
            default=self.model_type,
            help="GNN gin, gin-virtual, or gcn, or gcn-virtual",
        )
        parser.add_argument(
            "--drop_ratio", 
            type=float, 
            default=drop_ratio, 
            help="dropout ratio (default: 0.5) gcn-virtua: 0.1"
        )
        parser.add_argument(
            "--num_layer",
            type=int,
            default=num_layer,
            help="number of GNN message passing layers (default: 5)",
        )
        parser.add_argument(
            "--emb_dim",
            type=int,
            default=300,
            help="dimensionality of hidden units in GNNs (default: 300)",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=batch_size,
            help="input batch size for training (default: 1024)",
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=300,
            help="number of epochs to train (default: 300)",
        )
        parser.add_argument(
            '--lr', 
            type=float, 
            default=lr,
            help='Learning rate (gcn-virtual: 1e-3)')
        parser.add_argument(
            "--patience",
            type=int,
            default=100,
            help="number of epochs to stop training(default: 100)",
        )
        parser.add_argument(
            "--num_workers", 
            type=int, 
            default=0, 
            help="number of workers (default: 0)"
        )
        parser.add_argument(
            "--no_print", 
            action="store_true", 
            help="no print if activated (default: False)"
        )

        self.args = parser.parse_args()

    def model_seed(self,i):       
        train_dataset = PygPolymerDataset(name="prediction", root="data_pyg", set_name="train",task_name=self.task_name,repeat_times=self.r,_use_concat_train = self._use_concat_train)
        valid_dataset = PygPolymerDataset(name="prediction", root="data_pyg", set_name="valid",task_name=self.task_name,repeat_times=self.r,_use_concat_train = self._use_concat_train)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
        )
        
        # initiate model
        if self.args.gnn == "gin":
            model = GNN(
                gnn_type="gin",
                repeat_time = train_dataset.repeat_times,
                num_task=train_dataset.num_tasks,
                num_layer=self.args.num_layer,
                emb_dim=self.args.emb_dim,
                drop_ratio=self.args.drop_ratio,
                virtual_node=False,
            ).to(device)
        elif self.args.gnn == "gin-virtual":
            model = GNN(
                gnn_type="gin",
                test = self.test_rep[0],
                repeat_time = train_dataset.repeat_times,
                num_task=train_dataset.num_tasks,
                num_layer=self.args.num_layer,
                emb_dim=self.args.emb_dim,
                drop_ratio=self.args.drop_ratio,
                virtual_node=True,
            ).to(device)
        elif self.args.gnn == "gcn":
            model = GNN(
                gnn_type="gcn",
                repeat_time = train_dataset.repeat_times,
                num_task=train_dataset.num_tasks,
                num_layer=self.args.num_layer,
                emb_dim=self.args.emb_dim,
                drop_ratio=self.args.drop_ratio,
                virtual_node=False,
            ).to(device)
        elif self.args.gnn == "gcn-virtual":
            model = GNN(
                gnn_type="gcn",
                repeat_time = train_dataset.repeat_times,
                num_task=train_dataset.num_tasks,
                num_layer=self.args.num_layer,
                emb_dim=self.args.emb_dim,
                drop_ratio=self.args.drop_ratio,
                virtual_node=True,
            ).to(device)
        else:
            raise ValueError("Invalid GNN type")
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)

        ck_name = osp.join(self.ck_path,'model-{}.pt'.format(i))
                        
        
        best_train, best_valid, best_params = None, None, None
        if self._use_ck:
            # checkpoint
            if not osp.exists(ck_name):
                print("************************** No model found! **************************")
            else:
                state = torch.load(ck_name)
                model.load_state_dict(state['model'])
                optimizer.load_state_dict(state['optimizer'])
        else:
            # Training settings
            best_epoch = 0
            print("Start training...")

            for epoch in range(self.args.epochs):
                self.training(model, train_loader, optimizer, device)
                valid_perf = self.validate(model, valid_loader,device)
                if epoch == 0 or valid_perf['mae'] <  best_valid['mae']:
                    train_perf = self.validate(model, train_loader, device)
                    best_params = parameters_to_vector(model.parameters())

                    best_valid = valid_perf
                    best_train = train_perf

                    best_epoch = epoch
                else:   
                    # save checkpoints
                    if epoch - best_epoch > self.args.patience:
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
        self.add_arg()
        # testing
        if self._use_ck:
            # information reserved for testing results
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
            num_t = len(self.test_rep)
            dfs = [df] * num_t
        for i in range(self.seed_num):
            seed_torch(i)
            model,device,best_train,best_valid = self.model_seed(i)
            '''
            print(len(list(model.parameters())))
            for param in model.parameters():
                #print(param)
                print(param.size())
            sys.exit()
            '''
            if self._use_ck:
                # start testing
                print("Start testing...")
                test_res = self.test(model,device)
                for j in range(len(test_res)):
                    test_perf = test_res[j]
                    cur_df = dfs[j]
                    new_results = {
                        "model": self.model_type,
                        "train_repeat_time":self.r,
                        "test_repeat_time":self.test_rep[j],
                        "task": self.task_name,
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
        