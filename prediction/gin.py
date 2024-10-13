import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from sklearn.metrics import mean_squared_error, mean_absolute_error,root_mean_squared_error,r2_score
import argparse
from tqdm.auto import tqdm
import os
import glob
import math
from opc import PygPolymerDataset
import os.path as osp
import numpy as np
from model import GNN
from dataset_produce import SmilesRepeat

#from dataset import TestDevPolymer

criterion = torch.nn.L1Loss()


def seed_torch(seed=0):
    print("Seed", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def training(model, device, loader, optimizer):
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

def add_arg(model_type, drop_ratio, num_layer, lr, batch_size):

    parser = argparse.ArgumentParser(
        description="GNN baselines for polymer property prediction"
    )
    parser.add_argument(
        "--device", type=int, default=0, help="which gpu to use if any (default: 0)"
    )
    parser.add_argument(
        "--gnn",
        type=str,
        default=model_type,
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

    args = parser.parse_args()
    
    return args


def validate(model, device, loader):
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


def save_results(model, device, loader, filename="result.csv"):
    from opc.utils.features import task_properties

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


def test(args, device, model, task_name, batch_size):
    # test on 10 different test sets
    rep = [1,2,3,4,5,6,7,8,9,10]
    test_res = []
    for r in rep:
        dataset = PygPolymerDataset(name="prediction", root="data_pyg", set_name = "test", repeat_times=r,task_name=task_name)
        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        test_perf = validate(model,device,test_loader)
        print_info('Test result of test set with {} repeating units'.format(r), test_perf)
        test_res.append(test_perf)

    print('Finished testing!')

    return (
        test_res
    )

def main_train(seed, args, device, model, optimizer, train_loader, valid_loader,check_name):

    # Training settings
    best_train, best_valid, best_params = None, None, None
    best_epoch = 0
    print("Start training...")

    for epoch in range(args.epochs):
        training(model, device, train_loader, optimizer)
        valid_perf = validate(model, device, valid_loader)
        if epoch == 0 or valid_perf['mae'] <  best_valid['mae']:
            train_perf = validate(model, device, train_loader)
            best_params = parameters_to_vector(model.parameters())

            best_valid = valid_perf
            best_train = train_perf

            best_epoch = epoch
        else:   
            # save checkpoints
            if epoch - best_epoch > args.patience:
                break
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(state, check_name)

    print('Finished training! Best validation results from epoch {}.'.format(best_epoch))
    print('Model saved as {}.'.format(check_name))
    print_info('train', best_train)
    print_info('valid', best_valid)

    vector_to_parameters(best_params, model.parameters())

    return (
        model,
        best_train,
        best_valid
    )

# obtain the oldest_checkpoint to replace
def oldest_checkpoint(path):
    names = glob.glob(os.path.join(path, "*.pt"))

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


def latest_checkpoint(path):
    names = glob.glob(os.path.join(path, "*.pt"))

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
def find_checkname(path, max_to_keep=1):

    checkpoints = glob.glob(os.path.join(path, "*.pt"))

    if max_to_keep and len(checkpoints) >= max_to_keep:
        checkpoint = oldest_checkpoint(path)
        os.remove(checkpoint)
    else:
        if not checkpoints:
            counter = 1
        else:
            checkpoint = latest_checkpoint(path)
            counter = int(checkpoint.rstrip(".pt").split("-")[-1]) + 1

        checkpoint = os.path.join(path, "model-%d.pt" % counter)
    return checkpoint

def print_info(set_name, perf):
    output_str = '{}\t\t'.format(set_name)
    for metric_name in perf.keys():
        output_str += '{}: {:<10.4f} \t'.format(metric_name, perf[metric_name])
    print(output_str)

def save_prediction(model, device, test_feature, smiles_list, target_list, out_file="out.json"):
    from opc.utils.features import task_properties
    import json
    task_properties = task_properties['prediction']
    task_count = {}
    pred_json = []
    loader = DataLoader(test_feature, batch_size=1, shuffle=False)
    for idx, batch in enumerate(loader):
        batch = batch.to(device)
        y_pred = model(batch)
        targets = target_list[idx]
        entry = {"SMILES": smiles_list[idx]}
        for i, target in enumerate(targets):
            pred_value = y_pred.detach().cpu().numpy()[0, task_properties.index(target)]
            entry[target] = float(pred_value)
            task_count[target] = task_count.get(target, 0) + 1
        pred_json.append(entry)

    with open(out_file, "w") as f:
        json.dump(pred_json, f, indent=4)
    
    task_weight = torch.sqrt(
        1 / torch.tensor(list(task_count.values()), dtype=torch.float32)
    )
    task_weight = task_weight / task_weight.sum() * len(task_weight)

    print(
        f"Predictions saved to {out_file}, to be evaluated with weights {task_weight} for each task."
    )


def hyper(repeat_times, model_type, task_name,drop_ratio,num_layer,lr,batch_size):
    #ck_path = osp.join('checkpoints/',str(task_name),model_type,str(repeat_times))
    # Training settings
    args = add_arg(model_type, drop_ratio, num_layer, lr, batch_size)

    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    ### automatic dataloading and splitting

    train_dataset = PygPolymerDataset(name="prediction", root="data_pyg", task_name=task_name,repeat_times=repeat_times,set_name="train")
    #split_idx = dataset.get_idx_split()
    valid_dataset = PygPolymerDataset(name="prediction", root="data_pyg", task_name=task_name,repeat_times=repeat_times,set_name="valid")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    if args.gnn == "gin":
        model = GNN(
            gnn_type="gin",
            num_task=train_dataset.num_tasks,
            num_layer=args.num_layer,
            emb_dim=args.emb_dim,
            drop_ratio=args.drop_ratio,
            virtual_node=False,
        ).to(device)
    elif args.gnn == "gin-virtual":
        model = GNN(
            gnn_type="gin",
            num_task=train_dataset.num_tasks,
            num_layer=args.num_layer,
            emb_dim=args.emb_dim,
            drop_ratio=args.drop_ratio,
            virtual_node=True,
        ).to(device)
    elif args.gnn == "gcn":
        model = GNN(
            gnn_type="gcn",
            num_task=train_dataset.num_tasks,
            num_layer=args.num_layer,
            emb_dim=args.emb_dim,
            drop_ratio=args.drop_ratio,
            virtual_node=False,
        ).to(device)
    elif args.gnn == "gcn-virtual":
        model = GNN(
            gnn_type="gcn",
            num_task=train_dataset.num_tasks,
            num_layer=args.num_layer,
            emb_dim=args.emb_dim,
            drop_ratio=args.drop_ratio,
            virtual_node=True,
        ).to(device)
    else:
        raise ValueError("Invalid GNN type")

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_train, best_valid, best_params = None, None, None
    best_epoch = 0
    #print("Start training...")
    for epoch in range(300):
        training(model, device, train_loader, optimizer)
        valid_perf = validate(model, device, valid_loader)
        if epoch == 0 or valid_perf['mae'] <  best_valid['mae']:
            train_perf = validate(model, device, train_loader)
            best_params = parameters_to_vector(model.parameters())
            best_valid = valid_perf
            #print(best_valid)
            #print(best_valid_lg)
            best_train = train_perf
            best_epoch = epoch
        elif epoch - best_epoch > args.patience:
            break

    vector_to_parameters(best_params, model.parameters())

    return np.mean(best_valid['r2'])

_use_ck = False
_use_param = True
if __name__ == "__main__":
    import os
    import pandas as pd
    import optuna
    param_csv_name = "hyperparameters_summary.csv"    
    parameters = {
        "model_type":[],
        "task":[],
        "drop_ratio":[],
        "num_layer":[],
        "lr":[],
        "batch_size":[],
    }
    df_p = pd.DataFrame(parameters)
    tasks = ['tg']
    for task_name in tasks:
        rep_times = [11]
        
        model_types = ["gin-virtual"]

        for model_type in model_types:
            set_up_param = False
            
            def objective(trial):
                drop_ratio=trial.suggest_float("drop_ratio",0.1,0.5,step=0.1)
                num_layer=trial.suggest_int("num_layer",5,10,step=1)
                lr=trial.suggest_float("learning_rate",1e-3,1e-2,log=True)
                batch_size=trial.suggest_int("batch_size",256,1024,step=256)
                return hyper(1, model_type,task_name,drop_ratio,num_layer,lr,batch_size)
            if not _use_param:
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
                        "model_type":model_type,
                        "task":task_name,
                        "drop_ratio":drop_ratio,
                        "num_layer":num_layer,
                        "lr":lr,
                        "batch_size":batch_size,
                    }   
                    # save best parameters
                    df_p= pd.concat([df_p, pd.DataFrame([parameters_new])], ignore_index=True)
                    if os.path.exists(param_csv_name):
                        df_p.to_csv(param_csv_name, mode="a", header=False, index=False)
                    else:
                        df_p.to_csv(param_csv_name, index=False)
                    set_up_param = True 
            # use params stored before     
            else:
                if os.path.exists(param_csv_name):
                    p = pd.read_csv(param_csv_name)
                    params = p[p['model_type']==model_type]
                    params = params[params['task']==task_name]
                    drop_ratio = float(params['drop_ratio'].iloc[0])
                    num_layer = int(params['num_layer'].iloc[0])
                    lr = float(params['lr'].iloc[0])
                    batch_size = int(params['batch_size'].iloc[0])
                else:
                    print("****************** No parameters stored before ******************")
            
            args = add_arg(model_type, drop_ratio, num_layer, lr, batch_size)
            device = (
                torch.device("cuda:" + str(args.device))
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            for r in rep_times:
                # dataset
                raw_file = 'data_pyg/prediction/{}/{}_raw_{}/{}_raw.csv'.format(task_name,task_name,r,task_name)

                if not os.path.exists(raw_file):
                    dataproduce = SmilesRepeat(r, task_name, root='data_pyg/prediction/')
                    dataproduce.repeat()
                print("************************** Work on {} task use {} model of {} layers repeat {} times **************************".format(task_name,model_type,num_layer,r))
                
                #split_idx = dataset.get_idx_split()
                train_dataset = PygPolymerDataset(name="prediction", root="data_pyg", set_name="train",task_name=task_name,repeat_times=r)
                valid_dataset = PygPolymerDataset(name="prediction", root="data_pyg", set_name="valid",task_name=task_name,repeat_times=r)

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                )
                valid_loader = DataLoader(
                    valid_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                )
                
                # initiate model
                if args.gnn == "gin":
                    model = GNN(
                        gnn_type="gin",
                        num_task=train_dataset.num_tasks,
                        num_layer=args.num_layer,
                        emb_dim=args.emb_dim,
                        drop_ratio=args.drop_ratio,
                        virtual_node=False,
                    ).to(device)
                elif args.gnn == "gin-virtual":
                    model = GNN(
                        gnn_type="gin",
                        num_task=train_dataset.num_tasks,
                        num_layer=args.num_layer,
                        emb_dim=args.emb_dim,
                        drop_ratio=args.drop_ratio,
                        virtual_node=True,
                    ).to(device)
                elif args.gnn == "gcn":
                    model = GNN(
                        gnn_type="gcn",
                        num_task=train_dataset.num_tasks,
                        num_layer=args.num_layer,
                        emb_dim=args.emb_dim,
                        drop_ratio=args.drop_ratio,
                        virtual_node=False,
                    ).to(device)
                elif args.gnn == "gcn-virtual":
                    model = GNN(
                        gnn_type="gcn",
                        num_task=train_dataset.num_tasks,
                        num_layer=args.num_layer,
                        emb_dim=args.emb_dim,
                        drop_ratio=args.drop_ratio,
                        virtual_node=True,
                    ).to(device)
                else:
                    raise ValueError("Invalid GNN type")
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                
                ck_path = osp.join('checkpoints/',str(task_name),model_type,str(r))
                if not osp.exists(ck_path):
                    os.makedirs(ck_path)
                ck_name = osp.join(ck_path,'model-1.pt')
                
                # checkpoint
                if _use_ck:
                    if not osp.exists(ck_name):
                        print("************************** No model found! **************************")
                    else:
                        state = torch.load(ck_name)
                        model.load_state_dict(state['model'])
                        optimizer.load_state_dict(state['optimizer'])

                
                                
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
                # store different test sets on one model
                df_save_name = "./gnn_res/tg/result_{}_{}.csv".format(model_type,r)
                dfs = [df] * 10
                for i in range(10):
                    seed_torch(i)
                    ck_name = osp.join(ck_path,'model-{}.pt'.format(i))
                    model, best_train, best_valid = main_train(i, args, device, model, optimizer, train_loader, valid_loader,ck_name)
                    test_res = test(args, device, model, task_name, batch_size)
                    for j in range(len(test_res)):
                        test_perf = test_res[j]
                        cur_df = dfs[j]
                        new_results = {
                            "model": model_type,
                            "train_repeat_time":r,
                            "test_repeat_time":j+1,
                            "task": task_name,
                            "test_mae":test_perf["mae"],
                            "test_rmse":test_perf["rmse"],
                            "test_lgmae":test_perf["lgmae"],
                            "test_r2":test_perf["r2"],
                        }
                        new_df = pd.DataFrame([new_results])
                        if os.path.exists(df_save_name):
                            new_df.to_csv(df_save_name, mode="a", header=False, index=False)
                        else:
                            new_df.to_csv(df_save_name, index=False)
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
                    res_csv_name = "./gnn_res/{}/result_{}_{}.csv".format(task_name,model_type,task_name)     
                    if os.path.exists(res_csv_name):
                        df_summary.to_csv(res_csv_name, mode="a", header=False, index=False)
                    else:
                        df_summary.to_csv(res_csv_name, index=False)
                    print(df_summary)
        