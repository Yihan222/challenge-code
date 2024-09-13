import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error,r2_score
import argparse
from tqdm.auto import tqdm

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
    #rmse = root_mean_squared_error(y_true, y_pred)
    #mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    y1,y2 = [],[]
    for i in range(len(y_pred)):
        if y_pred[i] != np.nan and y_pred[i] > 0:
            y1.append(y_pred[i])
            y2.append(y_true[i])
    if y1:
        lgmae = mean_absolute_error(np.log(y2),np.log(y1))
    else:
        lgmae = np.nan
    r2 = r2_score(y_true, y_pred)
    
    perf ={'mae': mae, 'lgmae':lgmae,'r2': r2}

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


def main(seed, repeat_times, model_type, task_name):
    ck_path = osp.join('checkpoints/',str(task_name),model_type,str(repeat_times))
    # Training settings
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
        "--drop_ratio", type=float, default=0.1, help="dropout ratio (default: 0.5) gcn-virtua: 0.1"
    )
    parser.add_argument(
        "--num_layer",
        type=int,
        default=5,
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
        default=512,
        help="input batch size for training (default: 1024)",
    )
    parser.add_argument('--wdecay', default=1e-5, type=float,
                        help='weight decay')
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="number of epochs to train (default: 300)",
    )
    if model_type=='gcn-virtual' or model_type=='gcn':
        parser.add_argument('--lr', '--learning-rate', type=float, default=1e-3,
                            help='Learning rate (gcn-virtual: 1e-3)')
        parser.add_argument(
            "--patience",
            type=int,
            default=100,
            help="number of epochs to stop training(gcn-virtual: 100)",
        )
    if model_type=='gin-virtual' or model_type=='gin':
        parser.add_argument('--lr', '--learning-rate', type=float, default=1e-2,
                            help='Learning rate (default: 1e-2) ')
        parser.add_argument(
            "--patience",
            type=int,
            default=50,
            help="number of epochs to stop training(default: 50)",
        )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="number of workers (default: 0)"
    )
    parser.add_argument(
        "--no_print", action="store_true", help="no print if activated (default: False)"
    )

    args = parser.parse_args()

    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    ### automatic dataloading and splitting

    dataset = PygPolymerDataset(name="prediction", root="data_pyg", task_name=task_name,repeat_times=repeat_times)
    split_idx = dataset.get_idx_split()
    #train_weight = dataset.get_task_weight(split_idx["train"])
    #valid_weight = dataset.get_task_weight(split_idx["valid"])

    ### automatic evaluator. takes dataset name as input
    #evaluator = Evaluator("prediction")

    train_loader = DataLoader(
        dataset[split_idx["train"]],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        dataset[split_idx["valid"]],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        dataset[split_idx["test"]],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    if args.gnn == "gin":
        model = GNN(
            gnn_type="gin",
            num_task=dataset.num_tasks,
            num_layer=args.num_layer,
            emb_dim=args.emb_dim,
            drop_ratio=args.drop_ratio,
            virtual_node=False,
        ).to(device)
    elif args.gnn == "gin-virtual":
        model = GNN(
            gnn_type="gin",
            num_task=dataset.num_tasks,
            num_layer=args.num_layer,
            emb_dim=args.emb_dim,
            drop_ratio=args.drop_ratio,
            virtual_node=True,
        ).to(device)
    elif args.gnn == "gcn":
        model = GNN(
            gnn_type="gcn",
            num_task=dataset.num_tasks,
            num_layer=args.num_layer,
            emb_dim=args.emb_dim,
            drop_ratio=args.drop_ratio,
            virtual_node=False,
        ).to(device)
    elif args.gnn == "gcn-virtual":
        model = GNN(
            gnn_type="gcn",
            num_task=dataset.num_tasks,
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
    print("Start training...")
    for epoch in range(args.epochs):
        training(model, device, train_loader, optimizer)
        valid_perf = validate(model, device, valid_loader)
        if epoch == 0 or valid_perf['mae'] <  best_valid['mae']:
            train_perf = validate(model, device, train_loader)
            best_params = parameters_to_vector(model.parameters())
            # save checkpoints
            if not os.path.exists(ck_path):
                os.makedirs(ck_path)
            if epoch > 30:
                save_checkpoint(epoch, model, optimizer, ck_path, 10)

            best_valid = valid_perf
            #print(best_valid)
            #print(best_valid_lg)
            best_train = train_perf

            best_epoch = epoch
            test_perf = validate(model,device,test_loader)
            if not args.no_print:
                print_info('Train', train_perf)
                print_info('Valid', valid_perf)
                print_info('Test', test_perf)
        else:
            if not args.no_print:
                print_info('Train', train_perf)
                print_info('Valid', valid_perf)
                
            if epoch - best_epoch > args.patience:
                break

    print('Finished training! Best validation results from epoch {}.'.format(best_epoch))
    print_info('train', best_train)
    print_info('valid', best_valid)
    print_info('test', test_perf)

    vector_to_parameters(best_params, model.parameters())

    #test_dev = TestDevPolymer(name="prediction")
    #test_feature, smiles_list, target_list = test_dev.prepare_feature(
    #    transform="PyG"
    #)
    #path = 'out_cached'
    #ifEx = os.path.exists(path)
    #if not ifEx:
    #    os.makedirs(path)
    #save_prediction(model, device, test_feature, smiles_list, target_list, out_file=f"out_cached/out-{args.gnn}-{seed}.json")

    return (
        best_train,
        best_valid,
        test_perf
    )

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
import os
import glob
import torch


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


def save(state, path, max_to_keep=10):
    checkpoints = glob.glob(os.path.join(path, "*.pt"))

    if max_to_keep and len(checkpoints) >= max_to_keep:
        checkpoint = oldest_checkpoint(path)
        os.remove(checkpoint)

    if not checkpoints:
        counter = 1
    else:
        checkpoint = latest_checkpoint(path)
        counter = int(checkpoint.rstrip(".pt").split("-")[-1]) + 1

    checkpoint = os.path.join(path, "model-%d.pt" % counter)
    print("Saving checkpoint: %s" % checkpoint)
    torch.save(state, checkpoint)

def save_checkpoint(epoch, model, optimizer, path, max_ck):
    state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
    }
    save(state, path, max_ck)


if __name__ == "__main__":
    import os
    import pandas as pd
    tasks = ['He','H2','CH4','mt','density','tg']
    for t_name in tasks:
        rep_times = [0,1,2,3,4,5,6,7,8,9,10]
        model_types = ["gin-virtual","gcn-virtual"]
        task_name = t_name
        for r in rep_times:
            raw_file = 'data_pyg/prediction/{}/{}_raw_{}/{}_raw.csv'.format(task_name,task_name,r,task_name)
            if not os.path.exists(raw_file):
                dataproduce = SmilesRepeat(r, task_name, root='data_pyg/prediction/')
                dataproduce.repeat()
            for model_type in model_types:
                for t in range(5):
                    print("************************** Work on {} task use {} model repeat {} for {} time **************************".format(task_name,model_type,r,t+1))
                    results = {
                        "model": [],
                        "repeat_time":[],
                        "task": [],
                        "train_mae": [],
                        "train_lgmae":[],
                        "train_r2":[],
                        "valid_mae":[],
                        "valid_lgmae":[],
                        "valid_r2": [],
                        "test_mae":[],
                        "test_lgmae":[],
                        "test_r2":[],
                    }
                    df = pd.DataFrame(results)
                    symbol = False
                    for i in range(5):
                        seed_torch(i)
                        train_perf, valid_perf,test_perf= main(i,r,model_type,task_name)

                        new_results = {
                            "model": model_type,
                            "repeat_time":r,
                            "task": task_name,
                            "train_mae": train_perf["mae"],
                            "train_lgmae":train_perf["lgmae"],
                            "train_r2":train_perf["r2"],
                            "valid_mae":valid_perf["mae"],
                            "valid_lgmae":valid_perf["lgmae"],
                            "valid_r2": valid_perf["r2"],
                            "test_mae":test_perf["mae"],
                            "test_lgmae":test_perf["lgmae"],
                            "test_r2":test_perf["r2"],
                        }
                        df = pd.concat([df, pd.DataFrame([new_results])], ignore_index=True)
                        
                        
                    # Calculate mean and std, and format them as "mean±std".
                    summary_cols = ["model", "repeat_time", "task"]
                    df_mean = df.groupby(summary_cols).mean().round(4)
                    df_std = df.groupby(summary_cols).std().round(4)

                    df_mean = df_mean.reset_index()
                    df_std = df_std.reset_index()
                    df_summary = df_mean[summary_cols].copy()
                    # Format 'train', 'valid' columns as "mean±std".
                    for name in ['train','test','valid']:
                        for metric in ['r2','mae','lgmae']:
                            col_name = name+'_'+metric
                            df_summary[col_name] = df_mean[col_name].astype(str) + "±" + df_std[col_name].astype(str)

                    # Save and print the summary DataFrame.
                    res_csv_name = "result_summary_"+task_name+".csv"          
                    if os.path.exists(res_csv_name):
                        df_summary.to_csv(res_csv_name, mode="a", header=False, index=False)
                    else:
                        df_summary.to_csv(res_csv_name, index=False)
                    print(df_summary)
