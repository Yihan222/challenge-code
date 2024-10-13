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

        try:
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
        except:
            del batch
            pass



def validate(model, device, loader):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch)
        try:
            y_true.append(batch.y.detach().cpu())
            y_pred.append(pred.detach().cpu())
        except:
            continue
    try:
        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()

        y_true, y_pred = y_true.reshape(-1), y_pred.reshape(-1)
    except:
        pass
    #rmse = root_mean_squared_error(y_true, y_pred)
    #mse = mean_squared_error(y_true, y_pred)
    try:
        mae = mean_absolute_error(y_true,y_pred)
    except:
        mae = np.nan
    y1,y2 = [],[]
    for i in range(len(y_pred)):
        if y_pred[i] != np.nan and y_true[i] and y_pred[i]/y_true[i]>0:
            y1.append(math.log(abs(y_pred[i])))
            y2.append(math.log(abs(y_true[i])))
    try:
        lgmae = mean_absolute_error(y1,y2)
    except:
        lgmae = np.nan
    try:
        r2 = r2_score(y_true, y_pred)
    except:
        r2=np.nan
    
    perf ={'mae': mae, 'lgmae':lgmae,'r2': r2}

    return perf


def save_results(model, device, loader, filename, column):
    from opc.utils.features import task_properties

    task_properties = task_properties["prediction"]
    model.eval()

    y_pred = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch)
        if not pred.detach().cpu():
            y_pred.append(np.nan)
        else:
            y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim=0).numpy()
    df1 = pd.read_csv(filename)

    df1[column] = y_pred
    
    df1.to_csv(filename, mode="w", index=False)

    print(f"Predictions saved to {filename}")

def test(repeat_times, model_type, task_name):
    ck_path = osp.join('checkpoints/',str(task_name),model_type,str(repeat_times))
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
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    dataset = PygPolymerDataset(name="prediction", root="data_pyg", task_name=task_name,repeat_times=1)
    #split_idx = dataset.get_idx_split()
    test_idx = pd.read_csv(osp.join('./data_pyg/prediction/{}/{}_raw_1/split/random/test.csv.gz'.format(task_name,task_name)), compression='gzip', header = None).values.T[0]

    test_loader = DataLoader(
        dataset[torch.tensor(test_idx, dtype = torch.long)],
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )
    if args.gnn == "gin":
        model = GNN(
            gnn_type="gin",
            num_task=dataset.num_tasks,
            virtual_node=False,
        ).to(device)
    elif args.gnn == "gin-virtual":
        model = GNN(
            gnn_type="gin",
            num_task=dataset.num_tasks,
            virtual_node=True,
        ).to(device)
    elif args.gnn == "gcn":
        model = GNN(
            gnn_type="gcn",
            num_task=dataset.num_tasks,
            virtual_node=False,
        ).to(device)
    elif args.gnn == "gcn-virtual":
        model = GNN(
            gnn_type="gcn",
            num_task=dataset.num_tasks,
            virtual_node=True,
        ).to(device)
    else:
        raise ValueError("Invalid GNN type")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    ck_files = os.listdir(ck_path)
    test_res = []
    for f in ck_files:
        ck_file = ck_path+'/'+f
        state = torch.load(ck_file)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        save_results(model, device, test_loader, filename="results_{}.csv".format(task_name),column='{}_{}'.format(task_name,repeat_times))
    print('Finished testing!')



def main_train(seed, repeat_times, model_type, task_name):
    ck_path = osp.join('checkpoints/',str(task_name),model_type,str(repeat_times))
    if not os.path.exists(ck_path):
        os.makedirs(ck_path)
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
        default='gin-virtual',
        help="GNN gin, gin-virtual, or gcn, or gcn-virtual",
    )
    parser.add_argument(
        "--drop_ratio", 
        type=float, 
        default=0.1, 
        help="dropout ratio (default: 0.5) gcn-virtua: 0.1"
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
        default=32,
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
        default=0.001,
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

    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    ### automatic dataloading and splitting

    dataset = PygPolymerDataset(name="prediction", root="data_pyg", task_name=task_name,repeat_times=repeat_times)
    #split_idx = dataset.get_idx_split()
    train_idx = pd.read_csv(osp.join('./data_pyg/prediction/{}/{}_raw_1/split/random/train.csv.gz'.format(task_name,task_name)), compression='gzip', header = None).values.T[0]
    valid_idx = pd.read_csv(osp.join('./data_pyg/prediction/{}/{}_raw_1/split/random/valid.csv.gz'.format(task_name,task_name)), compression='gzip', header = None).values.T[0]
    train_loader = DataLoader(
        dataset[torch.tensor(train_idx, dtype = torch.long)],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        dataset[torch.tensor(valid_idx, dtype = torch.long)],
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
    check_name = ''
    for epoch in range(args.epochs):
        training(model, device, train_loader, optimizer)
        valid_perf = validate(model, device, valid_loader)
        if epoch == 0 or valid_perf['mae'] <  best_valid['mae']:
            train_perf = validate(model, device, train_loader)
            best_params = parameters_to_vector(model.parameters())

            best_valid = valid_perf
            best_train = train_perf

            best_epoch = epoch
            # save checkpoints
            if epoch > 30:
                if not check_name:
                    check_name = find_checkname(ck_path, 1)
                    print("Saving checkpoint: %s" % check_name)
                state = {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict()
                }
                torch.save(state, check_name)
            '''
            if not args.no_print:
                print_info('Train', train_perf)
                print_info('Valid', valid_perf)
                print_info('Test', test_perf)
                '''
        else:
            '''
            if not args.no_print:
                print_info('Train', train_perf)
                print_info('Valid', valid_perf)
            '''    
            if epoch - best_epoch > args.patience:
                break

    print('Finished training! Best validation results from epoch {}.'.format(best_epoch))
    print('Model saved as {}.'.format(check_name))
    print_info('train', best_train)
    print_info('valid', best_valid)

    vector_to_parameters(best_params, model.parameters())

    return (
        best_train,
        best_valid,
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
        rep_times = [5]
        
        model_types = ["gin-virtual"]

        for model_type in model_types:
            set_up_param = False

            # use params to start training
            for r in rep_times:

                raw_file = 'data_pyg/prediction/{}/{}_raw_{}/{}_raw.csv'.format(task_name,task_name,r,task_name)
                if not os.path.exists(raw_file):
                    dataproduce = SmilesRepeat(r, task_name, root='data_pyg/prediction/')
                    dataproduce.repeat()
                print("************************** Work on {} task use {} model of repeat {} times **************************".format(task_name,model_type,r))
                for i in range(1):
                    seed_torch(i)
                    main_train(i,r,model_type,task_name)

                # use checkpoints to evaluate
                print("************************** Test on {} task use {} model repeat {} **************************".format(task_name,model_type,r))
                test(r,model_type,task_name)
    