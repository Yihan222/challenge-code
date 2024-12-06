"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from sklearn.metrics import mean_squared_error, mean_absolute_error,root_mean_squared_error,r2_score

criterion = torch.nn.L1Loss()

def print_info(set_name, perf):
    output_str = '{}\t\t'.format(set_name)
    for metric_name in perf.keys():
        output_str += '{}: {:<10.4f} \t'.format(metric_name, perf[metric_name])
    print(output_str)

def seed_torch(seed=0):
    print("Seed", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
"""
    For GCNs
"""
def training_sparse(model, data_loader, optimizer, device):
    model.train()
    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        try:
            batch_pos_enc = batch_graphs.ndata['pos_enc'].to(device)
            sign_flip = torch.rand(batch_pos_enc.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            batch_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
            pred = model.forward(batch_graphs, batch_x, batch_e, batch_pos_enc).unsqueeze(-1)
        except:
            pred = model.forward(batch_graphs, batch_x, batch_e).unsqueeze(-1)
        loss = model.loss(pred, batch_labels)
        loss.backward()
        optimizer.step()
        del batch_graphs,batch_labels,pred,loss
        torch.cuda.empty_cache()
   
def validate_sparse(model, data_loader,device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            try:
                batch_pos_enc = batch_graphs.ndata['pos_enc'].to(device)
                pred = model.forward(batch_graphs, batch_x, batch_e, batch_pos_enc)
            except:
                pred = model.forward(batch_graphs, batch_x, batch_e)
            y_true.append(batch_labels)
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    y_true, y_pred = y_true.reshape(-1), y_pred.reshape(-1)
    mae = mean_absolute_error(y_true,y_pred)
    rmse = root_mean_squared_error(y_true,y_pred)
    r2 = r2_score(y_true, y_pred)

    perf ={'mae': mae, 'rmse':rmse, 'r2': r2}
    return perf  
       

"""
    For WL-GNNs
"""
def validate_dense(model, optimizer, device, data_loader, epoch, batch_size):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    optimizer.zero_grad()
    for iter, (x_with_node_feat, labels) in enumerate(data_loader):
        x_with_node_feat = x_with_node_feat.to(device)
        labels = labels.to(device)
        
        scores = model.forward(x_with_node_feat)
        loss = model.loss(scores, labels) 
        loss.backward()
        
        if not (iter%batch_size):
            optimizer.step()
            optimizer.zero_grad()
            
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(scores, labels)
        nb_data += labels.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_acc /= nb_data
    
    return epoch_loss, epoch_train_acc, optimizer

def validate_dense(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (x_with_node_feat, labels) in enumerate(data_loader):
            x_with_node_feat = x_with_node_feat.to(device)
            labels = labels.to(device)
            
            scores = model.forward(x_with_node_feat)
            loss = model.loss(scores, labels) 
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(scores, labels)
            nb_data += labels.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data
        
    return epoch_test_loss, epoch_test_acc




def check_patience(all_losses, best_loss, best_epoch, curr_loss, curr_epoch, counter):
    if curr_loss < best_loss:
        counter = 0
        best_loss = curr_loss
        best_epoch = curr_epoch
    else:
        counter += 1
    return best_loss, best_epoch, counter