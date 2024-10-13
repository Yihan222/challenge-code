import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error,r2_score
from tqdm.auto import tqdm
import os
import pandas as pd
import os.path as osp
import numpy as np
from model import GNN
from dataset_produce import SmilesRepeat
from diameter import PygPolymerDataset
tasks = ['O2']
results = {'task': [],
           'repeat_times':[],
           'max_diameter':[],
           'mode_diameter':[]
           }
for task_name in tasks:
    for repeat_times in [2,3,4,5,6,7,8,9,10]:
        raw_file = 'data_pyg/prediction/{}/{}_raw_{}/{}_raw.csv'.format(task_name,task_name,repeat_times,task_name)
        if not os.path.exists(raw_file):
            dataproduce = SmilesRepeat(repeat_times, task_name, root='data_pyg/prediction/')
            dataproduce.repeat()
        dataset = PygPolymerDataset(name="prediction", root="data_pyg", task_name=task_name,repeat_times=repeat_times)
        
        df=pd.DataFrame(results)
        diameter_list = dataset.process()

        counts = np.bincount(diameter_list)
        max_d = max(diameter_list)
        mode_d = np.argmax(counts)

        new_results = {'task': task_name,
                       'repeat_times':repeat_times,
                       'max_diameter':max_d,
                       'mode_diameter':mode_d
                       }
        df = pd.concat([df, pd.DataFrame([new_results])], ignore_index=True)
        res_csv_name = "diameter_summary.csv"          
        if os.path.exists(res_csv_name):
            df.to_csv(res_csv_name, mode="a", header=False, index=False)
        else:
            df.to_csv(res_csv_name, index=False)