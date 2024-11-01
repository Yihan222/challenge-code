import os
import ast
import json
import copy
import os.path as osp
import pandas as pd
import networkx as nx
import numpy as np

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import to_networkx


from opc.utils.mol import smiles2graph

from tqdm.auto import tqdm




class PygPolymerDataset(InMemoryDataset):
    def __init__(
        self, name="prediction", root="data_pyg",repeat_times = 1, task_name = 'tg', set_name='train', _use_concat_train = False, transform=None, pre_transform=None
    ):
        """
        - name (str): name of the dataset == prediction or generation
        - root (str): root directory to store the dataset folder
        - transform, pre_transform (optional): transform/pre-transform graph objects
        """
        #print('testing')
        self.name = name
        self.repeat_times = repeat_times
        self.task_properties = [task_name]
        self.set_name = set_name
        self.task_type = self.task_properties[0]
        if _use_concat_train:
            self.root = osp.join(root, name, self.task_type, 'concat2', str(self.repeat_times),self.set_name)
        else:
            raw_root = osp.join(root, name, self.task_type, str(self.repeat_times))

            self.root = osp.join(raw_root,self.set_name)



        if self.task_properties is None:
            self.num_tasks = None
            self.eval_metric = "jaccard"
        else:
            self.num_tasks = len(self.task_properties)
            self.eval_metric = "wmae"

        super(PygPolymerDataset, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
  
  

    @property
    def processed_file_names(self):
        return ["data_dev_processed.pt"]

    def process(self):
        raw_file_name = "{}.csv".format(self.set_name)
        csv_file = osp.join(self.root,raw_file_name)
        data_df = pd.read_csv(csv_file)

        pyg_graph_list = []
        networkx_list = []
        for idx, row in data_df.iterrows():
            smiles = row["SMILES"]
            graph = smiles2graph(smiles, add_fp=self.name == "prediction")

            g = Data()
            g.num_nodes = graph["num_nodes"]
            g.edge_index = torch.from_numpy(graph["edge_index"])

            del graph["num_nodes"]
            del graph["edge_index"]

            if graph["edge_feat"] is not None:
                g.edge_attr = torch.from_numpy(graph["edge_feat"])
                del graph["edge_feat"]

            if graph["node_feat"] is not None:
                g.x = torch.from_numpy(graph["node_feat"])
                del graph["node_feat"]

            try:
                g.fp = torch.tensor(graph["fp"], dtype=torch.int8).view(1, -1)
                del graph["fp"]
            except:
                pass

            if self.task_properties is not None:
                y = []
                for task in self.task_properties:
                    y.append(float(row[task]))
                g.y = torch.tensor(y, dtype=torch.float32).view(1, -1)
            else:
                g.y = torch.tensor(
                    ast.literal_eval(row["labels"]), dtype=torch.float32
                ).view(1, -1)
            networkX_graph = to_networkx(g, node_attrs=["x"], edge_attrs=["edge_attr"])
            networkx_list.append(nx.diameter(networkX_graph))
            pyg_graph_list.append(g)
        #print(networkx_list)
        pyg_graph_list = (
            pyg_graph_list
            if self.pre_transform is None
            else self.pre_transform(pyg_graph_list)
        )
        print("Saving...")
        torch.save(self.collate(pyg_graph_list), self.processed_paths[0])

        return networkx_list

if __name__ == "__main__":
    tasks = ['tg']
    results = {'task': [],
            'repeat_times':[],
            'max_diameter':[],
            'mode_diameter':[]
            }
    test_rep = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    for task_name in tasks:
        for repeat_times in test_rep:
            dataset = PygPolymerDataset(repeat_times=repeat_times, set_name='test')
            
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
            res_csv_name = "test_diameter_summary.csv"          
            if os.path.exists(res_csv_name):
                df.to_csv(res_csv_name, mode="a", header=False, index=False)
            else:
                df.to_csv(res_csv_name, index=False)