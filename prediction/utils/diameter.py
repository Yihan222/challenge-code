import os
import ast
import json
import os.path as osp
import pandas as pd
import networkx as nx


import torch
import copy
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import to_networkx

from sklearn.model_selection import train_test_split

from opc.utils.mol import smiles2graph
from opc.utils.split import scaffold_split, similarity_split
#from opc.utils.features import task_properties


class PygPolymerDataset(InMemoryDataset):
    def __init__(
        self, name="prediction", root="data_pyg",repeat_times = 1, task_name = 'O2',transform=None, pre_transform=None
    ):
        """
        - name (str): name of the dataset == prediction or generation
        - root (str): root directory to store the dataset folder
        - transform, pre_transform (optional): transform/pre-transform graph objects
        """

        self.name = name
        self.repeat_times = repeat_times
        self.task_properties = [task_name]
        self.task_type = self.task_properties[0]
        self.fileroot = "{}_raw_{}".format(self.task_type,self.repeat_times)
        self.root = osp.join(root, name, self.task_type, self.fileroot)

    def process(self):
        raw_file_name = "{}_raw.csv".format(self.task_type)
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
            addition_prop = copy.deepcopy(graph)
            for key in addition_prop.keys():
                g[key] = torch.tensor(graph[key])
                del graph[key]
            networkX_graph = to_networkx(g, node_attrs=["x"], edge_attrs=["edge_attr"])
            #print(nx.diameter(networkX_graph))
            try:
                networkx_list.append(nx.diameter(networkX_graph))
            except:
                continue
        
        return networkx_list

if __name__ == "__main__":
    pass