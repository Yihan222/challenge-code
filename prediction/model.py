import torch
import torch.nn as nn
import os
import os.path as osp
import numpy as np
import heapq
import sys
import pandas as pd
from torch_geometric.nn import (
    global_add_pool,
    global_mean_pool,
    global_max_pool,
    GlobalAttention,
)

from layer import GNN_node, GNN_node_Virtualnode

class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        linear_layer = nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return x
    
class GNN(torch.nn.Module):

    def __init__(
        self,
        num_task,
        test,
        repeat_time,
        num_layer=5,
        emb_dim=300,
        gnn_type="gin",
        virtual_node=True,
        residual=False,
        drop_ratio=0.1,
        JK="last",
        graph_pooling="sum",
    ):
        """
        num_tasks (int): number of labels to be predicted
        virtual_node (bool): whether to add virtual node or not
        """

        super(GNN, self).__init__()
        self.test = test
        self.repeat_time = repeat_time
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_task = num_task
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(
                num_layer,
                emb_dim,
                JK=JK,
                drop_ratio=drop_ratio,
                residual=residual,
                gnn_type=gnn_type,
            )
        else:
            self.gnn_node = GNN_node(
                num_layer,
                emb_dim,
                JK=JK,
                drop_ratio=drop_ratio,
                residual=residual,
                gnn_type=gnn_type,
            )

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(
                    torch.nn.Linear(emb_dim, 2 * emb_dim),
                    torch.nn.BatchNorm1d(2 * emb_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2 * emb_dim, 1),
                )
            )
        else:
            raise ValueError("Invalid graph pooling type.")

        self.predictor = MLP(
            emb_dim + 2048, hidden_features=4 * emb_dim, out_features=num_task
        )

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)
        #self.similarity(h_node,h_graph)
        h_graph = torch.cat([h_graph, batched_data.fp.type_as(h_graph)], dim=1)
        #self.save_h_graph(h_graph)
        return self.predictor(h_graph)
    
    def save_h_graph(self, h_graph):
        save_path = './embeddings/h_graph/gin-virtual/train/{}'.format(self.repeat_time) # the number here is the test repeat time
        if not osp.exists(save_path):
            os.makedirs(save_path)
        save_name = osp.join(save_path,'{}.npy'.format(self.test))
        h = h_graph.cpu()
        h = h[0].numpy()

        try:
            preh = np.load(save_name)
            newh = np.vstack((preh, h))
            np.save(save_name, newh)
        except IOError:
            # if not such file
            np.save(save_name, h)
        
    # compare similarity of single node representation in the graph with the graph representation
    def similarity(self, h_node, h_graph):
        ngsim = []
        cosi = torch.nn.CosineSimilarity(dim=0)
        for tensorn in h_node:
            simi = cosi(tensorn, h_graph[0])
            ngsim.append(simi.item())

        result = {
            "gnn_type":'gin_virtual',
            "train_repeat_time":self.repeat_time,
            "num_nodes":len(h_node),
            "max_similarity":np.round(heapq.nlargest(5,ngsim),decimals=2),
            "min_similarity":np.round(heapq.nsmallest(5,ngsim),decimals=2),
        }
        df = pd.DataFrame([result])
        save_name = './gnn_res/tg/similarity/node_graph_simi_test1.csv'
        if osp.exists(save_name):
            df.to_csv(save_name, mode="a", header=False, index=False)
        else:
            df.to_csv(save_name, index=False)
        print('****** {} saved! ******'.format(save_name))


if __name__ == "__main__":
    pass
