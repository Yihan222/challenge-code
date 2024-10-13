import os
import ast
import json
import os.path as osp
import pandas as pd

import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import to_networkx

from sklearn.model_selection import train_test_split

from opc.utils.mol import smiles2graph
from opc.utils.split import scaffold_split, similarity_split
#from opc.utils.features import task_properties


class PygPolymerDataset(InMemoryDataset):
    def __init__(
        self, name="prediction", root="data_pyg",repeat_times = 1, task_name = 'tg', set_name='train', transform=None, pre_transform=None
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
        self.fileroot = "{}_raw_{}".format(self.task_type,self.repeat_times)
        self.root = osp.join(root, name, self.task_type, self.fileroot,self.set_name)

        if self.task_properties is None:
            self.num_tasks = None
            self.eval_metric = "jaccard"
        else:
            self.num_tasks = len(self.task_properties)
            self.eval_metric = "wmae"

        super(PygPolymerDataset, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
  

    def get_idx_split(self, split_type="random"):
        
        path = osp.join(self.root, "split", split_type)
        if not os.path.exists(path):
            os.makedirs(path)
        
        try: 
            train_idx = pd.read_csv('./data_pyg/prediction/tg/tg_raw_1/split/random/train.csv.gz', compression='gzip', header = None).values.T[0]
            valid_idx = pd.read_csv('./data_pyg/prediction/tg/tg_raw_1/split/random/valid.csv.gz', compression='gzip', header = None).values.T[0]
            test_idx = pd.read_csv('./data_pyg/prediction/tg/tg_raw_1/split/random/test.csv.gz', compression='gzip', header = None).values.T[0]
        except:
            raw_file_name = "{}_raw.csv".format(self.task_type)
            csv_file = osp.join(self.root,raw_file_name)
            data_df = pd.read_csv(csv_file)
            #print(len(data_df))
            #print('Splitting with random seed 42 and ratio 0.6/0.1/0.3')
            if split_type=='scaffold':
                train_idx, valid_idx, test_idx = scaffold_split(data_df, train_ratio=0.6, valid_ratio=0.1,test_ratio=0.3)
            elif split_type=='random':
                full_idx = list(range(len(data_df)))
                train_ratio, valid_ratio, test_ratio = 0.6, 0.1, 0.3
                train_idx, test_idx, _, test_df = train_test_split(full_idx, data_df, test_size=test_ratio, random_state=42)
                train_idx, valid_idx, _, _ = train_test_split(train_idx, train_idx, test_size=valid_ratio/(valid_ratio+train_ratio), random_state=42)
            df_train = pd.DataFrame({'train': train_idx})
            df_valid = pd.DataFrame({'valid': valid_idx})
            df_test = pd.DataFrame({'test': test_idx})
            #df_test_all = pd.DataFrame({'smiles':test_df['SMILES'],'tg_groundtruth':test_df['tg']})
            df_train.to_csv(osp.join(path, 'train.csv.gz'), index=False, header=False, compression="gzip")
            df_valid.to_csv(osp.join(path, 'valid.csv.gz'), index=False, header=False, compression="gzip")
            df_test.to_csv(osp.join(path, 'test.csv.gz'), index=False, header=False, compression="gzip")
            #df_test_all.to_csv('test.csv')
        return {'train': torch.tensor(train_idx, dtype = torch.long), 'valid': torch.tensor(valid_idx, dtype = torch.long), 'test': torch.tensor(test_idx, dtype = torch.long)}


    def get_task_weight(self, ids):
        if self.task_properties is not None:
            try:
                labels = self._data.y[torch.LongTensor(ids)]
                task_weight = []
                for i in range(labels.shape[1]):
                    valid_num = labels[:, i].eq(labels[:, i]).sum()
                    task_weight.append(valid_num)
                task_weight = torch.sqrt(
                    1 / torch.tensor(task_weight, dtype=torch.float32)
                )
                print('****************2:\n')
                print(task_weight)
                print(len(task_weight))
                print(task_weight / task_weight.sum() * len(task_weight))
                return task_weight / task_weight.sum() * len(task_weight)
            except Exception as e:
                print(f"An error occurred: {e}")
        else:
            return None

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
                    print(task)
                    print(row[task])
                    y.append(float(row[task]))
                g.y = torch.tensor(y, dtype=torch.float32).view(1, -1)
            else:
                g.y = torch.tensor(
                    ast.literal_eval(row["labels"]), dtype=torch.float32
                ).view(1, -1)
            networkX_graph = to_networkx(g, node_attrs=["x"], edge_attrs=["edge_attr"])
            networkx_list.append(networkX_graph)
            pyg_graph_list.append(g)
        print(networkx_list)
        pyg_graph_list = (
            pyg_graph_list
            if self.pre_transform is None
            else self.pre_transform(pyg_graph_list)
        )
        print("Saving...")
        torch.save(self.collate(pyg_graph_list), self.processed_paths[0])

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


if __name__ == "__main__":
    pass