import dgl
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.datasets import FakeDataset
from torch_geometric.data import Data
from opc.dataset_pyg import PygPolymerDataset


def pyg_to_dgl(pyg_data):
    """
    convert pyg data to dgl
    """
    # dgl figure
    g = dgl.graph((pyg_data.edge_index[0], pyg_data.edge_index[1]), num_nodes=pyg_data.num_nodes)

    # node feature
    if pyg_data.x is not None:
        g.ndata['feat'] = pyg_data.x

    # edge feature
    if pyg_data.edge_attr is not None:
        g.edata['feat'] = pyg_data.edge_attr

    # graph label
    if pyg_data.y is not None:
        g.graph_data = {'label': pyg_data.y}

    return g


class DGLDatasetFromPyG(Dataset):
    """
    transform pyg data to dgl
    """
    def __init__(self, pyg_dataset):
        self.dgl_graphs = [pyg_to_dgl(data) for data in pyg_dataset]

    def __len__(self):
        return len(self.dgl_graphs)

    def __getitem__(self, idx):
        return self.dgl_graphs[idx]


def collate_gnn(batch):
    """
    将一批 DGL 图对象合并为一个批处理图。
    """
    batched_graph = dgl.batch(batch)

    graph_labels = torch.stack([g.graph_data['label'] for g in batch if 'label' in g.graph_data])

    return batched_graph, graph_labels

  
# prepare dense tensors for GNNs using them; such as RingGNN, 3WLGNN
def collate_dense_gnn(batch):

    # 合并图级数据（如标签），保存在字典中
    labels = torch.stack([g.graph_data['label'] for g in batch if 'label' in g.graph_data])

    g = batch[0]
    adj = sym_normalize_adj(g.adjacency_matrix().to_dense())        
    """
        Adapted from https://github.com/leichen2018/Ring-GNN/
        Assigning node and edge feats::
        we have the adjacency matrix in R^{n x n}, the node features in R^{d_n} and edge features R^{d_e}.
        Then we build a zero-initialized tensor, say T, in R^{(1 + d_n + d_e) x n x n}. T[0, :, :] is the adjacency matrix.
        The diagonal T[1:1+d_n, i, i], i = 0 to n-1, store the node feature of node i. 
        The off diagonal T[1+d_n:, i, j] store edge features of edge(i, j).
    """

    zero_adj = torch.zeros_like(adj)
    
    in_dim = g.ndata['feat'].shape[1]
    
    # use node feats to prepare adj
    adj_node_feat = torch.stack([zero_adj for j in range(in_dim)])
    adj_node_feat = torch.cat([adj.unsqueeze(0), adj_node_feat], dim=0)
    
    for node, node_feat in enumerate(g.ndata['feat']):
        adj_node_feat[1:, node, node] = node_feat

    x_node_feat = adj_node_feat.unsqueeze(0)
    
    return x_node_feat, labels

def sym_normalize_adj(adj):
    deg = torch.sum(adj, dim = 0)#.squeeze()
    deg_inv = torch.where(deg>0, 1./torch.sqrt(deg), torch.zeros(deg.size()))
    deg_inv = torch.diag(deg_inv)
    return torch.mm(deg_inv, torch.mm(adj, deg_inv))

def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']
        
        
        This function is called inside a function in TUsDataset class.
    """
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']
    
    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)
    
    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g
    
if __name__ == "__main__":

    # 示例：PyG 数据集到 DGL 数据集转换
    dataset = PygPolymerDataset(name="prediction", root="data_pyg", set_name="train",task_name='o2',repeat_times=1,_use_concat_train = False)
    dgl_dataset = DGLDatasetFromPyG(dataset)

    # 创建 DataLoader
    batch_size = 4
    dataloader = DataLoader(dgl_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_dense_gnn)

    # 查看一个批次数据
    for batched_graph, labels in dataloader:
        print("Batched DGL Graph:")
        print(batched_graph)
        print("Graph Labels:")
        print(labels)
        break