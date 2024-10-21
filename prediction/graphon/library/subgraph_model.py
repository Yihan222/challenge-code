import torch
import torch.nn as nn

from layer import KaryGNN_node

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
    
class KaryGNN(torch.nn.Module):
    """ Graph model that applies a GNN to the graph of
        k-sized graphlets and represents a graph as weighted sum of
        the nodes of the graphlets, where the weight is the number of
        occurrences of the graphlet within the graph

    Args:
        gnn: GNN model used for vertex representation
        graphlet_sz
    """

    def __init__(
        self,
        num_layer=5,
        emb_dim=300,
        gnn_type="gin",
        drop_ratio=0.5,
        JK="last",
        graphlet_sz=5,
    ):
        """
        num_tasks (int): number of labels to be predicted
        virtual_node (bool): whether to add virtual node or not
        """

        super(KaryGNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.graphlet_sz = graphlet_sz
        self.graphlets_repr = None

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        self.gnn_node = KaryGNN_node(
            num_layer,
            emb_dim,
            JK=JK,
            drop_ratio=drop_ratio,
            gnn_type=gnn_type,
        )


    def forward(self, batch):
        h_node = self.gnn_node(batch.x, batch.edge_index)
        # Represent a graphlet as the sum of its nodes
        graphlets_repr = (
            h_node.reshape(-1, self.graphlet_sz, h_node.size(-1)).sum(dim=1)
        )
        self.graphlets_repr = graphlets_repr  # save for regularization

        eps = 0.0001
        normalized_estimates = batch.graph_has_graphlet / (batch.graph_has_graphlet.sum(dim=-1).unsqueeze(-1) + eps)
        return normalized_estimates.matmul(graphlets_repr)

class FinalLayer(nn.Module):
    def __init__(
            self,
            graph_embedder,
            emb_dim: int = 300,
            batch_norm: Optional[nn.Module] = None,
            dropout: float = 0.0,
    ):
        super(FinalLayers, self).__init__()
        self.graph_embedder = graph_embedder
        self.emb_dim = emb_dim
        self.batch_norm = batch_norm
        self.predictor = MLP(
            emb_dim + 2048, hidden_features=4 * emb_dim, out_features=1
        )

    def forward(self, batch):
        out = self.graph_embedder(batch)
        if self.batch_norm is not None:
            out = self.batch_norm(out)

        out = self.predictor(out)
        return out


if __name__ == "__main__":
    pass
