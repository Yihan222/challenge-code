from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Any, List


class ModelName(Enum):
    KaryGNN = "KaryGNN"
    GNN = "GNN"

    def __str__(self):
        return self.name


@dataclass
class BatchNormConfig:
    presence: bool
    affine: bool


@dataclass
class Config:
    """
    Attributes:

        dataset_name: Name of the dataset to be used. The datasets currently
            supported are the ones in `torch_geometric.datasets.TUDataset`

        model: Name of the model to be used. Should be present in the
            enumeration present in `models.ModelName`

        num_splits: Number of splits in the k_fold. This is used to split
            the dataset in three parts: train, validation, test by
            `load_data.ksplit`

        seed: Seed for reproducibility

        batch_size

        lr: Learning rate

        num_layers: Number of GNN recursion layers in all models that use a
            GNN. In the case of models.GraphletCounting it is the number of
            Linear layers to apply to the count vector

        mlp_num_hidden: Number of hidden layers in an MLP. It should always
            be >= 1. If a MLP is instantiated with mlp_num_hidden = 1 then
            the model is [Linear, Activation, Linear]. Look at
            `models.build_mlp` for further information

        mlp_hidden_dim: Dimensionality of the hidden layer of a MLP. For
            simplicity also the dimensionality of the output
            of the MLP (indeed `self.vertex_embed_dim` returns the same value).

        graph_embed_dim: Dimensionality of the graph representation. This is
            needed only when `self.vertex_embed_dim` is not specified, since in
            that case the graph representation dimension depends on it.

        graphlet_size

        dev: device where most of the computation should be carried out

        num_epochs

        data_dir: Directory where torch_geometric.datasets.TUDataset should
            store the data. Additionally this directory is used for ESCAPE
            temporary results
    """

    dataset_name: str
    model: ModelName
    num_splits: int = 1
    split: int = 1
    gnn_type: str = "gin"
    seed: int = 42
    batch_size: int = 32
    lr: float = 0.001
    jk: bool = True
    graph_pooling: str = "sum"
    graphlet_size: int = 5

    only_common: bool = False
    num_epochs: int = 500
    data_dir: str = "data"
    train_size: int = 80
    num_out: int = 1
    batch_norm: BatchNormConfig = BatchNormConfig(True, False)


    @property
    def vertex_embed_dim(self) -> int:
        return self.mlp_hidden_dim

    @property
    def data_path(self) -> Path:
        if self.data_dir:
            ret = Path(self.data_dir)
        else:
            ret = Path.home() / "graphlet_data/"
        if not ret.exists():
            ret.mkdir(parents=True, exist_ok=True)
        return ret

    @property
    def data_path_complete(self) -> Path:
        return self.data_path / str(self.graphlet_size) / self.dataset_name


@dataclass
class HyperConfig:
    split: List[int]
    lr: List[float]
    batch_size: List[int]
    classifier_num_hidden: List[int]
    classifier_h_dim: List[int]
    classifier_dropout: List[float]
    gpu_perc: float
    seed: List[int]


@dataclass
class HyperConfigAnyGNN(HyperConfig):
    num_layers: List[int]
    mlp_hidden_dim: List[int]
    jk: List[bool]
    irm: List[Optional[float]]
    cutoff: List[Optional[float]]
    reg_const: List[Optional[float]]
    label_smooth: List[Optional[float]]


@dataclass
class HyperConfigGraphletCounting(HyperConfig):
    graph_embed_dim: List[int]
    gc_num_layers: List[int]
