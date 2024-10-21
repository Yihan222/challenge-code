import copy
import json
import shlex
import subprocess
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Callable

import networkx as nx
import pandas as pd
import numpy as np
import torch
import torch_geometric
from fastremap import renumber
from torch.utils.data import Dataset, ConcatDataset
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data.dataloader import Collater
from torch_geometric.datasets import TUDataset as _TUDataset
from torch_geometric.utils import from_networkx
from torch_sparse import SparseTensor

from lib.config import Config, ModelName
from lib.graphlets_edge_lists import graphlets_tensor as edge_lists
from lib.run_escape import names, run_escape
from preprocess.to_graphlets import Subgraph

import os.path as osp


class TUDataset(_TUDataset):
    @property
    def raw_dir(self):
        name = 'raw{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, name)  # osp.join(self.root, self.name, name)

    @property
    def processed_dir(self):
        name = 'processed{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, name)  # osp.join(self.root, self.name, name)


last_p_elements = {5: 21, 4: 6, 3: 2}
first_index = {k: len(names[k]) - last_p_elements[k] - 1 for k in last_p_elements}


def to_unattributed_graphlets(
        graphlet_size: int,
        id2graphlet: Dict[int, Subgraph],
        tmp_dir: Path = Path("/tmp/escape_files/"),
) -> Callable[[torch_geometric.data.Data, Optional[Path]], torch_geometric.data.Data]:
    if not tmp_dir.exists():
        tmp_dir.mkdir(exist_ok=True)

    for graphlet_i, graphlet_name in enumerate(
            names[graphlet_size][first_index[graphlet_size] + 1:]
    ):
        id2graphlet[graphlet_i] = Subgraph(
            x=torch.ones((graphlet_size, 1)),
            edge_index=edge_lists[graphlet_size][graphlet_name] - 1,
        )
    id2graphlet[-1] = Subgraph(
        x=torch.zeros((graphlet_size, 1)),
        edge_index=torch.tensor([[], []], dtype=torch.int64),
    )

    def fun(data: torch_geometric.data.Data) -> torch_geometric.data.Data:

        tmp_f_name = tmp_dir / str(uuid.uuid1())

        # Remove one of the bidirectional edges
        edge_index = (
            data.edge_index[:, data.edge_index[0] < data.edge_index[1]]
                .permute(1, 0)
                .numpy()
        )

        """
        Save the edge list to a file to then invoke ESCAPE.
        ESCAPE expects the following format

            <n_nodes> <m_edges>
            <node_id1> <node_id2>
            ...
            # This is a comment in the file
        """
        np.savetxt(
            str(tmp_f_name),
            edge_index,
            delimiter=" ",
            fmt="%d",
            comments="",
            header=f"{data.num_nodes} {len(edge_index)}",
        )

        induced = run_escape(tmp_f_name, graphlet_size)

        # consider only connected graphlets of size graphlet_size
        induced = induced[graphlet_size][first_index[graphlet_size] + 1:]

        # get occurrences of each graphlet
        phashes, estimates = [], []
        for graphlet_i, count in enumerate(induced):
            if count > 0:
                phashes.append(graphlet_i)
                estimates.append(count)

        x = torch.tensor(phashes, dtype=torch.long).view(-1, 1)
        if x.size(0) < 1:
            phash = -1
            x = torch.tensor([[phash]])
            estimates = [0.0]
            if phash not in id2graphlet:
                id2graphlet[phash] = Subgraph(
                    x=torch.zeros((graphlet_size, 1)),
                    edge_index=torch.tensor([[], []], dtype=torch.int64),
                )

        new_data = Data.from_dict(
            {
                "y": data.y,
                "x": x,
                "estimates": torch.tensor(estimates, dtype=torch.float32),
            }
        )

        return new_data

    return fun


def to_attributed_graphlets(
        graphlet_size: int,
        id2graphlet: Dict[int, Subgraph],
        num_node_labels: int,
        seed: int,
        tmp_dir: Path = Path("/tmp/matryoshka_files/"),
) -> Callable[[torch_geometric.data.Data, Optional[Path]], torch_geometric.data.Data]:
    tmp_dir.mkdir(exist_ok=True)

    def fun(data: torch_geometric.data.Data) -> torch_geometric.data.Data:

        # Remove one of the bidirectional edges
        edge_list = (
            data.edge_index[:, data.edge_index[0] < data.edge_index[1]]
                .permute(1, 0)
                .numpy()
        )

        # Transform one-hot features to ids
        # TODO: consider use_nodes_attr=True in load_data
        x = np.argmax(data.x[:, -num_node_labels:], axis=1)

        tmp_f_name = tmp_dir / str(uuid.uuid1())
        with tmp_f_name.open("w") as f:
            np.savetxt(
                f,
                np.vstack(
                    (np.repeat("v", x.shape[0]).astype(str), np.arange(x.shape[0]), x)
                ).T,
                delimiter=" ",
                fmt="%s",
                comments="",
            )
            np.savetxt(
                f,
                np.hstack(
                    (
                        np.repeat("e", edge_list.shape[0]).astype(str).reshape(-1, 1),
                        edge_list,
                        np.repeat("1", edge_list.shape[0]).astype(str).reshape(-1, 1),
                    )
                ),
                delimiter=" ",
                fmt="%s",
                comments="",
            )

        try:
            config = Path.cwd() / "matryoshka-config.txt"
            subprocess.run(
                shlex.split(
                    f"matryoshka-main -i {str(tmp_f_name)} -o {str(tmp_f_name) + '.out'} "
                    f"--psize {graphlet_size} --seed {seed} -c {config}"
                ),
                check=True,
                capture_output=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with open(str(tmp_f_name) + ".out") as f:
                pattern_stats = json.load(f)

        except RuntimeError:
            print(f"Matryoshka exited: {seed}")

            pattern_stats = []

        phashes = []
        estimates = []
        for phash, pstat in pattern_stats:
            if phash not in id2graphlet:
                # Convert back to one-hot
                x = torch.nn.functional.one_hot(
                    torch.tensor(pstat["node_labels"]), num_node_labels
                ).float()

                edge_list = torch.tensor(pstat["edge_list"]).permute(1, 0)
                e1 = torch.cat((edge_list[0], edge_list[1]))
                e2 = torch.cat((edge_list[1], edge_list[0]))
                edge_list = torch.stack((e1, e2))
                edge_list = edge_list[:, edge_list[0].argsort()]

                id2graphlet[phash] = Subgraph(x=x, edge_index=edge_list)

            phashes.append(phash)
            estimates.append(pstat["estimate"])

        x = torch.tensor(phashes, dtype=torch.long).view(-1, 1)
        if x.size(0) < 1:
            phash = -1
            x = torch.tensor([[phash]])
            estimates = [0.0]
            if phash not in id2graphlet:
                id2graphlet[phash] = Subgraph(
                    x=torch.zeros((graphlet_size, num_node_labels)),
                    edge_index=torch.tensor([[], []], dtype=torch.int64),
                )

        new_data = Data.from_dict(
            {"y": data.y, "x": x, "estimates": torch.tensor(estimates)}
        )

        return new_data

    return fun


class Batch(object):
    def __init__(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            graph_has_graphlet: torch.Tensor,
            graphlet_ids: List[int],
            y: torch.Tensor
    ):
        self.x = x
        self.edge_index = edge_index
        self.graphlet_ids = graphlet_ids
        self.graph_has_graphlet = graph_has_graphlet
        self.y = y

    def __len__(self):
        return len(self.y)

    def to(self, dev):
        self.x = self.x.to(dev)
        self.edge_index = self.edge_index.to(dev)
        self.graph_has_graphlet = self.graph_has_graphlet.to(dev)
        self.y = self.y.to(dev)

        return self

    @staticmethod
    def build_batch(
            data: torch_geometric.data.Batch, id2graphlet: Dict[int, Subgraph], common_file=None
    ) -> "Batch":
        # Check if graphlet_id == 0 exists in x because of how remap works
        graphlet_id_zero = (data.x == 0).any().item()

        # remap graphlet_ids to (0..len(data.x.unique()))
        remapped_graphlet_ids, mapping = renumber(
            data.x.numpy(),
            start=0 + int(graphlet_id_zero),
            in_place=False,
            preserve_zero=graphlet_id_zero
        )

        batch_graphlet_indices = torch.tensor(remapped_graphlet_ids.flatten(), dtype=torch.int64)

        graphlet_ids_sorted_by_new_id = (
            list(
                map(lambda e: e[0],  # get key of
                    # (key, value) sorted by value
                    sorted(mapping.items(), key=lambda e: e[1])))
        )

        xs = []
        edge_indices = []

        # create list of xs and edge_indices where
        # xs[i] are the features of the graphlet that was mapped to
        # new_id == i etc.
        for i, graphlet_id in enumerate(graphlet_ids_sorted_by_new_id):
            graphlet = id2graphlet[graphlet_id]
            xs.append(graphlet.x)
            edge_indices.append(graphlet.edge_index + i * graphlet.x.size(0))

        if common_file is not None:
            common = np.loadtxt(str(common_file), dtype=np.int64)
            common = torch.tensor(list(map(lambda x: mapping.get(x, -100), common)))
            common = (batch_graphlet_indices == common.reshape(-1, 1)).any(0)
            data.estimates[~common] = 0

        # Sparse matrix where each row represents a graph
        # and each column a graphlet where
        # m[graph][graphlet] == count of graphlet in graph
        graph_has_graphlet = SparseTensor(
            row=data.batch,
            col=batch_graphlet_indices,
            value=data.estimates
        )

        if graph_has_graphlet.density() > .75:  # FIXME: update parameter if necessary
            graph_has_graphlet = graph_has_graphlet.to_dense()

        return Batch(
            x=torch.cat(xs, dim=0),
            edge_index=torch.cat(edge_indices, dim=1),
            graph_has_graphlet=graph_has_graphlet,
            graphlet_ids=graphlet_ids_sorted_by_new_id,
            y=data.y
        )


def stratified_ksplit(dataset: Dataset, k: int, fold: int = 0):
    classes = dataset.data.y.unique()

    train_per_class, val_per_class = [], []

    for cl in classes:
        cl = cl.item()
        class_subset = dataset[torch.where(dataset.data.y == cl)[0]]
        train, val = ksplit(class_subset, k, fold)
        train_per_class.append(train)
        val_per_class.append(val)

    return (
        ConcatDataset(train_per_class),
        ConcatDataset(val_per_class),
    )


def ksplit(dataset: Dataset, k: int, fold: int = 0):
    dataset = dataset.shuffle()
    n_in_fold = len(dataset) // k

    val_start = n_in_fold * fold
    val_end = n_in_fold * (fold + 1)

    val = dataset[val_start:val_end]

    train_left = dataset[:val_start] if val_start > 0 else None
    train_right = dataset[val_end:] if val_end < len(dataset) else None

    if not train_left:
        train = train_right
    elif not train_right:
        train = train_left
    else:
        train = ConcatDataset([train_left, train_right])

    return train, val


def split_data(dataset, val_size, test_size, get_target):
    classes = dict()
    for idx, data in enumerate(dataset):
        target = get_target(data)
        if target not in classes:
            classes[target] = []
        classes[target].append(idx)

    test, val, train = [], [], []
    for cl in classes:
        ids = torch.tensor(classes[cl])
        perm = torch.randperm(len(ids))
        ids = ids[perm]

        val.append(ids[:val_size])
        test.append(ids[val_size: val_size + test_size])
        train.append(ids[test_size + val_size:])

    return torch.cat(train), torch.cat(val), torch.cat(test)



def get_new_data(conf, ds, id2graphlet):
    common_file = conf.data_path_complete / "common.txt"
    assert common_file.exists()
    common = torch.from_numpy(np.loadtxt(str(common_file), dtype=np.int64))

    new_data_list = []
    for data in ds:
        is_common = (data.x.squeeze() == common.reshape(-1, 1)).any(0)
        data.x = data.x[is_common]
        data.estimates = data.estimates[is_common]
        new_data_list.append(data)

    id2graphlet = {c: id2graphlet[c] for c in common.numpy()}

    return new_data_list, id2graphlet


def size_split(dataset: Dataset, split_path: Path) -> (Dataset, Dataset, Dataset):
    train_idxes = np.loadtxt(str(split_path / f"train_idx.txt"), dtype=np.int64)
    val_idxes = np.loadtxt(str(split_path / f"val_idx.txt"), dtype=np.int64)
    test_idxes = np.loadtxt(str(split_path / f"test_idx.txt"), dtype=np.int64)
    return (
        dataset[torch.from_numpy(train_idxes)],
        dataset[torch.from_numpy(val_idxes)],
        dataset[torch.from_numpy(test_idxes)],
    )


def load_TUDatasets(
        args, graphlet_size: int, epoch: int = 0, to_graphlets=True,
) -> (Dataset, Dataset, Dataset, Dict[int, Subgraph]):
    num_node_labels = {
        "PROTEINS": 3,
        "tg": 0,
    }

    id2graphlet = {}
    if to_graphlets and num_node_labels[args.dataset_name] > 0:
        f = to_attributed_graphlets(
            graphlet_size,
            id2graphlet,
            num_node_labels=num_node_labels[args.dataset_name],
            seed=epoch + 1,
        )
    elif to_graphlets:
        f = to_unattributed_graphlets(graphlet_size, id2graphlet)
    else:
        f = None

    dataset_path = conf.data_path / str(graphlet_size) / args.dataset_name / str(epoch)
    ret = TUDataset(
        str(dataset_path if f is not None else dataset_path / "original"),
        name=args.dataset_name,
        use_node_attr=to_graphlets,
        # True here is only for correct processing of graphlets, attributes are not used
        pre_transform=f,
    )

    id2graphlet_path = str(dataset_path / "id2graphlet")
    if len(id2graphlet) > 0:
        torch.save(id2graphlet, id2graphlet_path)
    elif to_graphlets:
        id2graphlet = torch.load(id2graphlet_path)

    train, val, test = size_split(ret, dataset_path.parent)
    return train, val, test, id2graphlet


def load_data(args, epoch=0):

    if args.gnn == 'karygin':
        max_epoch = 1  # FIXME
        if epoch == 0:

            np.random.seed(10)
            torch.manual_seed(10)

            test_epoch = torch.randint(0, max_epoch, (1,)).tolist()
            val_ds_list, test_ds_list, id2graphlet_list = [], [], []

            for epoch in test_epoch:
                train_ds, val_ds, test_ds, id2graphlet = load_TUDatasets(args, args.graphlet_size, epoch=epoch)
                val_ds_list.append(val_ds)
                test_ds_list.append(test_ds)
                id2graphlet_list.append(id2graphlet)

            return train_ds, val_ds_list, test_ds_list, id2graphlet_list
        else:
            train_ds, _, _, id2graphlet = load_TUDatasets(args, args.graphlet_size, epoch=epoch % max_epoch)
            return train_ds, [], [], [id2graphlet]
    else:
        raise ValueError("Invalid GNN type")


class Batcher(object):
    def __init__(self, id2graphlet, common_file=None):
        self.collater = Collater(follow_batch=[])
        self.id2graphlet = id2graphlet
        self.common_file = common_file

    def __call__(self, l):
        boh = self.collater(l)
        return Batch.build_batch(boh, self.id2graphlet, common_file=self.common_file)
