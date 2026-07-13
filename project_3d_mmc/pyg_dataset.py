"""PyTorch Geometric dataset loader for exported MMC graph npz files."""

from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, Dataset


def _read_split_ids(split_file):
    if split_file is None:
        return None
    with open(split_file, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def _trajectory_id_from_file(path):
    name = Path(path).name
    if name.startswith("traj_"):
        return "_".join(name.split("_")[:2])
    return ""


def _npz_tensor(npz, key, shape, dtype=torch.float32):
    if key in npz:
        return torch.as_tensor(npz[key], dtype=dtype)
    return torch.zeros(shape, dtype=dtype)


def _npz_scalar_tensor(npz, key, default=0.0, dtype=torch.float32):
    if key in npz:
        return torch.as_tensor([float(npz[key])], dtype=dtype)
    return torch.as_tensor([default], dtype=dtype)


class MMCStepDataset(Dataset):
    """Load graph npz files as PyTorch Geometric `Data` objects.

    `data.y` is the continuous eta target. Extra targets are included for
    multi-task training:
    `eta_label_index`, `eta_feasible_mask`, `eta_failure_flag`,
    `response_targets`, `compliance_candidates`, and `volume_candidates`.
    """

    def __init__(self, root, transform=None, pre_transform=None, target_key="eta_label", split_file=None):
        self.root_dir = Path(root)
        self.target_key = target_key
        split_ids = _read_split_ids(split_file)
        files = sorted(self.root_dir.glob("*_graph.npz"))
        if split_ids is not None:
            files = [p for p in files if _trajectory_id_from_file(p) in split_ids]
        self.files = files
        super().__init__(str(self.root_dir), transform, pre_transform)

    def len(self):
        """Return the number of graph files."""
        return len(self.files)

    def get(self, idx):
        """Load one graph npz file."""
        with np.load(self.files[idx]) as npz:
            x = torch.as_tensor(npz["node_features"], dtype=torch.float32)
            edge_index = torch.as_tensor(npz["edge_index"], dtype=torch.long)
            edge_attr = torch.as_tensor(npz["edge_attr"], dtype=torch.float32)
            num_nodes = x.shape[0]
            num_edges = edge_index.shape[1]
            eta_label = float(npz["eta_label"]) if "eta_label" in npz else float(npz[self.target_key])
            eta_label_index = int(npz["eta_label_index"]) if "eta_label_index" in npz else 0
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.as_tensor([float(npz[self.target_key])], dtype=torch.float32),
            )
            data.feature_schema_version = torch.as_tensor([int(npz["feature_schema_version"]) if "feature_schema_version" in npz else 1], dtype=torch.long)
            data.eta_label = torch.as_tensor([eta_label], dtype=torch.float32)
            if "node_features_raw" in npz:
                data.x_raw = torch.as_tensor(npz["node_features_raw"], dtype=torch.float32)
            else:
                data.x_raw = torch.zeros((num_nodes, 11), dtype=torch.float32)
            if "global_features" in npz:
                data.global_features = torch.as_tensor(npz["global_features"], dtype=torch.float32)
            else:
                data.global_features = torch.zeros(7, dtype=torch.float32)
            data.compliance = torch.as_tensor([float(npz["compliance"])], dtype=torch.float32)
            data.volume_fraction = torch.as_tensor([float(npz["volume_fraction"])], dtype=torch.float32)
            data.iteration = torch.as_tensor([int(npz["iteration"])], dtype=torch.long)
            data.params = torch.as_tensor(npz["params"], dtype=torch.float32)
            data.delta_params = torch.as_tensor(npz["delta_params"], dtype=torch.float32)
            data.eta_candidates = torch.as_tensor(npz["eta_candidates"], dtype=torch.float32)
            data.compliance_candidates = torch.as_tensor(npz["compliance_candidates"], dtype=torch.float32)
            data.volume_candidates = torch.as_tensor(npz["volume_candidates"], dtype=torch.float32)
            if "response_targets" in npz:
                data.response_targets = torch.as_tensor(npz["response_targets"], dtype=torch.float32)
            else:
                data.response_targets = torch.stack([data.compliance_candidates, data.volume_candidates], dim=1)
            data.load_point = _npz_tensor(npz, "load_point", (3,))
            data.node_load_distance_norm = _npz_tensor(npz, "node_load_distance_norm", (num_nodes,))
            data.node_load_prior = _npz_tensor(npz, "node_load_prior", (num_nodes,))
            data.edge_load_prior = _npz_tensor(npz, "edge_load_prior", (num_edges, 1))
            data.eta_node_label = _npz_tensor(npz, "eta_node_label", (num_nodes,)) + (
                0.0 if "eta_node_label" in npz else eta_label
            )
            data.eta_node_label_index = _npz_tensor(npz, "eta_node_label_index", (num_nodes,), dtype=torch.long) + (
                0 if "eta_node_label_index" in npz else eta_label_index
            )
            data.trust_actual_delta = _npz_scalar_tensor(npz, "trust_actual_delta")
            data.trust_predicted_delta = _npz_scalar_tensor(npz, "trust_predicted_delta")
            data.trust_bias = _npz_scalar_tensor(npz, "trust_bias")
            data.component_strain_energy_norm = _npz_tensor(npz, "component_strain_energy_norm", (num_nodes,))
            if "eta_label_index" in npz:
                data.eta_label_index = torch.as_tensor([int(npz["eta_label_index"])], dtype=torch.long)
            else:
                data.eta_label_index = torch.as_tensor([eta_label_index], dtype=torch.long)
            if "eta_feasible_mask" in npz:
                data.eta_feasible_mask = torch.as_tensor(npz["eta_feasible_mask"], dtype=torch.bool)
            else:
                data.eta_feasible_mask = torch.ones_like(data.eta_candidates, dtype=torch.bool)
            if "eta_failure_flag" in npz:
                data.eta_failure_flag = torch.as_tensor([int(npz["eta_failure_flag"])], dtype=torch.long)
            else:
                data.eta_failure_flag = torch.zeros(1, dtype=torch.long)
            if "trajectory_id" in npz:
                data.trajectory_id = str(npz["trajectory_id"])
            else:
                data.trajectory_id = _trajectory_id_from_file(self.files[idx])
        return data


def create_dataloader(dataset_dir, batch_size=8, shuffle=True, split_file=None, **dataset_kwargs):
    """Create a PyG DataLoader for MMC step-size graph data."""
    from torch_geometric.loader import DataLoader

    dataset = MMCStepDataset(dataset_dir, split_file=split_file, **dataset_kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
