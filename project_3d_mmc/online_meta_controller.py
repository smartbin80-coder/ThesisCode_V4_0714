"""Online adaptation and safety utilities for GAT-guided MMC steps."""

import numpy as np
import torch
import torch.nn.functional as F


ETA_MIN = 0.25
ETA_MAX = 1.5


def clamp_eta(values, eta_min=ETA_MIN, eta_max=ETA_MAX):
    """Clamp node-wise eta values to the physical safety range."""
    if torch.is_tensor(values):
        return torch.clamp(values, eta_min, eta_max)
    return np.clip(np.asarray(values, dtype=float), eta_min, eta_max)


@torch.no_grad()
def predict_component_eta(model, graph_data, device=None, damping_factor=1.0):
    """Predict component eta values with optional damping."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    graph_data = graph_data.to(device)
    outputs = model(graph_data)
    eta = clamp_eta(outputs["node_eta_pred"] * float(damping_factor))
    return eta.detach().cpu().numpy(), outputs


def apply_component_step_scale(params_old, params_new, eta_node_pred):
    """Apply component-wise step scales to flattened 9-parameter MMC vectors."""
    old = np.asarray(params_old, dtype=float).reshape(-1, 9)
    new = np.asarray(params_new, dtype=float).reshape(-1, 9)
    eta = clamp_eta(eta_node_pred).reshape(-1, 1)
    if eta.shape[0] != old.shape[0]:
        raise ValueError("eta_node_pred length must match number of components")
    return (old + eta * (new - old)).reshape(-1)


def check_trust_bias(actual_delta, predicted_delta, threshold=0.10, eps=1e-12):
    """Return whether online adaptation should trigger from trust-bias drift."""
    bias = abs(float(actual_delta) - float(predicted_delta)) / max(abs(float(predicted_delta)), eps)
    return bias > float(threshold), bias


def fallback_to_safe_eta_search(params_old, params_new, evaluator, eta_candidates, volfrac):
    """Conservatively choose a graph-level eta using true FEM responses."""
    candidates = sorted(set(float(v) for v in eta_candidates) | {0.0}, reverse=True)
    rows = []
    for eta in candidates:
        trial = np.asarray(params_old, dtype=float) + eta * (np.asarray(params_new, dtype=float) - np.asarray(params_old, dtype=float))
        compliance, volume_fraction, *_ = evaluator(trial)
        violation = max(0.0, float(volume_fraction) - float(volfrac))
        rows.append((eta, trial, float(compliance), float(volume_fraction), violation))
    feasible = [row for row in rows if row[4] <= 1e-8]
    if feasible:
        return min(feasible, key=lambda row: row[2])
    return min(rows, key=lambda row: (row[4], row[2]))


def node_degrees(edge_index, num_nodes):
    """Compute undirected-style node degree from directed edge_index."""
    if torch.is_tensor(edge_index):
        edge_index = edge_index.detach().cpu().numpy()
    deg = np.zeros(num_nodes, dtype=float)
    if edge_index.size:
        for node in edge_index.reshape(2, -1).ravel():
            deg[int(node)] += 1.0
    if deg.max() > 0:
        deg /= deg.max()
    return deg


def select_key_components(graph_data, top_k=5):
    """Select components for sparse true labels with fixed physics-first priority."""
    num_nodes = int(graph_data.x.shape[0])
    top_k = max(1, min(int(top_k), num_nodes))
    if hasattr(graph_data, "component_strain_energy_norm") and torch.max(graph_data.component_strain_energy_norm).item() > 0:
        score = graph_data.component_strain_energy_norm.detach().cpu().numpy()
    elif hasattr(graph_data, "node_load_prior") and torch.max(graph_data.node_load_prior).item() > 0:
        score = graph_data.node_load_prior.detach().cpu().numpy()
    else:
        score = node_degrees(graph_data.edge_index, num_nodes)
    return np.argsort(-score)[:top_k]


def build_sparse_true_component_labels(
    params_old,
    params_new,
    current_eta_pred,
    evaluator,
    key_indices,
    volfrac,
    local_candidates=(0.5, 1.0, 1.5),
):
    """Generate sparse FEM-verified component labels and soft pseudo labels."""
    current_eta_pred = clamp_eta(current_eta_pred)
    true_labels = np.asarray(current_eta_pred, dtype=float).copy()
    true_mask = np.zeros_like(true_labels, dtype=bool)
    for comp_idx in key_indices:
        best = None
        for eta in local_candidates:
            trial_eta = np.asarray(current_eta_pred, dtype=float).copy()
            trial_eta[int(comp_idx)] = float(eta)
            trial = apply_component_step_scale(params_old, params_new, trial_eta)
            compliance, volume_fraction, *_ = evaluator(trial)
            feasible = float(volume_fraction) <= float(volfrac) + 1e-8
            row = (feasible, float(compliance), float(volume_fraction), float(eta))
            if best is None:
                best = row
            elif row[0] and not best[0]:
                best = row
            elif row[0] == best[0] and row[1] < best[1]:
                best = row
        true_labels[int(comp_idx)] = best[3]
        true_mask[int(comp_idx)] = True
    return true_labels, true_mask


def weighted_online_mse(pred, true_label, true_mask, true_weight=5.0, pseudo_weight=0.1):
    """Weighted sparse-true plus soft-pseudo online adaptation loss."""
    true_mask = true_mask.bool()
    pseudo_mask = ~true_mask
    loss = pred.sum() * 0.0
    if torch.any(true_mask):
        loss = loss + float(true_weight) * F.mse_loss(pred[true_mask], true_label[true_mask])
    if torch.any(pseudo_mask):
        loss = loss + float(pseudo_weight) * F.mse_loss(pred[pseudo_mask], true_label[pseudo_mask])
    return loss


def adapt_online_on_sparse_labels(model, graph_data, true_labels, true_mask, lr=1e-3, steps=3):
    """Freeze GAT layers and adapt only the final heads on sparse component labels."""
    device = next(model.parameters()).device
    graph_data = graph_data.to(device)
    true_labels = torch.as_tensor(true_labels, dtype=torch.float32, device=device)
    true_mask = torch.as_tensor(true_mask, dtype=torch.bool, device=device)
    for param in model.gat_layers.parameters():
        param.requires_grad_(False)
    trainable = list(model.node_eta_head.parameters()) + list(model.node_class_head.parameters())
    optimizer = torch.optim.SGD(trainable, lr=lr)
    model.train()
    for _ in range(int(steps)):
        optimizer.zero_grad()
        pred = model(graph_data)["node_eta_pred"]
        loss = weighted_online_mse(pred, true_labels, true_mask)
        loss.backward()
        optimizer.step()
    for param in model.gat_layers.parameters():
        param.requires_grad_(True)
    return model


class SafetyDampingController:
    """Track fallback events and damp GAT eta outputs for a few steps."""

    def __init__(self, damping_factor=0.8, damping_steps=3):
        self.damping_factor = float(damping_factor)
        self.damping_steps = int(damping_steps)
        self.remaining = 0

    def on_fallback(self):
        """Activate damping after a safety fallback."""
        self.remaining = self.damping_steps

    def factor(self):
        """Return the current damping factor and advance the counter."""
        if self.remaining <= 0:
            return 1.0
        self.remaining -= 1
        return self.damping_factor
