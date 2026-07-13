"""Export MMC component states as GNN graph data."""

import numpy as np


def approximate_overlap(comp_i, comp_j):
    """Estimate overlap with bounding-sphere radii."""
    dist = np.linalg.norm(comp_i.center() - comp_j.center())
    ri = np.linalg.norm([comp_i.L1, comp_i.L2, comp_i.L3])
    rj = np.linalg.norm([comp_j.L1, comp_j.L2, comp_j.L3])
    return dist < (ri + rj), max(0.0, 1.0 - dist / (ri + rj + 1e-12))


def _domain_volume(config):
    return float(config.DL * config.DW * config.DH)


def _reshape_or_zero(values, num_components):
    if values is None:
        return np.zeros((num_components, 9), dtype=float)
    return np.asarray(values, dtype=float).reshape(num_components, 9)


def build_load_prior_features(components, config):
    """Build load-distance priors for physics-biased graph attention."""
    load_y = getattr(config, "load_y", config.DW / 2)
    load_z = getattr(config, "load_z", config.DH / 2)
    load_point = np.array([config.DL, load_y, load_z], dtype=float)
    centers = np.vstack([c.center() for c in components]) if components else np.zeros((0, 3), dtype=float)
    dist = np.linalg.norm(centers - load_point, axis=1)
    max_dim = max(config.DL, config.DW, config.DH)
    node_load_distance_norm = dist / max(max_dim, 1e-12)
    prior_lambda = float(getattr(config, "load_prior_lambda", 2.0))
    node_load_prior = np.exp(-prior_lambda * node_load_distance_norm)
    return load_point, node_load_distance_norm.astype(float), node_load_prior.astype(float)


def build_edge_load_prior(edge_index, node_load_prior):
    """Build one scalar physical attention bias for every directed edge."""
    if edge_index.shape[1] == 0:
        return np.zeros((0, 1), dtype=float)
    src = edge_index[0]
    dst = edge_index[1]
    return (node_load_prior[src] + node_load_prior[dst]).reshape(-1, 1).astype(float)


def build_component_step_labels(eta_label, eta_label_index, num_components):
    """Broadcast graph-level eta labels as a v1 component-level training target."""
    eta_node_label = np.full(num_components, float(eta_label), dtype=float)
    eta_node_label_index = np.full(num_components, int(eta_label_index), dtype=np.int64)
    return eta_node_label, eta_node_label_index


def build_node_features(components, config, compliance_grad=None, volume_grad=None, delta_params=None):
    """Build normalized training features and raw physical features."""
    n_comp = len(components)
    raw = np.vstack([c.as_node_feature() for c in components])
    cgrad = _reshape_or_zero(compliance_grad, n_comp)
    vgrad = _reshape_or_zero(volume_grad, n_comp)
    delta = _reshape_or_zero(delta_params, n_comp)

    comp_grad_norm = np.linalg.norm(cgrad, axis=1)
    vol_grad_norm = np.linalg.norm(vgrad, axis=1)
    delta_center_norm = np.linalg.norm(delta[:, 0:3] / np.array([config.DL, config.DW, config.DH]), axis=1)
    delta_size_norm = np.linalg.norm(delta[:, 3:6] / np.array([config.DL, config.DW, config.DH]), axis=1)
    delta_angle_norm = np.linalg.norm(delta[:, 6:9], axis=1) / np.pi
    delta_norm = np.linalg.norm(delta, axis=1)

    comp_grad_scaled = np.log1p(np.abs(comp_grad_norm))
    if np.max(comp_grad_scaled) > 0:
        comp_grad_scaled = comp_grad_scaled / np.max(comp_grad_scaled)
    vol_grad_scaled = np.log1p(np.abs(vol_grad_norm))
    if np.max(vol_grad_scaled) > 0:
        vol_grad_scaled = vol_grad_scaled / np.max(vol_grad_scaled)
    delta_scaled = np.log1p(np.abs(delta_norm))
    if np.max(delta_scaled) > 0:
        delta_scaled = delta_scaled / np.max(delta_scaled)

    params = raw[:, :9]
    max_dim = max(config.DL, config.DW, config.DH)
    centers = params[:, 0:3] / np.array([config.DL, config.DW, config.DH])
    sizes = params[:, 3:6] / max_dim
    angles = params[:, 6:9]
    volume = raw[:, 9:10] / _domain_volume(config)
    active = raw[:, 10:11]

    features = np.column_stack(
        [
            centers,
            sizes,
            np.sin(angles),
            np.cos(angles),
            volume,
            active,
            comp_grad_scaled,
            vol_grad_scaled,
            delta_scaled,
            delta_center_norm,
            delta_size_norm,
            delta_angle_norm,
        ]
    )
    return features.astype(float), raw.astype(float)


def build_edge_features(comp_i, comp_j, config):
    """Build directional and alignment edge features."""
    dvec = comp_j.center() - comp_i.center()
    dist = float(np.linalg.norm(dvec))
    overlap, score = approximate_overlap(comp_i, comp_j)
    Ri = comp_i.direction_vectors()
    Rj = comp_j.direction_vectors()
    axis_dot = np.sum(Ri * Rj, axis=1)
    alignment_trace = float(np.trace(Ri @ Rj.T) / 3.0)
    norm_dvec = dvec / np.array([config.DL, config.DW, config.DH])
    norm_dist = dist / max(config.DL, config.DW, config.DH)
    return np.array([norm_dist, *norm_dvec, alignment_trace, *axis_dot, float(overlap), score], dtype=float)


def build_global_features(config, iteration, compliance, volume_fraction):
    """Build graph-level features used by PyG loaders."""
    load_y = getattr(config, "load_y", config.DW / 2)
    load_z = getattr(config, "load_z", config.DH / 2)
    return np.array(
        [
            float(iteration) / max(float(config.max_iter), 1.0),
            float(compliance),
            float(volume_fraction),
            float(config.volfrac),
            float(load_y) / config.DW,
            float(load_z) / config.DH,
            float(config.num_components) / max(float(getattr(config, "max_components_for_dataset", 12)), 1.0),
        ],
        dtype=float,
    )


def build_component_graph(
    components,
    config,
    compliance_grad=None,
    volume_grad=None,
    delta_params=None,
    iteration=0,
    compliance=0.0,
    volume_fraction=0.0,
):
    """Build a bidirectional MMC graph with normalized features."""
    node_features, node_features_raw = build_node_features(components, config, compliance_grad, volume_grad, delta_params)
    edges, attrs = [], []
    threshold = 0.45 * max(config.DL, config.DW, config.DH)
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            ci, cj = components[i], components[j]
            dist = float(np.linalg.norm(cj.center() - ci.center()))
            overlap, _ = approximate_overlap(ci, cj)
            if dist < threshold or overlap:
                attr_ij = build_edge_features(ci, cj, config)
                attr_ji = build_edge_features(cj, ci, config)
                edges += [[i, j], [j, i]]
                attrs += [attr_ij, attr_ji]
    if not edges:
        edge_index = np.zeros((2, 0), dtype=int)
        edge_attr = np.zeros((0, 10), dtype=float)
    else:
        edge_index = np.asarray(edges, dtype=int).T
        edge_attr = np.asarray(attrs, dtype=float)
    global_features = build_global_features(config, iteration, compliance, volume_fraction)
    return node_features, node_features_raw, edge_index, edge_attr, global_features


def build_graph_auxiliary_features(components, edge_index, config):
    """Build schema-v2 auxiliary graph fields without changing legacy features."""
    load_point, node_load_distance_norm, node_load_prior = build_load_prior_features(components, config)
    edge_load_prior = build_edge_load_prior(edge_index, node_load_prior)
    return {
        "feature_schema_version": np.asarray(int(getattr(config, "feature_schema_version", 2)), dtype=np.int64),
        "load_point": load_point,
        "node_load_distance_norm": node_load_distance_norm,
        "node_load_prior": node_load_prior,
        "edge_load_prior": edge_load_prior,
    }


def save_graph_npz(path, node_features, node_features_raw, edge_index, edge_attr, global_features, labels):
    """Save graph data in an npz format that is easy to load with PyG."""
    payload = {
        "node_features": node_features,
        "node_features_raw": node_features_raw,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "global_features": global_features,
    }
    payload.update(labels)
    np.savez(path, **payload)
