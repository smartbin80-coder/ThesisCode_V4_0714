"""Batch trajectory generator for GNN step-size datasets."""

import argparse
import copy
import csv
import tempfile
from pathlib import Path

import numpy as np

from config import config as base_config
from fem3d import generate_hex_mesh, hex8_element_stiffness
from loads_boundary import define_cantilever_boundary_conditions
from mmc3d_components import create_random_components
from optimizer import MMCOptimizer3D
from utils import ensure_dir, set_random_seed


def writable_dir(path, fallback_name):
    """Return a writable directory, falling back to a temp location if needed."""
    candidates = [Path(path), Path(__file__).resolve().parent.parent / Path(path).name, Path(tempfile.gettempdir()) / "project_3d_mmc" / fallback_name]
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return candidate
        except OSError:
            continue
    raise OSError(f"No writable directory found for {path}")


def run_single_trajectory(cfg):
    """Run one randomized trajectory and export graph npz files."""
    set_random_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    nodes, elements = generate_hex_mesh(cfg.DL, cfg.DW, cfg.DH, cfg.nelx, cfg.nely, cfg.nelz)
    fixed_dofs, force_vector = define_cantilever_boundary_conditions(
        nodes,
        cfg.DL,
        cfg.DW,
        cfg.DH,
        load_y=cfg.load_y,
        load_z=cfg.load_z,
    )
    ke = hex8_element_stiffness(cfg.E0, cfg.nu, cfg.DL / cfg.nelx, cfg.DW / cfg.nely, cfg.DH / cfg.nelz)
    components = create_random_components(cfg, rng)
    optimizer = MMCOptimizer3D(cfg, components, nodes, elements, fixed_dofs, force_vector, ke)
    result, final_eval = optimizer.run()
    return {
        "trajectory_id": cfg.trajectory_id,
        "seed": cfg.seed,
        "num_components": cfg.num_components,
        "load_y": cfg.load_y,
        "load_z": cfg.load_z,
        "num_graphs": optimizer.iteration,
        "final_compliance": float(final_eval[0]),
        "final_volume_fraction": float(final_eval[1]),
        "success": bool(getattr(result, "success", False)),
    }


def build_config_for_trajectory(args, trajectory_index):
    """Create a trajectory-specific config copy."""
    cfg = copy.deepcopy(base_config)
    cfg.dataset_dir = str(Path(args.output_dir))
    cfg.results_dir = str(Path(args.results_dir) / f"traj_{trajectory_index:04d}")
    cfg.seed = int(args.seed_start + trajectory_index)
    rng = np.random.default_rng(cfg.seed)
    cfg.trajectory_id = f"traj_{trajectory_index:04d}"
    cfg.max_iter = int(args.max_iter)
    cfg.save_density = bool(args.save_density)
    cfg.save_graph = True
    cfg.num_components = int(rng.integers(args.min_components, args.max_components + 1))
    cfg.load_y = float(rng.uniform(args.load_y_min * cfg.DW, args.load_y_max * cfg.DW))
    cfg.load_z = float(rng.uniform(args.load_z_min * cfg.DH, args.load_z_max * cfg.DH))
    cfg.max_components_for_dataset = int(args.max_components)
    return cfg


def write_index(index_path, rows):
    """Write trajectory metadata for leakage-safe splitting."""
    fieldnames = [
        "trajectory_id",
        "seed",
        "num_components",
        "load_y",
        "load_z",
        "num_graphs",
        "final_compliance",
        "final_volume_fraction",
        "success",
    ]
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _array_to_cell(values):
    """Serialize a 1D array into a compact CSV cell."""
    arr = np.asarray(values).ravel()
    return ";".join(f"{float(v):.10g}" for v in arr)


def write_graph_index(dataset_dir, index_path=None):
    """Write one CSV row per graph snapshot for Excel inspection."""
    dataset_dir = Path(dataset_dir)
    if index_path is None:
        index_path = dataset_dir / "graph_index.csv"
    else:
        index_path = Path(index_path)

    fieldnames = [
        "trajectory_id",
        "iteration",
        "file",
        "seed",
        "num_components",
        "node_count",
        "edge_count",
        "load_y",
        "load_z",
        "compliance",
        "volume_fraction",
        "constraint_margin",
        "eta_label",
        "eta_label_index",
        "eta_failure_flag",
        "eta_feasible_count",
        "eta_candidates",
        "eta_feasible_mask",
        "compliance_candidates",
        "volume_candidates",
        "feature_schema_version",
        "mean_node_load_prior",
        "max_component_strain_energy_norm",
        "trust_actual_delta",
        "trust_predicted_delta",
        "trust_bias",
    ]

    rows = []
    for graph_path in sorted(dataset_dir.glob("*_graph.npz")):
        with np.load(graph_path) as data:
            volume_fraction = float(data["volume_fraction"])
            volfrac = float(data["global_features"][3]) if "global_features" in data else 0.4
            feasible_mask = np.asarray(data["eta_feasible_mask"], dtype=int) if "eta_feasible_mask" in data else np.zeros(0, dtype=int)
            rows.append(
                {
                    "trajectory_id": str(data["trajectory_id"]) if "trajectory_id" in data else "",
                    "iteration": int(data["iteration"]),
                    "file": graph_path.name,
                    "seed": int(data["seed"]) if "seed" in data else "",
                    "num_components": int(data["num_components"]) if "num_components" in data else data["node_features"].shape[0],
                    "node_count": int(data["node_features"].shape[0]),
                    "edge_count": int(data["edge_index"].shape[1]),
                    "load_y": float(data["load_y"]) if "load_y" in data else "",
                    "load_z": float(data["load_z"]) if "load_z" in data else "",
                    "compliance": float(data["compliance"]),
                    "volume_fraction": volume_fraction,
                    "constraint_margin": volfrac - volume_fraction,
                    "eta_label": float(data["eta_label"]),
                    "eta_label_index": int(data["eta_label_index"]) if "eta_label_index" in data else "",
                    "eta_failure_flag": int(data["eta_failure_flag"]) if "eta_failure_flag" in data else "",
                    "eta_feasible_count": int(np.sum(feasible_mask)) if feasible_mask.size else "",
                    "eta_candidates": _array_to_cell(data["eta_candidates"]),
                    "eta_feasible_mask": ";".join(str(int(v)) for v in feasible_mask),
                    "compliance_candidates": _array_to_cell(data["compliance_candidates"]),
                    "volume_candidates": _array_to_cell(data["volume_candidates"]),
                    "feature_schema_version": int(data["feature_schema_version"]) if "feature_schema_version" in data else 1,
                    "mean_node_load_prior": float(np.mean(data["node_load_prior"])) if "node_load_prior" in data else "",
                    "max_component_strain_energy_norm": float(np.max(data["component_strain_energy_norm"])) if "component_strain_energy_norm" in data else "",
                    "trust_actual_delta": float(data["trust_actual_delta"]) if "trust_actual_delta" in data else "",
                    "trust_predicted_delta": float(data["trust_predicted_delta"]) if "trust_predicted_delta" in data else "",
                    "trust_bias": float(data["trust_bias"]) if "trust_bias" in data else "",
                }
            )

    with open(index_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return index_path


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate batched 3D-MMC GNN trajectories.")
    parser.add_argument("--num-trajectories", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--min-components", type=int, default=6)
    parser.add_argument("--max-components", type=int, default=12)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="dataset")
    parser.add_argument("--results-dir", type=str, default="results/batch")
    parser.add_argument("--save-density", action="store_true")
    parser.add_argument("--load-y-min", type=float, default=0.25)
    parser.add_argument("--load-y-max", type=float, default=0.75)
    parser.add_argument("--load-z-min", type=float, default=0.25)
    parser.add_argument("--load-z-max", type=float, default=0.75)
    return parser.parse_args()


def main():
    """Generate multiple independent optimization trajectories."""
    args = parse_args()
    args.output_dir = str(writable_dir(args.output_dir, "dataset"))
    args.results_dir = str(writable_dir(args.results_dir, "results"))
    rows = []
    for trajectory_index in range(args.num_trajectories):
        cfg = build_config_for_trajectory(args, trajectory_index)
        print(
            f"trajectory {cfg.trajectory_id} | seed={cfg.seed} | "
            f"components={cfg.num_components} | load=({cfg.load_y:.3f}, {cfg.load_z:.3f})"
        )
        rows.append(run_single_trajectory(cfg))
        write_index(Path(args.output_dir) / "dataset_index.csv", rows)
        write_graph_index(args.output_dir)
    print(f"Dataset generation finished: {args.output_dir}")


if __name__ == "__main__":
    main()
