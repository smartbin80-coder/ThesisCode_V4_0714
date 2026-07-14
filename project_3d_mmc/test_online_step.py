"""Single-step smoke test for online GAT-guided MMC step scaling."""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch

from config import config as base_config
from fem3d import generate_hex_mesh, hex8_element_stiffness
from loads_boundary import define_cantilever_boundary_conditions
from mmc3d_components import MMCComponent3D
from models import MMCStepGAT
from online_meta_controller import apply_component_step_scale, predict_component_eta
from optimizer import MMCOptimizer3D
from pyg_dataset import MMCStepDataset


PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = PROJECT_DIR.parent
DEFAULT_DATASET_DIR = WORKSPACE_DIR / "dataset_debug"
DEFAULT_RESULTS_DIR = WORKSPACE_DIR / "results_debug" / "online_step"
DEFAULT_MAML_CHECKPOINT = WORKSPACE_DIR / "results_debug" / "gat_maml_model.pt"
DEFAULT_GAT_CHECKPOINT = WORKSPACE_DIR / "results_debug" / "gat_step_model.pt"


def components_from_params(params):
    """Build component templates from one flattened 9-parameter vector."""
    return [MMCComponent3D(*row) for row in np.asarray(params, dtype=float).reshape(-1, 9)]


def _sample_scalar(sample, key, default, cast=float):
    value = getattr(sample, key, None)
    if value is None:
        return default
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        value = value.view(-1)[0].item()
    return cast(value)


def build_optimizer_from_sample(sample, dataset_dir, results_dir, overrides=None):
    """Create the same FEM evaluator used by the optimizer for one graph sample."""
    overrides = overrides or {}
    cfg = copy.deepcopy(base_config)
    cfg.DL = float(overrides.get("DL") or _sample_scalar(sample, "DL", base_config.DL))
    cfg.DW = float(overrides.get("DW") or _sample_scalar(sample, "DW", base_config.DW))
    cfg.DH = float(overrides.get("DH") or _sample_scalar(sample, "DH", base_config.DH))
    cfg.nelx = int(overrides.get("nelx") or _sample_scalar(sample, "nelx", base_config.nelx, int))
    cfg.nely = int(overrides.get("nely") or _sample_scalar(sample, "nely", base_config.nely, int))
    cfg.nelz = int(overrides.get("nelz") or _sample_scalar(sample, "nelz", base_config.nelz, int))
    cfg.E0 = float(overrides.get("E0") or _sample_scalar(sample, "E0", base_config.E0))
    cfg.Emin = float(overrides.get("Emin") or _sample_scalar(sample, "Emin", base_config.Emin))
    cfg.nu = float(overrides.get("nu") or _sample_scalar(sample, "nu", base_config.nu))
    cfg.volfrac = float(overrides.get("volfrac") or _sample_scalar(sample, "volfrac", base_config.volfrac))
    cfg.max_iter = int(overrides.get("max_iter") or _sample_scalar(sample, "max_iter", base_config.max_iter, int))
    cfg.num_components = int(sample.params.numel() // 9)
    cfg.load_y = float(sample.load_point[1]) if hasattr(sample, "load_point") else float(base_config.load_y)
    cfg.load_z = float(sample.load_point[2]) if hasattr(sample, "load_point") else float(base_config.load_z)
    cfg.dataset_dir = str(dataset_dir)
    cfg.results_dir = str(results_dir)

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
    components = components_from_params(sample.params.detach().cpu().numpy())
    return MMCOptimizer3D(cfg, components, nodes, elements, fixed_dofs, force_vector, ke)


def load_model(checkpoint_path, sample, device):
    """Load a checkpoint and infer architecture values from saved args."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_args = checkpoint.get("args", {})
    model = MMCStepGAT(
        node_channels=sample.x.shape[1],
        edge_channels=sample.edge_attr.shape[1],
        hidden_channels=int(saved_args.get("hidden_channels", 64)),
        heads=int(saved_args.get("heads", 4)),
        num_layers=int(saved_args.get("num_layers", 2)),
        response_candidates=int(saved_args.get("response_candidates", 6)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model


def resolve_checkpoint(checkpoint_arg):
    """Resolve explicit checkpoint or fall back from MAML to supervised GAT."""
    if checkpoint_arg:
        checkpoint_path = Path(checkpoint_arg)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    for checkpoint_path in (DEFAULT_MAML_CHECKPOINT, DEFAULT_GAT_CHECKPOINT):
        if checkpoint_path.exists():
            return checkpoint_path
    raise FileNotFoundError(
        "No checkpoint found. Expected one of: "
        f"{DEFAULT_MAML_CHECKPOINT} or {DEFAULT_GAT_CHECKPOINT}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run one online MMC GAT smoke-test step.")
    parser.add_argument("--dataset-dir", type=str, default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--nelx", type=int, default=None)
    parser.add_argument("--nely", type=int, default=None)
    parser.add_argument("--nelz", type=int, default=None)
    parser.add_argument("--Emin", type=float, default=None)
    return parser.parse_args()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    results_dir = Path(args.results_dir)
    dataset = MMCStepDataset(dataset_dir)
    if len(dataset) == 0:
        raise RuntimeError(f"No graph samples found in {dataset_dir}")
    checkpoint_path = resolve_checkpoint(args.checkpoint)

    sample = dataset.get(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, sample, device)

    eta_pred, _ = predict_component_eta(model, sample, device=device)
    params_old = sample.params.detach().cpu().numpy()
    params_candidate = params_old + sample.delta_params.detach().cpu().numpy()
    trial_params = apply_component_step_scale(params_old, params_candidate, eta_pred)

    overrides = {key: value for key, value in vars(args).items() if key in {"nelx", "nely", "nelz", "Emin"} and value is not None}
    optimizer = build_optimizer_from_sample(sample, dataset_dir=dataset_dir, results_dir=results_dir, overrides=overrides)
    compliance, volume_fraction, *_ = optimizer.evaluate(trial_params)

    print(f"加载模型: {checkpoint_path}")
    print("预测的组件步长:", np.array2string(eta_pred, precision=4, separator=", "))
    print(f"试探后的柔度值: {float(compliance):.6e}")
    print(f"试探后的体积分数: {float(volume_fraction):.6e}")


if __name__ == "__main__":
    main()
