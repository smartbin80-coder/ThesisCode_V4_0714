"""MAML-style meta-learning utilities for MMC step prediction."""

import argparse
import copy
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch_geometric.data import Batch

from config import config as base_config
from models import MMCStepGAT
from pyg_dataset import MMCStepDataset
from train_gnn_step import build_optimizer, compute_step_losses, set_alpha_trainable


PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = PROJECT_DIR.parent


def group_indices_by_trajectory(dataset):
    """Group dataset indices by trajectory id for task construction."""
    groups = defaultdict(list)
    for idx, path in enumerate(dataset.files):
        name = Path(path).name
        if name.startswith("traj_"):
            trajectory_id = "_".join(name.split("_")[:2])
        else:
            trajectory_id = "default"
        groups[trajectory_id].append(idx)
    return {key: value for key, value in groups.items() if len(value) >= 2}


def sample_support_query(groups, support_size=4, query_size=4):
    """Randomly sample support/query step indices within one trajectory task."""
    trajectory_id = random.choice(list(groups.keys()))
    indices = groups[trajectory_id][:]
    random.shuffle(indices)
    need = support_size + query_size
    if len(indices) < need:
        indices = (indices * ((need // len(indices)) + 1))[:need]
    support = indices[:support_size]
    query = indices[support_size : support_size + query_size]
    return trajectory_id, support, query


def make_batch(dataset, indices, device):
    """Load selected graph steps and collate them into a PyG batch."""
    return Batch.from_data_list([dataset.get(i) for i in indices]).to(device)


def regularization_loss(model):
    """Small L2 regularization used by the explicit meta objective."""
    reg = None
    for param in model.parameters():
        term = torch.sum(param * param)
        reg = term if reg is None else reg + term
    return reg if reg is not None else torch.tensor(0.0)


def adapt_on_support(model, support_batch, inner_lr=1e-3, inner_steps=2, alpha_trainable=True, alpha_weight_decay=1e-4):
    """Adapt a copied model on support data for first-order MAML training."""
    adapted = copy.deepcopy(model)
    set_alpha_trainable(adapted, alpha_trainable)
    optimizer = build_optimizer(adapted, inner_lr, 0.0, optimizer_cls=torch.optim.SGD)
    for _ in range(inner_steps):
        optimizer.zero_grad()
        support_outputs = adapted(support_batch)
        support_loss, _ = compute_step_losses(support_outputs, support_batch)
        support_loss.backward()
        optimizer.step()
    return adapted


def first_order_maml_epoch(
    model,
    dataset,
    groups,
    optimizer,
    device,
    tasks_per_epoch=8,
    support_size=4,
    query_size=4,
    inner_lr=1e-3,
    inner_steps=2,
    lambda_reg=1e-5,
    alpha_trainable=True,
    alpha_weight_decay=1e-4,
):
    """Run one first-order MAML-style epoch with explicit support/query loss."""
    model.train()
    total = 0.0
    for _ in range(tasks_per_epoch):
        _, support_idx, query_idx = sample_support_query(groups, support_size=support_size, query_size=query_size)
        support_batch = make_batch(dataset, support_idx, device)
        query_batch = make_batch(dataset, query_idx, device)

        adapted = adapt_on_support(
            model,
            support_batch,
            inner_lr=inner_lr,
            inner_steps=inner_steps,
            alpha_trainable=alpha_trainable,
            alpha_weight_decay=alpha_weight_decay,
        )
        support_outputs = model(support_batch)
        support_loss, _ = compute_step_losses(support_outputs, support_batch)
        query_outputs = adapted(query_batch)
        query_loss, _ = compute_step_losses(query_outputs, query_batch)
        reg = regularization_loss(model)
        meta_loss = support_loss + query_loss + lambda_reg * reg

        optimizer.zero_grad()
        meta_loss.backward()
        optimizer.step()

        # First-order pull toward the query-adapted parameters.
        with torch.no_grad():
            for base_param, adapted_param in zip(model.parameters(), adapted.parameters()):
                base_param.add_(0.1 * (adapted_param - base_param))
        total += float((support_loss.detach() + query_loss.detach()).cpu())
    return total / max(tasks_per_epoch, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="MAML-style meta-training for MMC GAT step predictor.")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--tasks-per-epoch", type=int, default=8)
    parser.add_argument("--support-size", type=int, default=4)
    parser.add_argument("--query-size", type=int, default=4)
    parser.add_argument("--inner-lr", type=float, default=1e-3)
    parser.add_argument("--inner-steps", type=int, default=2)
    parser.add_argument("--meta-lr", type=float, default=5e-4)
    parser.add_argument("--lambda-reg", type=float, default=1e-5)
    parser.add_argument("--alpha-weight-decay", type=float, default=base_config.alpha_weight_decay)
    parser.add_argument("--alpha-freeze-epochs", type=int, default=base_config.alpha_freeze_epochs)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--response-candidates", type=int, default=6)
    parser.add_argument("--checkpoint", type=str, default=str(WORKSPACE_DIR / "results_debug" / "gat_maml_model.pt"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MMCStepDataset(args.dataset_dir)
    groups = group_indices_by_trajectory(dataset)
    if not groups:
        raise RuntimeError("No trajectory with at least two graph samples was found.")
    sample = dataset.get(0)
    model = MMCStepGAT(
        node_channels=sample.x.shape[1],
        edge_channels=sample.edge_attr.shape[1],
        hidden_channels=args.hidden_channels,
        heads=args.heads,
        num_layers=args.num_layers,
        response_candidates=args.response_candidates,
    ).to(device)
    optimizer = build_optimizer(model, args.meta_lr, args.alpha_weight_decay)
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        alpha_trainable = epoch > args.alpha_freeze_epochs
        set_alpha_trainable(model, alpha_trainable)
        loss = first_order_maml_epoch(
            model,
            dataset,
            groups,
            optimizer,
            device,
            tasks_per_epoch=args.tasks_per_epoch,
            support_size=args.support_size,
            query_size=args.query_size,
            inner_lr=args.inner_lr,
            inner_steps=args.inner_steps,
            lambda_reg=args.lambda_reg,
            alpha_trainable=alpha_trainable,
            alpha_weight_decay=args.alpha_weight_decay,
        )
        torch.save({"model_state": model.state_dict(), "args": vars(args)}, args.checkpoint)
        print(f"epoch={epoch:04d} meta_loss={loss:.6e}")


if __name__ == "__main__":
    main()
