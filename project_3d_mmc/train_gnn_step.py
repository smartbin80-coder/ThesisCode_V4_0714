"""Offline supervised training for MMC GAT step prediction."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from config import config as base_config
from models import MMCStepGAT
from pyg_dataset import create_dataloader


def _graph_count(data):
    batch = getattr(data, "batch", None)
    if batch is None:
        return 1
    return int(batch.max().item()) + 1 if batch.numel() else 1


def normalize_response_targets(response_targets):
    """Scale raw candidate compliance/volume targets for stable training."""
    out = response_targets.clone()
    out[..., 0] = torch.log1p(torch.clamp(out[..., 0], min=0.0))
    return out


def _zero_loss_like(tensor):
    """Return a differentiable zero on the same device as tensor."""
    return tensor.sum() * 0.0


def _classification_losses(outputs, data):
    """Compute class losses only for graphs whose candidate eta search succeeded."""
    device = outputs["node_eta_logits"].device
    failure_flag = getattr(data, "eta_failure_flag", None)
    if failure_flag is None:
        graph_valid_mask = torch.ones(outputs["graph_eta_logits"].size(0), dtype=torch.bool, device=device)
    else:
        graph_valid_mask = failure_flag.view(-1).to(device=device).long() == 0

    batch = getattr(data, "batch", None)
    if batch is None:
        node_valid_mask = graph_valid_mask
    else:
        node_valid_mask = graph_valid_mask[batch.to(device=device)]

    if torch.any(node_valid_mask):
        node_label_index = data.eta_node_label_index.to(device=device).long()[node_valid_mask]
        node_label_index = node_label_index.clamp(min=0, max=outputs["node_eta_logits"].size(1) - 1)
        node_class = F.cross_entropy(outputs["node_eta_logits"][node_valid_mask], node_label_index)
    else:
        node_class = _zero_loss_like(outputs["node_eta_logits"])

    if torch.any(graph_valid_mask):
        graph_label_index = data.eta_label_index.view(-1).to(device=device).long()[graph_valid_mask]
        graph_label_index = graph_label_index.clamp(min=0, max=outputs["graph_eta_logits"].size(1) - 1)
        graph_class = F.cross_entropy(outputs["graph_eta_logits"][graph_valid_mask], graph_label_index)
    else:
        graph_class = _zero_loss_like(outputs["graph_eta_logits"])

    return node_class, graph_class


def compute_step_losses(outputs, data, response_weight=0.1, class_weight=0.1, graph_weight=0.2):
    """Compute node, graph, class, and response supervision losses."""
    losses = {}
    losses["node_eta"] = F.mse_loss(outputs["node_eta_pred"], data.eta_node_label.float())
    losses["graph_eta"] = F.mse_loss(outputs["graph_eta_pred"], data.eta_label.view(-1).float())

    losses["node_class"], losses["graph_class"] = _classification_losses(outputs, data)

    num_graphs = _graph_count(data)
    response_pred = outputs["response_pred"]
    response_targets = data.response_targets.float()
    expected_rows = num_graphs * response_pred.size(1)
    if response_targets.numel() and response_targets.shape == (expected_rows, 2):
        response_targets = response_targets.view(num_graphs, response_pred.size(1), 2)
        response_targets = normalize_response_targets(response_targets)
        losses["response"] = F.mse_loss(response_pred, response_targets)
    else:
        losses["response"] = response_pred.sum() * 0.0

    total = (
        losses["node_eta"]
        + graph_weight * losses["graph_eta"]
        + class_weight * (losses["node_class"] + losses["graph_class"])
        + response_weight * losses["response"]
    )
    return total, losses


def alpha_parameters(model):
    """Return all physics-prior alpha parameters in GAT layers."""
    return [param for name, param in model.named_parameters() if name.endswith(".alpha")]


def non_alpha_parameters(model):
    """Return trainable parameters except physics-prior alpha scalars."""
    alpha_ids = {id(param) for param in alpha_parameters(model)}
    return [param for param in model.parameters() if id(param) not in alpha_ids]


def set_alpha_trainable(model, trainable):
    """Freeze or release physics-prior alpha parameters."""
    for param in alpha_parameters(model):
        param.requires_grad_(bool(trainable))


def build_optimizer(model, lr, alpha_weight_decay, optimizer_cls=torch.optim.Adam):
    """Build optimizer with a dedicated weight-decay group for alpha."""
    return optimizer_cls(
        [
            {"params": non_alpha_parameters(model), "lr": lr},
            {"params": alpha_parameters(model), "lr": lr, "weight_decay": float(alpha_weight_decay)},
        ]
    )


def train_epoch(model, loader, optimizer, device):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss, _ = compute_step_losses(outputs, data)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate average supervised loss."""
    model.eval()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        outputs = model(data)
        loss, _ = compute_step_losses(outputs, data)
        total_loss += float(loss.detach().cpu())
    return total_loss / max(len(loader), 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train physics-biased GAT for MMC step prediction.")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--split-file", type=str, default=None)
    parser.add_argument("--val-split-file", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha-weight-decay", type=float, default=base_config.alpha_weight_decay)
    parser.add_argument("--alpha-freeze-epochs", type=int, default=base_config.alpha_freeze_epochs)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--response-candidates", type=int, default=6)
    parser.add_argument("--checkpoint", type=str, default="gat_step_model.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = create_dataloader(args.dataset_dir, batch_size=args.batch_size, shuffle=True, split_file=args.split_file)
    val_loader = None
    if args.val_split_file:
        val_loader = create_dataloader(args.dataset_dir, batch_size=args.batch_size, shuffle=False, split_file=args.val_split_file)

    sample = train_loader.dataset.get(0)
    model = MMCStepGAT(
        node_channels=sample.x.shape[1],
        edge_channels=sample.edge_attr.shape[1],
        hidden_channels=args.hidden_channels,
        heads=args.heads,
        num_layers=args.num_layers,
        response_candidates=args.response_candidates,
    ).to(device)
    optimizer = build_optimizer(model, args.lr, args.alpha_weight_decay)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        set_alpha_trainable(model, epoch > args.alpha_freeze_epochs)
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device) if val_loader is not None else train_loss
        if val_loss <= best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(), "args": vars(args)}, Path(args.checkpoint))
        print(f"epoch={epoch:04d} train_loss={train_loss:.6e} val_loss={val_loss:.6e}")


if __name__ == "__main__":
    main()
