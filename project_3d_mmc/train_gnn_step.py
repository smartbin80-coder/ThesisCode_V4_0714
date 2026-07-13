"""Offline supervised training for MMC GAT step prediction."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

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


def compute_step_losses(outputs, data, response_weight=0.1, class_weight=0.1, graph_weight=0.2):
    """Compute node, graph, class, and response supervision losses."""
    losses = {}
    losses["node_eta"] = F.mse_loss(outputs["node_eta_pred"], data.eta_node_label.float())
    losses["graph_eta"] = F.mse_loss(outputs["graph_eta_pred"], data.eta_label.view(-1).float())

    node_label_index = data.eta_node_label_index.long().clamp(min=0, max=outputs["node_eta_logits"].size(1) - 1)
    graph_label_index = data.eta_label_index.view(-1).long().clamp(min=0, max=outputs["graph_eta_logits"].size(1) - 1)
    losses["node_class"] = F.cross_entropy(outputs["node_eta_logits"], node_label_index)
    losses["graph_class"] = F.cross_entropy(outputs["graph_eta_logits"], graph_label_index)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device) if val_loader is not None else train_loss
        if val_loss <= best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(), "args": vars(args)}, Path(args.checkpoint))
        print(f"epoch={epoch:04d} train_loss={train_loss:.6e} val_loss={val_loss:.6e}")


if __name__ == "__main__":
    main()
