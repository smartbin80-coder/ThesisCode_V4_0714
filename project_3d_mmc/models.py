"""Physics-biased GAT models for MMC component step prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import softmax


class PhysicsBiasedGATLayer(nn.Module):
    """GAT layer with a learnable scale for load-distance attention bias."""

    def __init__(self, in_channels, out_channels, edge_channels=10, heads=4, dropout=0.0, prior_alpha_init=0.1):
        super().__init__()
        if out_channels % heads != 0:
            raise ValueError("out_channels must be divisible by heads")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.edge_channels = int(edge_channels)
        self.heads = int(heads)
        self.head_dim = self.out_channels // self.heads
        self.dropout = float(dropout)

        self.node_proj = nn.Linear(self.in_channels, self.out_channels, bias=False)
        self.edge_proj = nn.Linear(self.edge_channels, self.out_channels, bias=False)
        self.att_src = nn.Parameter(torch.empty(self.heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.empty(self.heads, self.head_dim))
        self.att_edge = nn.Parameter(torch.empty(self.heads, self.head_dim))
        self.bias = nn.Parameter(torch.zeros(self.out_channels))
        self.alpha = nn.Parameter(torch.tensor(float(prior_alpha_init), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize trainable attention parameters."""
        nn.init.xavier_uniform_(self.node_proj.weight)
        nn.init.xavier_uniform_(self.edge_proj.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.xavier_uniform_(self.att_edge)
        nn.init.zeros_(self.bias)
        with torch.no_grad():
            self.alpha.fill_(0.1)

    def forward(self, x, edge_index, edge_attr, edge_load_prior=None):
        """Apply physics-biased attention message passing."""
        num_nodes = x.size(0)
        if edge_index.numel() == 0:
            return self.node_proj(x) + self.bias

        src, dst = edge_index
        h = self.node_proj(x).view(num_nodes, self.heads, self.head_dim)
        e = self.edge_proj(edge_attr).view(edge_attr.size(0), self.heads, self.head_dim)

        learned_logit = (
            (h[src] * self.att_src).sum(dim=-1)
            + (h[dst] * self.att_dst).sum(dim=-1)
            + (e * self.att_edge).sum(dim=-1)
        )
        learned_logit = F.leaky_relu(learned_logit, negative_slope=0.2)

        if edge_load_prior is None:
            edge_load_prior = edge_attr.new_zeros((edge_attr.size(0), 1))
        prior_bias = edge_load_prior.view(-1, 1).to(dtype=learned_logit.dtype, device=learned_logit.device)
        attention_logit = learned_logit + self.alpha * prior_bias
        attn = softmax(attention_logit, dst, num_nodes=num_nodes)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        messages = (h[src] + e) * attn.unsqueeze(-1)
        out = x.new_zeros((num_nodes, self.heads, self.head_dim))
        out.index_add_(0, dst, messages)
        return out.reshape(num_nodes, self.out_channels) + self.bias


class MMCStepGAT(nn.Module):
    """Predict component-level eta and graph-level auxiliary responses."""

    def __init__(
        self,
        node_channels=20,
        edge_channels=10,
        hidden_channels=64,
        heads=4,
        num_layers=2,
        dropout=0.1,
        global_channels=7,
        response_candidates=6,
    ):
        super().__init__()
        self.response_candidates = int(response_candidates)
        self.global_channels = int(global_channels)
        self.dropout = float(dropout)

        layers = []
        in_channels = int(node_channels)
        for _ in range(int(num_layers)):
            layers.append(
                PhysicsBiasedGATLayer(
                    in_channels,
                    hidden_channels,
                    edge_channels=edge_channels,
                    heads=heads,
                    dropout=dropout,
                    prior_alpha_init=0.1,
                )
            )
            in_channels = int(hidden_channels)
        self.gat_layers = nn.ModuleList(layers)
        self.node_norms = nn.ModuleList(nn.LayerNorm(hidden_channels) for _ in layers)
        self.node_eta_head = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, 1))
        self.node_class_head = nn.Linear(hidden_channels, self.response_candidates)
        self.graph_eta_head = nn.Sequential(
            nn.Linear(hidden_channels + self.global_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )
        self.graph_class_head = nn.Linear(hidden_channels + self.global_channels, self.response_candidates)
        self.response_head = nn.Sequential(
            nn.Linear(hidden_channels + self.global_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, self.response_candidates * 2),
        )

    @staticmethod
    def eta_from_raw(raw_eta):
        """Map unconstrained logits to the physical eta range [0.25, 1.5]."""
        return 0.25 + 1.25 * torch.sigmoid(raw_eta)

    def _global_features(self, global_features, num_graphs, device):
        if global_features is None:
            return torch.zeros((num_graphs, self.global_channels), dtype=torch.float32, device=device)
        global_features = global_features.to(device=device, dtype=torch.float32)
        if global_features.dim() == 1:
            global_features = global_features.reshape(num_graphs, -1)
        if global_features.size(1) < self.global_channels:
            pad = torch.zeros((num_graphs, self.global_channels - global_features.size(1)), device=device)
            global_features = torch.cat([global_features, pad], dim=1)
        global_features = global_features[:, : self.global_channels].clone()
        if global_features.size(1) > 1:
            global_features[:, 1] = torch.log1p(torch.clamp(global_features[:, 1], min=0.0))
        return global_features

    def forward(self, data):
        """Return node eta, graph eta, and candidate response predictions."""
        x = data.x
        edge_load_prior = getattr(data, "edge_load_prior", None)
        for layer, norm in zip(self.gat_layers, self.node_norms):
            residual = x
            x = layer(x, data.edge_index, data.edge_attr, edge_load_prior=edge_load_prior)
            x = norm(F.elu(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            if residual.shape == x.shape:
                x = x + residual

        raw_node_eta = self.node_eta_head(x).squeeze(-1)
        node_eta_pred = self.eta_from_raw(raw_node_eta)

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        pooled = global_mean_pool(x, batch)
        global_features = self._global_features(getattr(data, "global_features", None), num_graphs, x.device)
        graph_state = torch.cat([pooled, global_features], dim=1)
        graph_eta_pred = self.eta_from_raw(self.graph_eta_head(graph_state).squeeze(-1))
        graph_eta_logits = self.graph_class_head(graph_state)
        node_eta_logits = self.node_class_head(x)
        response_pred = self.response_head(graph_state).view(num_graphs, self.response_candidates, 2)
        return {
            "node_eta_pred": node_eta_pred,
            "graph_eta_pred": graph_eta_pred,
            "node_eta_logits": node_eta_logits,
            "graph_eta_logits": graph_eta_logits,
            "response_pred": response_pred,
            "node_embedding": x,
        }
