"""
src/gnn/propagation_gnn.py
-----------------------------
Graph Neural Network that models how misinformation spreads.
Posts = nodes, shared hashtags/users/retweets = edges.
This is the NOVEL part of VerifAI — very few papers combine CLIP + GNN.
TODO: Experiment with GAT (Graph Attention) vs GCN vs GraphSAGE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch


class PropagationGNN(nn.Module):
    """
    Graph Attention Network that captures misinformation propagation patterns.
    Input: CLIP embeddings as node features + social graph structure.
    Output: Graph-aware post embeddings enriched with propagation context.
    """

    def __init__(self, input_dim, hidden_dim=256, num_layers=3, dropout=0.3, heads=4):
        super().__init__()
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.convs.append(GATConv(input_dim, hidden_dim // heads, heads=heads, dropout=dropout))
        self.norms.append(nn.LayerNorm(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Final layer (single head for clean output)
        self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout))
        self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x: Node features [N, input_dim] (CLIP embeddings)
            edge_index: Graph edges [2, E]
            batch: Batch vector for graph pooling [N]
        Returns:
            Node-level embeddings [N, hidden_dim]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.norms[i](x)
            x = F.elu(x)
            x = self.dropout(x)

        return x


def build_social_graph(post_ids, shared_hashtags, shared_users):
    """
    Constructs a social graph from post metadata.
    Two posts are connected if they share hashtags or are from the same user.

    Args:
        post_ids: list of post IDs
        shared_hashtags: dict {post_id: [hashtag1, hashtag2, ...]}
        shared_users: dict {post_id: user_id}

    Returns:
        edge_index: torch.Tensor [2, E]

    TODO: Add edge weights based on similarity strength.
    TODO: Add temporal edges (posts close in time are connected).
    """
    edges = set()
    id_to_idx = {pid: i for i, pid in enumerate(post_ids)}

    # Connect posts sharing hashtags
    hashtag_to_posts = {}
    for pid, tags in shared_hashtags.items():
        for tag in tags:
            hashtag_to_posts.setdefault(tag, []).append(pid)

    for tag, posts in hashtag_to_posts.items():
        for i in range(len(posts)):
            for j in range(i + 1, len(posts)):
                u, v = id_to_idx[posts[i]], id_to_idx[posts[j]]
                edges.add((u, v))
                edges.add((v, u))  # undirected

    # Connect posts from same user
    user_to_posts = {}
    for pid, uid in shared_users.items():
        user_to_posts.setdefault(uid, []).append(pid)

    for uid, posts in user_to_posts.items():
        for i in range(len(posts)):
            for j in range(i + 1, len(posts)):
                u, v = id_to_idx[posts[i]], id_to_idx[posts[j]]
                edges.add((u, v))
                edges.add((v, u))

    if not edges:
        # Fallback: no edges — return empty graph
        return torch.zeros((2, 0), dtype=torch.long)

    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    return edge_index
