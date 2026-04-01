"""
src/classifier/verif_classifier.py
-------------------------------------
Final misinformation classifier combining CLIP + GNN + cluster features.
This is the SML phase of VerifAI.
TODO: Try focal loss to handle class imbalance in misinformation datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VerifAIClassifier(nn.Module):
    """
    Combines CLIP embeddings, GNN propagation features, and cluster IDs
    into a final misinformation probability score.

    Input:  [clip_fused_emb || gnn_emb || cluster_one_hot]
    Output: probability of being misinformation
    """

    def __init__(self, clip_dim=1024, gnn_dim=256, num_clusters=50, hidden_dim=128, dropout=0.3):
        super().__init__()

        input_dim = clip_dim + gnn_dim + num_clusters

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 1),  # Binary output
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, clip_emb, gnn_emb, cluster_one_hot):
        """
        Args:
            clip_emb:        [B, clip_dim]   — fused image+text CLIP embedding
            gnn_emb:         [B, gnn_dim]    — propagation-aware GNN embedding
            cluster_one_hot: [B, num_clusters] — which narrative cluster this post belongs to
        Returns:
            logits: [B, 1]
            probs:  [B, 1]  — misinformation probability
        """
        x = torch.cat([clip_emb, gnn_emb, cluster_one_hot], dim=-1)
        logits = self.classifier(x)
        probs = torch.sigmoid(logits)
        return logits, probs


class FocalLoss(nn.Module):
    """
    Focal Loss — handles class imbalance better than BCE.
    Misinformation datasets are often imbalanced (more real than fake).
    TODO: Tune alpha and gamma on your specific dataset.
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


def compute_metrics(preds, labels, threshold=0.5):
    """
    Compute F1, AUC-ROC, and accuracy.
    TODO: Add per-class metrics and confusion matrix logging.
    """
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

    binary_preds = (preds >= threshold).astype(int)
    return {
        "f1": f1_score(labels, binary_preds),
        "auc_roc": roc_auc_score(labels, preds),
        "accuracy": accuracy_score(labels, binary_preds),
    }
