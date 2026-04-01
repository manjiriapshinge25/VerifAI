"""
train.py
---------
Main training script for VerifAI.
Run: python train.py --config configs/config.yaml

Set data.mode in config.yaml to either:
  "synthetic" → uses your synthetic notebook's approach (no real images needed)
  "real"      → uses NewsCLIPpings or any real dataset with image_path column
"""

import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import os

from src.clustering.narrative_clusterer import NarrativeClusterer
from src.gnn.propagation_gnn import PropagationGNN, build_social_graph
from src.classifier.verif_classifier import VerifAIClassifier, FocalLoss, compute_metrics


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Dataset loaders ───────────────────────────────────────────────────────

def load_synthetic_dataset(n_train=800, n_val=200, n_test=200):
    """
    Generates synthetic news posts — same as your Jupyter notebook Section 1.
    No real images or downloads needed.
    """
    import random

    templates = [
        "Breaking news: {} spotted at {} event causing major controversy",
        "Scientists discover {} linked to {} in shocking new study",
        "Government confirms new {} policy affecting {} citizens nationwide",
        "Celebrity {} caught {} in leaked documents scandal",
        "Health officials warn {} may cause {} according to new research",
        "Study shows {} reduces {} by 40% claim researchers",
        "Local {} reports {} after mysterious incident downtown",
    ]
    fake_topics   = ["vaccines", "5G towers", "microchips", "fluoride", "chemtrails"]
    real_topics   = ["climate policy", "economic data", "election results", "sports"]
    fake_hashtags = [["#hoax", "#truth"], ["#wakeup", "#conspiracy"], ["#fakevirus"]]
    real_hashtags = [["#news", "#breaking"], ["#facts", "#science"], ["#economy"]]

    def make_rows(n, label, seed):
        np.random.seed(seed)
        random.seed(seed)
        rows = []
        topics = fake_topics if label == 1 else real_topics
        hashtags_pool = fake_hashtags if label == 1 else real_hashtags
        for i in range(n):
            t = random.choice(templates).format(
                random.choice(topics), random.choice(topics)
            )
            rows.append({
                "post_id":    f"{'fake' if label==1 else 'real'}_{seed}_{i}",
                "image_path": None,   # no real images in synthetic mode
                "text":       t,
                "label":      label,
                "hashtags":   random.choice(hashtags_pool),
                "user_id":    f"user_{np.random.randint(0, 50)}",
            })
        return rows

    def make_split(n, seed):
        half = n // 2
        rows = make_rows(half, 0, seed) + make_rows(half, 1, seed + 1000)
        np.random.shuffle(rows)
        return pd.DataFrame(rows)

    df_train = make_split(n_train, 42)
    df_val   = make_split(n_val,   1)
    df_test  = make_split(n_test,  2)
    df_all   = pd.concat([df_train, df_val, df_test], ignore_index=True)

    print(f"Synthetic dataset: {len(df_all)} samples "
          f"| Fake: {df_all.label.sum()} | Real: {(df_all.label==0).sum()}")
    return df_all


def load_real_dataset(config):
    """
    TODO: Load your real dataset here (NewsCLIPpings, MediaEval, etc.)
    Expected columns: [image_path, text, label, post_id, hashtags, user_id]
    """
    import json
    json_path = config["data"]["annotation_path"]
    with open(json_path) as f:
        data = json.load(f)

    rows = []
    for item in data["annotations"]:
        image_path = os.path.join(config["data"]["image_dir"], item["image_path"])
        if not os.path.exists(image_path):
            continue
        rows.append({
            "post_id":    str(item["id"]),
            "image_path": image_path,
            "text":       item["caption"],
            "label":      1 if item["falsified"] else 0,
            "hashtags":   [],
            "user_id":    "unknown",
        })
    return pd.DataFrame(rows)


# ── Embedding extraction ──────────────────────────────────────────────────

def extract_embeddings(df, config, device):
    mode = config["data"].get("mode", "synthetic")

    if mode == "synthetic":
        from src.embeddings.clip_embedder import extract_synthetic_embeddings
        return extract_synthetic_embeddings(df, embedding_dim=512)
    else:
        from src.embeddings.clip_embedder import CLIPEmbedder
        embedder = CLIPEmbedder(
            model_name=config["model"]["clip_model"], device=device
        )
        return embedder.extract_embeddings(
            df, batch_size=config["training"]["batch_size"]
        )


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args   = parser.parse_args()
    config = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[VerifAI] Device: {device}")
    print(f"[VerifAI] Mode  : {config['data'].get('mode', 'synthetic')}")

    os.makedirs("models",  exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # ── Step 1: Load data ─────────────────────────────────────────
    mode = config["data"].get("mode", "synthetic")
    if mode == "synthetic":
        df = load_synthetic_dataset()
    else:
        df = load_real_dataset(config)

    # ── Step 2: CLIP Embeddings ───────────────────────────────────
    print("\n[Phase 1] Extracting embeddings...")
    emb_dict = extract_embeddings(df, config, device)

    np.save("data/processed/fused_embs.npy", emb_dict["fused_embs"])
    np.save("data/processed/labels.npy",     emb_dict["labels"])

    # ── Step 3: Narrative Clustering ──────────────────────────────
    print("\n[Phase 2] Clustering narratives...")
    clusterer     = NarrativeClusterer(config)
    cluster_labels = clusterer.fit(emb_dict["fused_embs"])
    clusterer.evaluate(true_labels=emb_dict["labels"])
    clusterer.visualize(save_path="results/clusters_umap.png")
    np.save("data/processed/cluster_labels.npy", cluster_labels)

    # ── Step 4: Social Graph + GNN ───────────────────────────────
    print("\n[Phase 3] Building social graph + GNN...")
    edge_index = build_social_graph(
        post_ids=df["post_id"].tolist(),
        shared_hashtags=dict(zip(df["post_id"], df["hashtags"])),
        shared_users=dict(zip(df["post_id"], df["user_id"])),
    )

    gnn = PropagationGNN(
        input_dim=emb_dict["fused_embs"].shape[-1],
        hidden_dim=config["model"]["gnn_hidden_dim"],
        num_layers=config["model"]["gnn_num_layers"],
    ).to(device)

    node_features = torch.tensor(
        emb_dict["fused_embs"], dtype=torch.float32
    ).to(device)
    edge_index_gpu = edge_index.to(device)

    gnn.eval()
    with torch.no_grad():
        gnn_embs = gnn(node_features, edge_index_gpu).cpu().numpy()

    np.save("data/processed/gnn_embs.npy", gnn_embs)

    # ── Step 5: Classifier Training ──────────────────────────────
    print("\n[Phase 4] Training classifier...")
    from torch.utils.data import DataLoader, TensorDataset, random_split
    import torch.optim as optim
    import torch.nn.functional as F

    n_clusters = max(1, int(cluster_labels.max()) + 1)
    fused_t    = torch.tensor(emb_dict["fused_embs"], dtype=torch.float32)
    gnn_t      = torch.tensor(gnn_embs,               dtype=torch.float32)
    labels_t   = torch.tensor(emb_dict["labels"],     dtype=torch.float32)
    cluster_t  = torch.tensor(cluster_labels,         dtype=torch.long).clamp(min=0)

    cluster_oh = torch.zeros(len(cluster_t), n_clusters)
    cluster_oh.scatter_(1, cluster_t.unsqueeze(1), 1.0)

    dataset    = TensorDataset(fused_t, gnn_t, cluster_oh, labels_t)
    n          = len(dataset)
    train_size = int(0.70 * n)
    val_size   = int(0.15 * n)
    test_size  = n - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    BATCH        = config["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False)

    classifier = VerifAIClassifier(
        clip_dim=1024, gnn_dim=config["model"]["gnn_hidden_dim"],
        num_clusters=n_clusters
    ).to(device)

    optimizer = optim.AdamW(
        list(gnn.parameters()) + list(classifier.parameters()),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    criterion = FocalLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"]
    )

    best_val_f1 = 0
    patience    = config["training"]["early_stopping_patience"]
    no_improve  = 0

    for epoch in range(1, config["training"]["epochs"] + 1):
        # Train
        classifier.train()
        gnn.train()
        total_loss = 0
        for clip_emb, gnn_emb, c_oh, labels in train_loader:
            clip_emb = clip_emb.to(device)
            gnn_emb  = gnn_emb.to(device)
            c_oh     = c_oh.to(device)
            labels   = labels.to(device)
            optimizer.zero_grad()
            logits, _ = classifier(clip_emb, gnn_emb, c_oh)
            loss = criterion(logits.squeeze(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Validate
        classifier.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for clip_emb, gnn_emb, c_oh, labels in val_loader:
                _, probs = classifier(
                    clip_emb.to(device), gnn_emb.to(device), c_oh.to(device)
                )
                all_probs.extend(probs.squeeze().cpu().numpy())
                all_labels.extend(labels.numpy())

        val_m = compute_metrics(np.array(all_probs), np.array(all_labels))
        scheduler.step()

        print(f"Epoch {epoch:02d} | Loss: {total_loss/len(train_loader):.4f} "
              f"| Val F1: {val_m['f1']:.4f} | Val AUC: {val_m['auc_roc']:.4f}")

        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            no_improve  = 0
            torch.save({
                "classifier": classifier.state_dict(),
                "gnn":        gnn.state_dict(),
                "n_clusters": n_clusters,
                "epoch":      epoch,
                "val_f1":     best_val_f1,
            }, "models/verifai_best.pt")
            print(f"  ✅ Best model saved (F1={best_val_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n⏹  Early stopping at epoch {epoch}")
                break

    # Final test evaluation
    checkpoint = torch.load("models/verifai_best.pt", map_location=device)
    classifier.load_state_dict(checkpoint["classifier"])
    classifier.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for clip_emb, gnn_emb, c_oh, labels in test_loader:
            _, probs = classifier(
                clip_emb.to(device), gnn_emb.to(device), c_oh.to(device)
            )
            all_probs.extend(probs.squeeze().cpu().numpy())
            all_labels.extend(labels.numpy())

    test_m = compute_metrics(np.array(all_probs), np.array(all_labels))
    print("\n" + "="*40)
    print("       FINAL TEST RESULTS")
    print("="*40)
    for k, v in test_m.items():
        print(f"  {k:<12} : {v:.4f}")
    print("="*40)


if __name__ == "__main__":
    main()
