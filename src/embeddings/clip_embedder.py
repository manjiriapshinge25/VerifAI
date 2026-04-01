"""
src/embeddings/clip_embedder.py
--------------------------------
Extracts multimodal embeddings using CLIP.

Supports two modes:
  - mode="real"      → loads actual images from disk (for NewsCLIPpings etc.)
  - mode="synthetic" → generates random CLIP-like embeddings (for development/testing)

Your synthetic notebook uses mode="synthetic".
Switch to mode="real" when you have a real dataset.
"""

import torch
import numpy as np
from tqdm import tqdm


# ── Synthetic Mode ────────────────────────────────────────────────────────

def extract_synthetic_embeddings(df, embedding_dim=512, seed=42):
    """
    Generates random CLIP-like embeddings for synthetic data.

    This is what your Jupyter notebook does internally — we just
    wrap it here so train.py can call it the same way as the real version.

    Args:
        df: DataFrame with columns [text, label, post_id, hashtags, user_id]
        embedding_dim: size of each image/text embedding (CLIP uses 512)
        seed: random seed for reproducibility

    Returns:
        dict with keys: image_embs, text_embs, fused_embs, labels
    """
    np.random.seed(seed)
    n = len(df)

    # Generate random L2-normalised embeddings (mimics real CLIP output)
    image_embs = np.random.randn(n, embedding_dim).astype(np.float32)
    text_embs  = np.random.randn(n, embedding_dim).astype(np.float32)

    # L2 normalise — same as real CLIP does
    image_embs = image_embs / np.linalg.norm(image_embs, axis=1, keepdims=True)
    text_embs  = text_embs  / np.linalg.norm(text_embs,  axis=1, keepdims=True)

    # Make fake posts slightly more distinguishable from real ones
    # (adds a small signal so the classifier can actually learn something)
    fake_mask = df["label"].values == 1
    image_embs[fake_mask] += np.random.randn(fake_mask.sum(), embedding_dim) * 0.3
    text_embs[fake_mask]  += np.random.randn(fake_mask.sum(), embedding_dim) * 0.3

    # Re-normalise after adding signal
    image_embs = image_embs / np.linalg.norm(image_embs, axis=1, keepdims=True)
    text_embs  = text_embs  / np.linalg.norm(text_embs,  axis=1, keepdims=True)

    fused_embs = np.concatenate([image_embs, text_embs], axis=-1)
    labels     = df["label"].values.astype(np.int64)

    print(f"[SyntheticEmbedder] Generated embeddings for {n} samples")
    print(f"  image_embs : {image_embs.shape}")
    print(f"  text_embs  : {text_embs.shape}")
    print(f"  fused_embs : {fused_embs.shape}")

    return {
        "image_embs": image_embs,
        "text_embs":  text_embs,
        "fused_embs": fused_embs,
        "labels":     labels,
    }


# ── Real Mode ─────────────────────────────────────────────────────────────

class CLIPEmbedder:
    """
    Wraps CLIP model to extract real image + text embeddings.
    Use this when you switch to a real dataset (NewsCLIPpings, etc.)

    Your DataFrame must have columns: [image_path, text, label]
    """

    def __init__(self, model_name="ViT-B/32", device=None):
        try:
            import clip
            self.clip = clip
        except ImportError:
            raise ImportError("Run: pip install openai-clip")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = self.clip.load(model_name, device=self.device)
        self.model.eval()
        print(f"[CLIPEmbedder] Loaded {model_name} on {self.device}")

    def extract_embeddings(self, dataframe, batch_size=32):
        """
        Extract image, text, and fused embeddings for all samples.
        """
        from torch.utils.data import DataLoader, Dataset
        from PIL import Image

        clip = self.clip

        class PostDataset(Dataset):
            def __init__(self, df, preprocess):
                self.df = df.reset_index(drop=True)
                self.preprocess = preprocess

            def __len__(self):
                return len(self.df)

            def __getitem__(self, idx):
                row = self.df.iloc[idx]
                try:
                    image = self.preprocess(
                        Image.open(row["image_path"]).convert("RGB")
                    )
                except Exception:
                    image = torch.zeros(3, 224, 224)  # fallback for missing images

                text  = clip.tokenize([row["text"]], truncate=True)[0]
                label = torch.tensor(row["label"], dtype=torch.long)
                return image, text, label

        dataset = PostDataset(dataframe, self.preprocess)
        loader  = DataLoader(dataset, batch_size=batch_size,
                             shuffle=False, num_workers=2)

        image_embs, text_embs, all_labels = [], [], []

        with torch.no_grad():
            for images, texts, labels in tqdm(loader, desc="Extracting CLIP embeddings"):
                images = images.to(self.device)
                texts  = texts.to(self.device)

                img_feat = self.model.encode_image(images)
                txt_feat = self.model.encode_text(texts)

                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

                image_embs.append(img_feat.cpu().numpy())
                text_embs.append(txt_feat.cpu().numpy())
                all_labels.append(labels.numpy())

        image_embs = np.concatenate(image_embs)
        text_embs  = np.concatenate(text_embs)
        labels     = np.concatenate(all_labels)
        fused_embs = np.concatenate([image_embs, text_embs], axis=-1)

        print(f"[CLIPEmbedder] Extracted embeddings for {len(labels)} samples")
        return {
            "image_embs": image_embs,
            "text_embs":  text_embs,
            "fused_embs": fused_embs,
            "labels":     labels,
        }
