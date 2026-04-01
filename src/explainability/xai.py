"""
src/explainability/xai.py
---------------------------
Explainability layer using SHAP (text) and GradCAM (image).
This answers: WHY did VerifAI flag this post as misinformation?
TODO: Integrate GradCAM++ for better spatial explanations.
"""

import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
from PIL import Image


class TextExplainer:
    """
    SHAP-based explainer for text contributions to misinformation score.
    Highlights which words/phrases pushed the model toward fake/real.
    """

    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def explain(self, text, clip_image_emb, gnn_emb, cluster_one_hot, n_samples=100):
        """
        Compute SHAP values for each token in the text.

        Args:
            text: raw string input
            clip_image_emb: precomputed image embedding [1, D]
            gnn_emb: precomputed GNN embedding [1, D]
            cluster_one_hot: cluster assignment [1, K]

        Returns:
            shap_values: array of per-token importance scores
            tokens: list of token strings

        TODO: Implement full SHAP masking pipeline for CLIP text encoder.
        """
        # TODO: Implement SHAP masker that replaces tokens and measures output change
        # Reference: https://shap.readthedocs.io/en/latest/text_examples.html
        raise NotImplementedError("TODO: Implement SHAP text explainer")


class ImageExplainer:
    """
    GradCAM-based explainer for image contributions.
    Produces a heatmap showing which image regions triggered the fake flag.
    """

    def __init__(self, clip_model, device="cpu"):
        self.clip_model = clip_model
        self.device = device
        self.gradients = None
        self.activations = None

    def _register_hooks(self, target_layer):
        """Register forward/backward hooks on CLIP vision transformer layer."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def generate_heatmap(self, image_tensor, classifier_output):
        """
        Generate GradCAM heatmap for a given image.

        Args:
            image_tensor: preprocessed image [1, 3, 224, 224]
            classifier_output: model logit [1, 1]

        Returns:
            heatmap: np.array [224, 224], values in [0, 1]

        TODO: Hook into the last attention layer of CLIP ViT for best results.
        """
        # TODO: Implement GradCAM for CLIP vision transformer
        # Reference: https://github.com/jacobgil/pytorch-grad-cam
        raise NotImplementedError("TODO: Implement GradCAM for CLIP ViT")

    def visualize(self, original_image, heatmap, save_path=None):
        """Overlay heatmap on original image."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(original_image)
        axes[1].imshow(heatmap, alpha=0.5, cmap="jet")
        axes[1].set_title("GradCAM — Suspicious Regions")
        axes[1].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
