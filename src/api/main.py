"""
src/api/main.py
----------------
VerifAI REST API — production-ready FastAPI endpoint.
Run: uvicorn src.api.main:app --reload --port 8000

Endpoints:
  POST /analyze  — analyze a post (image + text)
  GET  /health   — health check
  GET  /clusters — return current cluster summary
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import torch
import io
from PIL import Image

app = FastAPI(
    title="VerifAI API",
    description="Real-time Multimodal Misinformation Detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisResult(BaseModel):
    post_id: Optional[str]
    misinformation_probability: float
    is_misinformation: bool
    confidence: str           # "high" / "medium" / "low"
    cluster_id: int
    cluster_label: str        # human-readable narrative name
    top_suspicious_words: list[str]
    model_version: str


@app.get("/health")
def health():
    return {"status": "ok", "model": "VerifAI v1.0", "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))}


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_post(
    image: UploadFile = File(...),
    caption: str = Form(...),
    hashtags: str = Form(""),
    post_id: Optional[str] = Form(None),
    threshold: float = Form(0.5),
):
    """
    Analyze a social media post for misinformation.

    TODO: Load model at startup using lifespan context manager.
    TODO: Implement real inference pipeline here.
    """
    # Validate image
    if image.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Invalid image format. Use JPEG or PNG.")

    img_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # TODO: Run real inference
    # 1. Extract CLIP embeddings for image + caption
    # 2. Look up/assign cluster
    # 3. Build mini-graph or use precomputed GNN features
    # 4. Run classifier
    # 5. Run SHAP for top suspicious words

    fake_prob = 0.5   # TODO: Replace with real model output
    cluster_id = 0    # TODO: Replace with real cluster assignment

    confidence = "high" if abs(fake_prob - 0.5) > 0.3 else ("medium" if abs(fake_prob - 0.5) > 0.15 else "low")

    return AnalysisResult(
        post_id=post_id,
        misinformation_probability=round(fake_prob, 4),
        is_misinformation=fake_prob >= threshold,
        confidence=confidence,
        cluster_id=cluster_id,
        cluster_label="TODO: narrative label",
        top_suspicious_words=["TODO", "implement", "SHAP"],
        model_version="VerifAI-v1.0",
    )


@app.get("/clusters")
def get_clusters():
    """
    Return summary of discovered misinformation narrative clusters.
    TODO: Load cluster metadata from saved clustering results.
    """
    return {
        "message": "TODO: Return cluster summaries",
        "clusters": []
    }
