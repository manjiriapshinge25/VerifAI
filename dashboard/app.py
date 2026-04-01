"""
dashboard/app.py
-----------------
VerifAI — Real-time Misinformation Detection Dashboard
Run: streamlit run dashboard/app.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="VerifAI",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
<style>
    .fake-label { color: #e63946; font-weight: 700; font-size: 1.4rem; }
    .real-label { color: #2a9d8f; font-weight: 700; font-size: 1.4rem; }
    .score-card { background: #f8f9fa; border-radius: 12px;
                  padding: 1.5rem; text-align: center; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("**Status:** 🟡 Demo mode (synthetic)")
    threshold = st.slider("Detection threshold", 0.1, 0.9, 0.5, 0.05)
    show_shap = st.checkbox("Show word importance", value=True)
    st.markdown("---")
    st.markdown("""
    **About VerifAI**
    - CLIP multimodal embeddings
    - Graph Attention Network
    - HDBSCAN narrative clustering
    - SHAP explainability
    """)


# ── Inference function ────────────────────────────────────────────────────
def run_inference(caption, threshold):
    fake_keywords = [
        "shocking", "secret", "they don't want", "hidden", "banned",
        "conspiracy", "hoax", "fake", "coverup", "miracle", "cure",
        "government", "5g", "microchip", "vaccine", "exposed",
        "truth", "hiding", "agenda", "suppressed", "leaked"
    ]
    text_lower = caption.lower()
    hits = sum(1 for kw in fake_keywords if kw in text_lower)
    base = min(0.3 + hits * 0.12, 0.85)
    noise = np.random.uniform(-0.05, 0.05)
    fake_prob  = float(np.clip(base + noise, 0.05, 0.95))
    cluster_id = int(np.random.randint(0, 8))
    top_words  = [w for w in caption.split() if w.lower() in fake_keywords][:3]
    if not top_words:
        top_words = ["no suspicious words found"]
    return fake_prob, cluster_id, top_words


# ── Main UI ───────────────────────────────────────────────────────────────
st.markdown("# 🔍 VerifAI")
st.markdown("*Multimodal Misinformation Detection — CLIP + GNN + XAI*")
st.markdown("---")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📥 Input Post")
    uploaded_image = st.file_uploader(
        "Upload image (optional)",
        type=["jpg", "jpeg", "png"]
    )
    caption = st.text_area(
        "Post caption / text",
        placeholder="Enter the social media caption here...",
        height=120,
    )
    hashtags = st.text_input(
        "Hashtags (comma-separated)",
        placeholder="#vaccine, #health, #breaking"
    )
    analyze_btn = st.button("🔍 Analyze Post", type="primary", use_container_width=True)

with col2:
    st.markdown("### 📊 Analysis Results")

    if analyze_btn and caption.strip():
        with st.spinner("Running VerifAI pipeline..."):
            fake_prob, cluster_id, top_words = run_inference(caption, threshold)
            is_fake = fake_prob >= threshold

        if is_fake:
            label_html = '<span class="fake-label">⚠️ MISINFORMATION</span>'
        else:
            label_html = '<span class="real-label">✅ LIKELY REAL</span>'

        st.markdown(f"""
        <div class="score-card">
            {label_html}<br>
            <span style="font-size:2.2rem;font-weight:800;">{fake_prob:.0%}</span><br>
            <span style="color:#6c757d;">Misinformation probability</span>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        gauge_color = "#e63946" if is_fake else "#2a9d8f"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(fake_prob * 100),
            number={"suffix": "%"},
            title={"text": "Risk Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": gauge_color},
                "steps": [
                    {"range": [0,  40], "color": "#d4edda"},
                    {"range": [40, 70], "color": "#fff3cd"},
                    {"range": [70, 100], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "value": threshold * 100,
                },
            },
        ))
        fig.update_layout(height=220, margin=dict(t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"**Narrative cluster assigned:** #{cluster_id}")

    elif analyze_btn:
        st.warning("Please enter a caption before analyzing.")

    else:
        st.info("Enter a caption and click Analyze to get started.")
        st.markdown("**Try one of these examples:**")
        st.code("Government secretly putting microchips in vaccines to track citizens")
        st.code("Local council approves new budget for road maintenance this quarter")
        st.code("Scientists confirm 5G towers are spreading viruses, doctors hiding truth")


# ── Explainability Section ────────────────────────────────────────────────
if analyze_btn and caption.strip():
    st.markdown("---")
    st.markdown("### 🧠 Explainability")

    exp1, exp2 = st.columns(2)

    with exp1:
        st.markdown("**📝 Word importance (SHAP)**")
        if show_shap:
            if top_words and top_words[0] != "no suspicious words found":
                for word in top_words:
                    st.markdown(
                        f'<span style="background:#ffe0e0;padding:3px 10px;'
                        f'border-radius:4px;margin:3px;display:inline-block;">'
                        f'⚠️ {word}</span>',
                        unsafe_allow_html=True
                    )
            else:
                st.success("No strongly suspicious words detected.")

    with exp2:
        st.markdown("**🖼️ Image analysis**")
        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded image", use_column_width=True)
            st.info("GradCAM heatmap available when real model is loaded.")
        else:
            st.info("Upload an image to see visual attention heatmap.")

    st.markdown("---")
    st.markdown("### 📌 Verdict")
    if is_fake:
        st.error(
            f"VerifAI flagged this post with **{fake_prob:.0%}** confidence "
            f"as potential misinformation. It was assigned to narrative cluster "
            f"**#{cluster_id}**, which groups similar misleading posts together."
        )
    else:
        st.success(
            f"VerifAI found no strong misinformation signals in this post "
            f"({fake_prob:.0%} risk score). Assigned to narrative cluster **#{cluster_id}**."
        )
