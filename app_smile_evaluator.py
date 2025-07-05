import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import tempfile
import os

# Optional: identity similarity using InsightFace
try:
    import insightface
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
    face_model = FaceAnalysis(name='buffalo_l')
    face_model.prepare(ctx_id=0)
    use_insightface = True
except Exception as e:
    use_insightface = False
    st.warning("InsightFace is not available. Facial identity check will be skipped.")

st.set_page_config(page_title="AI Smile Design Evaluator", layout="wide")
st.title("🦷 AI Smile Design Evaluator")
st.markdown("Upload your original and AI-generated smile images to compare.")

uploaded_original = st.file_uploader("Upload Original Smile Image", type=["jpg", "jpeg", "png"], key="orig")
uploaded_generated = st.file_uploader("Upload AI-Generated Frame", type=["jpg", "jpeg", "png"], key="gen")

if uploaded_original and uploaded_generated:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as orig_file:
        orig_path = orig_file.name
        orig_file.write(uploaded_original.read())

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as gen_file:
        gen_path = gen_file.name
        gen_file.write(uploaded_generated.read())

    # Load and resize
    orig_img = cv2.imread(orig_path)
    gen_img = cv2.imread(gen_path)
    orig_img = cv2.resize(orig_img, (512, 512))
    gen_img = cv2.resize(gen_img, (512, 512))

    col1, col2 = st.columns(2)
    with col1:
        st.image(orig_img[:, :, ::-1], caption="Original Image")
    with col2:
        st.image(gen_img[:, :, ::-1], caption="Generated Image")

    # SSIM
    gray_orig = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    gray_gen = cv2.cvtColor(gen_img, cv2.COLOR_BGR2GRAY)
    ssim_score, _ = ssim(gray_orig, gray_gen, full=True)
    st.metric("🔍 SSIM (Similarity Index)", f"{ssim_score:.4f}")

    # InsightFace Identity Match
    if use_insightface:
        faces_orig = face_model.get(orig_img)
        faces_gen = face_model.get(gen_img)
        if faces_orig and faces_gen:
            emb1 = faces_orig[0].embedding
            emb2 = faces_gen[0].embedding
            cosine_similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            st.metric("🧠 Identity Similarity (Cosine)", f"{cosine_similarity:.4f}")
            if cosine_similarity > 0.7:
                st.success("✅ Faces are likely the same")
            else:
                st.warning("⚠️ Faces may not match closely")
        else:
            st.warning("Could not detect faces in one or both images.")
