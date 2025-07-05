
import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from deepface import DeepFace
import mediapipe as mp
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="AI Smile Design Evaluator", layout="wide")
st.title("🦷 AI Smile Design Evaluator")
st.markdown("Compare original smile design images with AI-generated videos for fidelity and facial preservation.")

uploaded_original = st.file_uploader("Upload Original Smile Image", type=["jpg", "jpeg", "png"], key="orig")
uploaded_generated = st.file_uploader("Upload AI-Generated Frame", type=["jpg", "jpeg", "png"], key="gen")

if uploaded_original and uploaded_generated:
    with tempfile.NamedTemporaryFile(delete=False) as orig_file:
        orig_path = orig_file.name
        orig_file.write(uploaded_original.read())

    with tempfile.NamedTemporaryFile(delete=False) as gen_file:
        gen_path = gen_file.name
        gen_file.write(uploaded_generated.read())

    # Load and display
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

    # DeepFace
    try:
        result = DeepFace.verify(img1_path=orig_path, img2_path=gen_path, enforce_detection=False)
        st.metric("🧠 DeepFace Cosine Distance", f"{result['distance']:.4f}")
        st.success("✅ Faces Match" if result['verified'] else "❌ Faces Do Not Match")
    except Exception as e:
        st.warning(f"DeepFace error: {e}")

    # Facial Landmarks using MediaPipe
    st.markdown("### Facial Landmarks Comparison")
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(static_image_mode=True)

    def draw_landmarks(image):
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            annotated = image.copy()
            for landmarks in results.multi_face_landmarks:
                for lm in landmarks.landmark:
                    x, y = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
                    cv2.circle(annotated, (x, y), 1, (0, 255, 0), -1)
            return annotated
        return image

    col1, col2 = st.columns(2)
    with col1:
        st.image(draw_landmarks(orig_img)[:, :, ::-1], caption="Original Landmarks")
    with col2:
        st.image(draw_landmarks(gen_img)[:, :, ::-1], caption="Generated Landmarks")
