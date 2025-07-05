# 🦷 AI Smile Design Evaluator (InsightFace Edition)

This Streamlit app allows dentists and researchers to compare original smile design images with AI-generated results using:

- **SSIM (Structural Similarity Index)** for visual fidelity
- **InsightFace** for facial identity similarity (cosine distance)
- Compatible with Streamlit Cloud (no TensorFlow required)

## 🚀 Features

- Upload and compare smile images
- Quantify similarity and identity match
- No need for DeepFace or TensorFlow
- Lightweight and cloud-deployable

## 🛠️ Technologies

- Streamlit
- OpenCV
- scikit-image
- InsightFace
- ONNX Runtime
- NumPy, Pillow

## 📦 Installation

```bash
pip install -r requirements.txt
streamlit run app_smile_evaluator_insightface.py
```

## 📂 Files

| File                             | Description                      |
|----------------------------------|----------------------------------|
| `app_smile_evaluator_insightface.py` | Main Streamlit app using InsightFace |
| `requirements.txt`               | Cloud-safe dependencies          |
| `README.md`                      | Project documentation            |

## 🌐 Deployment

This app is ready to deploy on [Streamlit Cloud](https://streamlit.io/cloud):
1. Upload all files to a public GitHub repository.
2. Deploy by selecting `app_smile_evaluator_insightface.py` as the entry point.

---

Developed for AI-based digital smile design evaluations.
