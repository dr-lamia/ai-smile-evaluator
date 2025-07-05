# 🦷 AI Smile Design Evaluator

This Streamlit app allows clinicians and researchers to compare original smile design images with AI-generated videos for fidelity and facial identity preservation.

## 🚀 Features

- Upload original and AI-generated smile images
- Compare visual similarity using **SSIM**
- Verify facial identity using **DeepFace**
- Visualize facial landmarks using **MediaPipe**
- Side-by-side comparison and interactive evaluation

## 🛠️ Technologies

- Streamlit
- OpenCV
- NumPy
- scikit-image
- DeepFace
- MediaPipe
- Pillow

## 📦 Installation

```bash
pip install -r requirements.txt
streamlit run app_smile_evaluator.py
```

## 📂 Files

| File | Description |
|------|-------------|
| `app_smile_evaluator.py` | Main Streamlit app |
| `requirements.txt` | Dependencies for deployment |
| `README.md` | Project overview |

## 🌐 Deployment

You can deploy this app on [Streamlit Cloud](https://streamlit.io/cloud):
1. Upload files to a public GitHub repository.
2. Open Streamlit Cloud.
3. Select your repo and set `app_smile_evaluator.py` as the app entry.
4. Done!

## 👩‍⚕️ Use Case

Designed for dentists, prosthodontists, and researchers working with:
- Digital Smile Design (DSD)
- AI-generated smile animation
- Patient engagement in aesthetic dentistry

---

Developed for AI in Dentistry research and education.