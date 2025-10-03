import streamlit as st
import torch
import numpy as np
import joblib
from ultralytics import YOLO
from PIL import Image

# -------------------------
# Load models
# -------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model = YOLO("best.pt").to(device)
loaded_model_data = joblib.load("final.pkl")
xgb_model = loaded_model_data['xgb_model']
thresholds = loaded_model_data['thresholds']

# Class labels
CLASS_NAMES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Diabetic Retinopathy Detector", page_icon="🩺", layout="centered")

# -------------------------
# Custom Styling
# -------------------------
st.markdown(
    """
    <style>
        /* Background and general text */
        body, .stApp {
            background-color: white !important;
            color: #1a237e !important;
        }
        h1, h2, h3, h4, h5, h6, p, label, div {
            color: #1a237e !important;
        }

        /* File uploader */
        .stFileUploader label {
            color: white !important;
        }
        .stFileUploader {
            background-color: #1a237e !important;
            border-radius: 8px;
            padding: 10px;
            text-align: center;
        }

        /* Result card */
        .result-card {
            padding: 20px;
            border-radius: 12px;
            background-color: #ffffff;
            border: 2px solid #1a237e;
            box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
            text-align: center;
            margin-top: 20px;
        }
        .result-label {
            font-size: 26px;
            font-weight: bold;
            color: #1a237e;
        }
        .confidence {
            font-size: 18px;
            color: #1a237e;
        }

        /* Center buttons */
        .center-button {
            display: flex;
            justify-content: center;
            margin-top: 30px;
        }

        /* Streamlit button styling */
        .stButton>button {
            border-radius: 8px !important;
            padding: 10px 25px !important;
            font-size: 16px !important;
            border: none !important;
            cursor: pointer !important;
        }

        /* Run Diagnosis button */
        .run-btn {
            background-color: #1a237e !important;
            color: white !important;
        }

        /* Reset button */
        .reset-btn {
            background-color: white !important;
            color: #1a237e !important;
            border: 2px solid #1a237e !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# App Layout
# -------------------------
st.markdown("<h1 style='text-align:center;'>🩺 Diabetic Retinopathy Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload a retinal fundus image to detect the stage of Diabetic Retinopathy.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Fundus Image", use_column_width=True)

    # Center Run Diagnosis Button
    st.markdown("<div class='center-button'>", unsafe_allow_html=True)
    run_btn = st.button("🔍 Run Diagnosis")
    st.markdown("</div>", unsafe_allow_html=True)

    if run_btn:
        # Step 1: Extract YOLO features
        results = yolo_model.predict(image, verbose=False, device=device)
        yolo_probs = results[0].probs.data.cpu().numpy()

        # Step 2: Predict with XGBoost
        y_pred_proba = xgb_model.predict_proba([yolo_probs])[0]

        # Threshold adjustment
        assigned = False
        for c, t in thresholds.items():
            if y_pred_proba[c] > t:
                prediction = c
                assigned = True
                break
        if not assigned:
            prediction = np.argmax(y_pred_proba)

        confidence = y_pred_proba[prediction]

        # Display Results
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">{CLASS_NAMES[prediction]}</div>
                <div class="confidence">Confidence: {confidence:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Center Upload Another Image button
        st.markdown("<div class='center-button'>", unsafe_allow_html=True)
        st.markdown(
            """
            <form action="" method="get">
                <button type="submit" class="reset-btn">
                🔄 Upload Another Image
                </button>
            </form>
            """,
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
