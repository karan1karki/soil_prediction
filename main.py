# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import onnxruntime as ort
import os

# Set page layout configuration
st.set_page_config(page_title="AgriSmart Analytics", page_icon="🌱", layout="centered")

st.title("🌱 AgriSmart AI Diagnostics Engine")
st.write("Upload or capture a soil profile image and calibrate environmental data inputs to evaluate conditions.")

# -------------------------------------------------------------------------
# 1. Load ML Artifacts securely from disk
# -------------------------------------------------------------------------
@st.cache_resource
def load_models():
    try:
        soil_model = ort.InferenceSession('rf_soil_classifier.onnx')
        le_soil = joblib.load('label_encoder_soil.pkl')
        moisture_model = ort.InferenceSession('rf_moisture_regressor.onnx')
        fert_model = ort.InferenceSession('rf_fertilizer_classifier.onnx')
        le_fert = joblib.load('label_encoder_fertilizer.pkl')
        return soil_model, le_soil, moisture_model, fert_model, le_fert, True
    except Exception as e:
        st.error(f"Error loading model pkl files: {e}")
        return None, None, None, None, None, False

soil_model, le_soil, moisture_model, fert_model, le_fert, models_ready = load_models()

# -------------------------------------------------------------------------
# 2. Camera Integration & Image Rendering UI
# -------------------------------------------------------------------------
st.subheader("📷 Step 1: Capture or Upload Soil / Leaf Profile")
image_source = st.radio("Choose Input Method:", ("Use Camera to Snap", "Upload Image File"))

captured_image = None
if image_source == "Use Camera to Snap":
    captured_image = st.camera_input("Take a photo of the ground target")
else:
    captured_image = st.file_uploader("Browse files for target image...", type=["jpg", "jpeg", "png"])

if captured_image:
    # Display the uploaded image frame to match the analyze layout concept
    img = Image.open(captured_image)
    st.image(img, caption="Target Soil Sample Frame Cached Successfully", use_container_width=True)

# -------------------------------------------------------------------------
# 3. Micro-Climate Hardware Sliders Panel
# -------------------------------------------------------------------------
st.subheader("📊 Step 2: Calibrate Soil Telemetry Parameters")
st.info("Since Random Forest evaluates tabular vectors, adjust the values below to match real-world sensor conditions:")

col1, col2 = st.columns(2)
with col1:
    temperature = st.slider("Temperature (°C)", min_value=0.0, max_value=50.0, value=28.4, step=0.1)
    nitrogen = st.slider("Nitrogen (N) Level (mg/kg)", min_value=0, max_value=150, value=45)
    phosphorous = st.slider("Phosphorous (P) Level (mg/kg)", min_value=0, max_value=150, value=35)

with col2:
    humidity = st.slider("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=65.0, step=0.1)
    potassium = st.slider("Potassium (K) Level (mg/kg)", min_value=0, max_value=150, value=40)

# -------------------------------------------------------------------------
# 4. Multistage Random Forest Inference Engine Execution
# -------------------------------------------------------------------------
st.subheader("🧠 Step 3: Run Diagnostics")

if st.button("🚀 Analyze Soil Profile", type="primary"):
    if not captured_image:
        st.warning("Please snap or upload a photo first to register the target environment.")
    elif not models_ready:
        st.error("ML processing aborted: Model artifacts (.pkl files) are missing.")
    else:
        with st.spinner("Executing Random Forest multi-stage tree evaluation..."):
            
            # 1. Force the base input matrix to use 32-bit float types explicitly
            base_inputs = np.array([[temperature, humidity, nitrogen, phosphorous, potassium]], dtype=np.float32)
            
            # Task A: Predict Soil Classification String
            soil_outputs = soil_model.run(None, {'float_input': base_inputs})
            soil_idx = soil_outputs[0]
            predicted_soil = le_soil.inverse_transform(soil_idx)[0]
            
            # Task B: Predict Volumetric Moisture Value via Regressor model
            moisture_outputs = moisture_model.run(None, {'float_input': base_inputs})
            predicted_moisture = float(moisture_outputs[0].flatten()[0])

            # 2. Force the fertilizer input matrix to use 32-bit float types as well
            fert_inputs = np.array([[temperature, humidity, predicted_moisture, nitrogen, phosphorous, potassium]], dtype=np.float32)
            
            # Task C: Predict Fertilizer using calculated intermediate moisture element
            fert_outputs = fert_model.run(None, {'float_input': fert_inputs})
            fert_idx = fert_outputs[0]
            predicted_fertilizer = le_fert.inverse_transform(fert_idx)[0]
            
            # -------------------------------------------------------------------------
            # 5. Render Output Metrics Dashboard
            # -------------------------------------------------------------------------
            st.success("🎉 Evaluation Complete!")
            
            # Display Key Output Cards
            c1, c2, c3 = st.columns(3)
            c1.metric(label="Identified Soil Profile", value=str(predicted_soil))
            c2.metric(label="Calculated Moisture", value=f"{predicted_moisture:.1f}%")
            c3.metric(label="Suggested Fertilizer", value=str(predicted_fertilizer))
            
            # Graphical Feedback bars matching your layout goals
            st.write("### Matrix Parameters Breakdown")
            
            st.write(f"**Volumetric Soil Moisture**: {predicted_moisture:.1f}%")
            st.progress(min(max(predicted_moisture / 100.0, 0.0), 1.0))
            
            st.write(f"**Nitrogen Level Density**: {nitrogen} mg/kg")
            st.progress(min(max(nitrogen / 150.0, 0.0), 1.0))
            
            # Static tip banner matching application style guidelines
            st.warning(
                f"💡 **AI Action Tip:** Identified {predicted_soil} environment. "
                f"Apply tailored proportions of {predicted_fertilizer} to optimize yields under your "
                f"current {predicted_moisture:.1f}% moisture conditions."
            )