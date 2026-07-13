import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import pickle
import ee
from geopy.geocoders import Nominatim

# Set up Streamlit Page Configuration
st.set_page_config(page_title="AgriSmart Analytics Portal", layout="wide", page_icon="🌱")

# -------------------------------------------------------------
# 1. GLOBAL INITIALIZATION & MODEL LOADING
# -------------------------------------------------------------
# Extract credentials from Streamlit Secrets
secret_creds = st.secrets["gcp_service_account"]

# Create the Earth Engine credentials object
credentials = ee.ServiceAccountCredentials(
    secret_creds["client_email"], 
    key_data = secret_creds["private_key"]
)

# Initialize Earth Engine with the service account credentials
GCP_PROJECT_ID = 'agrismart-498704'

@st.cache_resource
def initialize_systems():
    # Initialize Earth Engine
    try:
            secret_creds = st.secrets["gcp_service_account"]
            
            # Clean up newline characters if they got escaped
            private_key = secret_creds["private_key"].replace("\\n", "\n")
            
            credentials = ee.ServiceAccountCredentials(
                secret_creds["client_email"], 
                key_data=private_key
            )
            ee.Initialize(credentials=credentials, project=GCP_PROJECT_ID)
            st.success("✅ Earth Engine successfully initialized!")
    except Exception as e:
            st.error(f"❌ Earth Engine initialization failed: {e}")
        # Load Leaf CNN Model
    try:
        leaf_model = tf.keras.models.load_model('rice_leaf_cnn_model.h5')
    except Exception:
        leaf_model = None

    # Load Crop Recommendation Model
    try:
        with open('crop_recommendation_model.pkl', 'rb') as file:
            crop_model = pickle.load(file)
    except Exception:
        crop_model = None

    return leaf_model, crop_model

leaf_model, crop_model = initialize_systems()

RICE_CLASSES = ['Bacterial Leaf Blight', 'Brown Spot', 'Healthy Rice Leaf', 'Leaf Blast']

# -------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -------------------------------------------------------------
def get_coordinates_from_name(location_name: str):
    try:
        geolocator = Nominatim(user_agent="agrismart_nepal_app")
        location = geolocator.geocode(location_name)
        return (location.latitude, location.longitude) if location else None
    except Exception:
        return None

def fetch_satellite_metrics(lat: float, lon: float):
    point = ee.Geometry.Point([lon, lat])
    
    # 1. Soil Moisture (Sentinel-2 NDMI) 🛰️
    try:
        s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                         .filterBounds(point)
                         .filterDate('2025-06-01', '2026-06-01')
                         .sort('CLOUDY_PIXEL_PERCENTAGE').first())
        ndmi = s2_collection.normalizedDifference(['B8', 'B11'])
        moisture_val = ndmi.reduceRegion(ee.Reducer.mean(), point, 30).get('ndmi').getInfo()
        soil_moisture = int(max(0, min(100, (moisture_val + 1) * 50))) if moisture_val else 45
    except Exception:
        soil_moisture = 48
        
    # 2. Temperature (ERA5 Land Hourly) 🌡️
    try:
        era5 = (ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
                .filterBounds(point)
                .sort('system:time_start', False).first())
        temp_k = era5.reduceRegion(ee.Reducer.mean(), point, 1000).get('temperature_2m').getInfo()
        temperature = round(temp_k - 273.15, 2) if temp_k else 26.5
    except Exception:
        temperature = 27.2

    # 3. Nitrogen (OpenLandMap Soil Nitrogen) 🌱
    try:
        map_n = ee.Image("OpenLandMap/SOL/SOL_CHEM-N_USDA-4A1A_M/v01").select('b0')
        n_val = map_n.reduceRegion(ee.Reducer.mean(), point, 250).get('b0').getInfo()
        nitrogen = int(n_val / 10) if n_val else 70
    except Exception:
        nitrogen = 65

    # 4. pH Level (OpenLandMap Soil pH) 🧪
    try:
        map_ph = ee.Image("OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v01").select('b0')
        ph_val = map_ph.reduceRegion(ee.Reducer.mean(), point, 250).get('b0').getInfo()
        # OpenLandMap stores pH multiplied by 10 (e.g., 64 instead of 6.4)
        ph = round(ph_val / 10, 1) if ph_val else 6.5
    except Exception:
        ph = 6.4

    # 5. Rainfall (CHIRPS Daily Precipitation) 🌧️
    try:
        # Summing daily rainfall over a recent historical window
        chirps = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                  .filterBounds(point)
                  .filterDate('2025-01-01', '2026-01-01'))
        total_rain = chirps.reduce(ee.Reducer.sum())
        # CHIRPS bands change name after reduction to 'precipitation_sum'
        rain_val = total_rain.reduceRegion(ee.Reducer.mean(), point, 5000).get('precipitation_sum').getInfo()
        rainfall = round(rain_val, 1) if rain_val else 150.0
    except Exception:
        rainfall = 145.0
        phosphorus = 42
        potassium = 38

    return {
        "N": nitrogen, 
        "P": 42, 
        "K": 38,
        "temperature": temperature, 
        "humidity": soil_moisture + 20, # Map moisture to relative humidity scale
        "ph": ph, 
        "rainfall": rainfall
    }
# -------------------------------------------------------------
# 3. STREAMLIT USER INTERFACE LAYOUT
# -------------------------------------------------------------
st.title("🌱 AgriSmart Nepal Unified Hub")


st.header("Geospatial Crop Analytics & Computer Vision Diagnostics")
location_input = st.text_input("Enter Field Location Name", placeholder="e.g., Bharatpur, Nepal")
uploaded_file = st.file_uploader("Upload a rice leaf image", type=["jpg", "jpeg", "png"])
if st.button("Analyze Location & Recommend"):
    if location_input:
        with st.spinner("Processing geospatial markers..."):
            coords = get_coordinates_from_name(location_input)
            if coords:
                lat, lon = coords
                metrics = fetch_satellite_metrics(lat, lon)
                
                st.success(f"📍 Mapped to Coordinates: {lat:.4f}, {lon:.4f}")
                
                # Display Metrics
                c1, c2, c3 , c4, c5, c6, c7= st.columns(7)
                c1.metric("Nitrogen (N)", f"{metrics['N']} mg/kg")
                c2.metric("Phosphorus (P)", f"{metrics['P']} mg/kg")
                c3.metric("Potassium (K)", f"{metrics['K']} mg/kg")
                c4.metric("Temperature", f"{metrics['temperature']}")
                c5.metric("Humidity", f"{metrics['humidity']}")
                c6.metric("PH_Level", f"{metrics['ph']}")
                c7.metric("Rainfall", f"{metrics['rainfall']}")
                
                # Make Prediction
                if crop_model is not None:
                    input_vector = [[metrics['N'], metrics['P'], metrics['K'], metrics['temperature'], metrics['humidity'], metrics['ph'], metrics['rainfall']]]
                    prediction = crop_model.predict(input_vector)[0]
                    st.subheader(f"🥇 Recommended Crop: **{prediction.upper()}**")
                else:
                    st.info("🎯 Simulation Mode Recommendation: MAIZE")
            else:
                st.error("Location not found.")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Leaf Profile", width=300)
        
        with st.spinner("Running CNN classification..."):
            if leaf_model is not None:
                resized_image = image.convert('RGB').resize((224, 224))
                img_tensor = np.expand_dims(np.asarray(resized_image) / 255.0, axis=0)
                predictions = leaf_model.predict(img_tensor)[0]
                max_idx = np.argmax(predictions)
                
                st.success(f"Diagnosis: **{RICE_CLASSES[max_idx]}**")
                st.info(f"Confidence Level: {float(predictions[max_idx])*100:.2f}%")
            else:
                st.warning("Running in simulation mode.")
                st.success("Diagnosis: **Brown Spot** (Simulation Confidence: 92.0%)")
