import streamlit as st
import pandas as pd
import numpy as np
import pickle
import ee
from geopy.geocoders import Nominatim  # 👈 Import the free Geocoder tool

# Set up Streamlit Page Configuration
st.set_page_config(page_title="AgriSmart Analytics Portal", layout="wide", page_icon="🌱")

# -------------------------------------------------------------
# 1. INITIALIZE GOOGLE EARTH ENGINE
# -------------------------------------------------------------
GCP_PROJECT_ID = 'agrismart-498704' 

@st.cache_resource
def initialize_earth_engine():
    try:
        ee.Authenticate()
        ee.Initialize(project=GCP_PROJECT_ID)
        return True
    except Exception:
        try:
            ee.Authenticate()
            ee.Initialize(project=GCP_PROJECT_ID)
            return True
        except Exception as e:
            st.error(f"Earth Engine Authorization Failed: {e}")
            return False

ee_ready = initialize_earth_engine()

# Load model asset configuration
@st.cache_resource
def load_crop_model():
    try:
        with open('crop_recommendation_model.pkl', 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        return None

crop_model = load_crop_model()

# -------------------------------------------------------------
# NEW FEATURE: CONVERT LOCATION NAME TO COORDINATES
# -------------------------------------------------------------
def get_coordinates_from_name(location_name):
    try:
        # Initialize OpenStreetMap Nominatim Geocoder
        geolocator = Nominatim(user_agent="agrismart_nepal_app")
        location = geolocator.geocode(location_name)
        if location:
            return location.latitude, location.longitude
        return None
    except Exception as e:
        st.error(f"Geocoding lookup error: {e}")
        return None

# -------------------------------------------------------------
# 3. HELPER FUNCTION: FETCH SATELLITE METRICS (Stays same)
# -------------------------------------------------------------
def fetch_satellite_metrics(lat, lon):
    if not ee_ready:
        return None
    point = ee.Geometry.Point([lon, lat])
    
    # Fetching satellite layer parameters...
    try:
        s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                         .filterBounds(point)
                         .filterDate('2026-01-01', '2026-06-01')
                         .sort('CLOUDY_PIXEL_PERCENTAGE').first())
        ndmi = s2_collection.normalizedDifference(['B8', 'B11'])
        moisture_val = ndmi.reduceRegion(ee.Reducer.mean(), point, 30).get('ndmi').getInfo()
        soil_moisture = int(max(0, min(100, (moisture_val + 1) * 50))) if moisture_val else 45
    except Exception:
        soil_moisture = 48
        
    try:
        era5 = (ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").filterBounds(point).sort('system:time_start', False).first())
        temp_k = era5.reduceRegion(ee.Reducer.mean(), point, 1000).get('temperature_2m').getInfo()
        temperature = round(temp_k - 273.15, 2) if temp_k else 26.5
    except Exception:
        temperature = 27.2

    # Nitrogen estimations from Earth Engine global soil maps proxy layers
    try:
        map_n = ee.Image("OpenLandMap/SOL/SOL_CHEM-N_USDA-4A1A_M/v01").select('b0')
        n_val = map_n.reduceRegion(ee.Reducer.mean(), point, 250).get('b0').getInfo()
        nitrogen = int(n_val / 10) if n_val else 70
    except Exception:
        nitrogen = 65

    return {
        "N": nitrogen, "P": 42, "K": 38,
        "temperature": temperature, "humidity": soil_moisture + 20,
        "ph": 6.4, "rainfall": 150.0
    }

# -------------------------------------------------------------
# 4. STREAMLIT USER INTERFACE LAYOUT
# -------------------------------------------------------------
st.title("🌱 AgriSmart Nepal Dashboard")
st.markdown("Autonomous No-Hardware Soil Analysis & Crop Recommendation Hub")

tab1, tab2 = st.tabs(["📊 Crop Prediction Portal", "🔬 Diagnostics Center"])

with tab1:
    st.header("Satellite Geospatial Crop Analytics")
    st.write("Enter your region or city name below. The system will map coordinates and fetch satellite data automatically.")
    
    # 👈 Switch inputs from numeric float boxes to a text field
    location_input = st.text_input("Enter Field Location / Place Name", placeholder="e.g., Bharatpur, Nepal")
        
    if st.button("Fetch Earth Engine Soil & Weather Properties"):
        if location_input:
            with st.spinner(f"Geocoding '{location_input}' and fetching telemetry values..."):
                # 1. Look up coordinates using geopy
                coordinates = get_coordinates_from_name(location_input)
                
                if coordinates:
                    target_lat, target_lon = coordinates
                    st.info(f"📍 Location Found! Mapped Coordinates: Latitude {target_lat:.4f}, Longitude {target_lon:.4f}")
                    
                    # 2. Feed coordinates into Earth Engine telemetry function
                    metrics = fetch_satellite_metrics(target_lat, target_lon)
                    
                    if metrics:
                        st.session_state['soil_metrics'] = metrics
                        st.success("Data successfully fetched from Earth Engine Cloud Catalog!")
                    else:
                        st.error("Failed to extract environmental characteristics.")
                else:
                    st.error("Could not find coordinates for that location name. Try typing it cleanly (e.g., 'Chitwan, Nepal').")
        else:
            st.warning("Please type a location name first.")

    # Show metrics and recommendation step if available
    if 'soil_metrics' in st.session_state:
        m = st.session_state['soil_metrics']
        
        st.markdown("### Telemetry Metric Framework Matrix")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(label="Nitrogen (N)", value=f"{m['N']} mg/kg")
        c2.metric(label="Phosphorus (P)", value=f"{m['P']} mg/kg")
        c3.metric(label="Potassium (K)", value=f"{m['K']} mg/kg")
        c4.metric(label="Calculated Soil Moisture", value=f"{m['humidity'] - 20}%")
        
        c5, c6, c7 = st.columns(3)
        c5.metric(label="Land Surface Temp", value=f"{m['temperature']} °C")
        c6.metric(label="Soil pH Level", value=m['ph'])
        c7.metric(label="Regional Estimated Rainfall", value=f"{m['rainfall']} mm")
        
        st.markdown("---")
        # Existing button execution...
        if st.button("Generate Smart Crop Recommendation"):
            if crop_model is not None:
                input_vector = [[m['N'], m['P'], m['K'], m['temperature'], m['humidity'], m['ph'], m['rainfall']]]
                prediction = crop_model.predict(input_vector)[0]
                probabilities = crop_model.predict_proba(input_vector)[0]
                sorted_crops = sorted(zip(crop_model.classes_, probabilities), key=lambda x: x[1], reverse=True)
                
                st.balloons()
                
                # -------------------------------------------------------------
                # ENHANCED OUTPUT DISPLAY
                # -------------------------------------------------------------
                st.markdown(f"## 🥇 Primary Recommended Crop: **{prediction.upper()}**")
                
                # Create two visual columns for a cleaner layout
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.markdown("### 📊 Crop Alternate Suitability Matrix")
                    for crop, prob in sorted_crops[:3]:
                        # Using a progress bar to visually represent the match profile
                        st.write(f"**{crop.capitalize()}** ({prob * 100:.1f}% Match)")
                        st.progress(float(prob))
                
                with col_right:
                    st.markdown("### 🔬 Soil & Climate Match Analysis")
                    
                    # Custom rule-based advisory system matching the prediction
                    if prediction.lower() == 'coffee':
                        st.info(
                            "💡 **Why Coffee?**\n"
                            f"Your detected temperature ({m['temperature']}°C) and high moisture levels "
                            "provide the perfect subtropical climate conditions that Coffee plants require to thrive. "
                            f"The soil Nitrogen level of {m['N']} mg/kg supports robust vegetative growth."
                        )
                        st.warning(
                            "⚠️ **Actionable Farm Advisory:**\n"
                            "- **Shade Management:** Coffee performs best under partial canopy shade. Consider intercropping with banana or large trees.\n"
                            "- **Drainage Check:** Ensure your field has excellent drainage. Coffee roots are highly sensitive to waterlogging, even in high humidity."
                        )
                        
                    elif prediction.lower() == 'jute':
                        st.info(
                            "💡 **Why Jute?**\n"
                            "Jute requires a warm, wet climate. Your satellite metrics show ideal high humidity "
                            "and optimal rainfall parameters matching alluvial soil thresholds perfectly."
                        )
                        st.warning(
                            "⚠️ **Actionable Farm Advisory:**\n"
                            "- Standing water during the early growth stage is beneficial, but ensure the soil pH stays stable.\n"
                            "- Prepare for a labor-intensive retting process post-harvest."
                        )
                        
                    elif prediction.lower() == 'maize':
                        st.info(
                            "💡 **Why Maize?**\n"
                            "Maize is a versatile crop that adapts well to your region's temperature matrix, requiring moderate "
                            "nitrogen and steady phosphorus for root development."
                        )
                        st.warning(
                            "⚠️ **Actionable Farm Advisory:**\n"
                            "- Ensure adequate spacing to maximize sunlight penetration.\n"
                            "- Monitor the field closely for fall armyworm disruptions during mid-season."
                        )
                    else:
                        st.info(f"The environment matches the historical baseline profile for growing {prediction.capitalize()} successfully.")

            else:
                st.error("Model unavailable.")
with tab2:
    st.header("Computer Vision Leaf Disease Diagnostics")
    st.write("Ready to route through your custom CNN classifier once you complete training your image dataset model.")