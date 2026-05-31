# backend/train_models.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ONNX Conversion Imports
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

# 1. Load your CSV dataset
df = pd.read_csv('data/soil_data.csv')

# Clean column white-spaces if any exist
df.columns = df.columns.str.strip()

# -------------------------------------------------------------------------
# 2. Setup Task A: Random Forest Classifier for 'Soil Type'
# -------------------------------------------------------------------------
X_soil = df[['Temparature', 'Humidity', 'Nitrogen', 'Phosphorous', 'Potassium']]
y_soil = df['Soil Type']

le_soil = LabelEncoder()
y_soil_encoded = le_soil.fit_transform(y_soil)

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_soil, y_soil_encoded, test_size=0.2, random_state=42)
soil_model = RandomForestClassifier(n_estimators=100, random_state=42)
soil_model.fit(X_train_s, y_train_s)

# -------------------------------------------------------------------------
# 3. Setup Task B: Random Forest Regressor for 'Moisture'
# -------------------------------------------------------------------------
X_moist = df[['Temparature', 'Humidity', 'Nitrogen', 'Phosphorous', 'Potassium']]
y_moist = df['Moisture']

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_moist, y_moist, test_size=0.2, random_state=42)
moisture_model = RandomForestRegressor(n_estimators=100, random_state=42)
moisture_model.fit(X_train_m, y_train_m)

# -------------------------------------------------------------------------
# 4. Setup Task C: Predict Fertilizer Recommendation
# -------------------------------------------------------------------------
X_fert = df[['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium']]
y_fert = df['Fertilizer Name']

le_fert = LabelEncoder()
y_fert_encoded = le_fert.fit_transform(y_fert)

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_fert, y_fert_encoded, test_size=0.2, random_state=42)
fert_model = RandomForestClassifier(n_estimators=100, random_state=42)
fert_model.fit(X_train_f, y_train_f)


# -------------------------------------------------------------------------
# 5. Export artifacts to ONNX and JSON/Joblib Encoders
# -------------------------------------------------------------------------

# Define input shapes for conversion 
# Both Soil and Moisture models take 5 float features: [Temparature, Humidity, Nitrogen, Phosphorous, Potassium]
initial_type_5 = [('float_input', FloatTensorType([None, 5]))]

# Fertilizer model takes 6 float features: [Temparature, Humidity, Moisture, Nitrogen, Phosphorous, Potassium]
initial_type_6 = [('float_input', FloatTensorType([None, 6]))]

# Convert and write Soil Type Classifier to ONNX
onnx_soil = to_onnx(soil_model, initial_types=initial_type_5, target_opset=15)
with open("rf_soil_classifier.onnx", "wb") as f:
    f.write(onnx_soil.SerializeToString())

# Convert and write Moisture Regressor to ONNX
onnx_moisture = to_onnx(moisture_model, initial_types=initial_type_5, target_opset=15)
with open("rf_moisture_regressor.onnx", "wb") as f:
    f.write(onnx_moisture.SerializeToString())

# Convert and write Fertilizer Classifier to ONNX
onnx_fert = to_onnx(fert_model, initial_types=initial_type_6, target_opset=15)
with open("rf_fertilizer_classifier.onnx", "wb") as f:
    f.write(onnx_fert.SerializeToString())

# Note: ONNX cannot save string LabelEncoders. We still save these small 
# encoders using joblib so we can map numerical model indexes back to string text labels.
joblib.dump(le_soil, 'label_encoder_soil.pkl')
joblib.dump(le_fert, 'label_encoder_fertilizer.pkl')

print("🎉 All Random Forest models converted to ONNX and successfully exported!")