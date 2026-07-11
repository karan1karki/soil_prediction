import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import os

DATASET_PATH = 'new_plant_diseases_dataset' 

print(f"📂 Dataset Path: {DATASET_PATH}")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5 

print("🔄 Loading dataset and preparing image generators...")


datagen = ImageDataGenerator(
    rescale=1.0/255,          
    validation_split=0.2,     
    rotation_range=20,     
    zoom_range=0.15,          
    horizontal_flip=True      
)

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'      
)

val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'       # 20% Validation Data
)

# Get the exact number of unique disease classes found in your folder
num_classes = train_generator.num_classes
print(f"📊 Detected {num_classes} distinct target classes: {list(train_generator.class_indices.keys())}")


print("🏗️ Fetching pre-trained MobileNetV2 base weights...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)  
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)              
predictions = Dense(num_classes, activation='softmax')(x) 

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("🚀 Starting Neural Network training loop...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)
MODEL_NAME = 'rice_leaf_cnn_model.h5' 
print(f"💾 Saving trained weights to disk as '{MODEL_NAME}'...")
model.save(MODEL_NAME)

print("🎯 Complete! Your .h5 file is generated and ready to be plugged into the FastAPI backend server.")
