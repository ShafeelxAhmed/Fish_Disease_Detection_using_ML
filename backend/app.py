from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input

app = FastAPI()

# Load model
model = tf.keras.models.load_model("../model.keras")

# Class labels (your fish disease dataset classes)
CLASS_NAMES = [
    "Bacterial Red disease",
    "Bacterial diseases - Aeromoniasis",
    "Bacterial gill disease",
    "Fungal diseases Saprolegniasis",
    "Healthy Fish",
    "Parasitic diseases",
    "Viral diseases White tail disease"
]

def preprocess(image):
    # Convert to RGB to avoid issues with grayscale/4-channel images
    image = image.convert("RGB")
    
    img = image.resize((256, 256))  # model input size
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file)

        input_tensor = preprocess(img)
        preds = model.predict(input_tensor)[0]
        
        label_index = int(np.argmax(preds))
        confidence = float(np.max(preds))

        return {
            "prediction": CLASS_NAMES[label_index],
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}