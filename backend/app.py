from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import requests

app = FastAPI()

# Model configuration
MODEL_URL = "https://huggingface.co/ShafeelXAhmed/fish-disease-detection-model/resolve/main/model.keras"
MODEL_PATH = "../model.keras"
CLASS_NAMES = [
    "Bacterial Red disease",
    "Bacterial diseases - Aeromoniasis",
    "Bacterial gill disease",
    "Fungal diseases Saprolegniasis",
    "Healthy Fish",
    "Parasitic diseases",
    "Viral diseases White tail disease"
]

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from HuggingFace...")
    with requests.get(MODEL_URL, stream=True) as r:
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Model downloaded!")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess(image):
    image = image.convert("RGB")
    img = image.resize((256, 256))  # Replace IMG_SIZE with correct value (like 256 or 224)
    img = np.array(img)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
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
    
#uvicorn app:app --reload

