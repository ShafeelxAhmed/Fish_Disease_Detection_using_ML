from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from PIL import Image
import numpy as np
import os, requests

app = FastAPI()

IMG_SIZE = 256  

#  Store model INSIDE backend so Render can write
MODEL_PATH = "model.keras"
MODEL_URL = "https://huggingface.co/ShafeelXAhmed/fish-disease-detection-model/resolve/main/model.keras"

CLASS_NAMES = [
    "Bacterial Red disease",
    "Bacterial diseases - Aeromoniasis",
    "Bacterial gill disease",
    "Fungal diseases Saprolegniasis",
    "Healthy Fish",
    "Parasitic diseases",
    "Viral diseases White tail disease"
]

# ✅ Download model if missing
if not os.path.exists(MODEL_PATH):
    print("Model not found — downloading from HuggingFace...")
    response = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("✅ Model downloaded!")


print("Looking for model at:", MODEL_PATH)
print("Exists locally?:", os.path.exists(MODEL_PATH))

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded!")

from tensorflow.keras.applications.resnet50 import preprocess_input

def preprocess(image):
    image = image.convert("RGB")
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file)
        input_tensor = preprocess(img)
        preds = model.predict(input_tensor)[0]
        label = CLASS_NAMES[int(np.argmax(preds))]
        confidence = float(np.max(preds))
        return {"prediction": label, "confidence": confidence}
    except Exception as e:
        return {"error": str(e)}

    
#uvicorn app:app --reload

