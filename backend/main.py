from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import os
from utils import preprocess_image, extract_textured_tire, is_tire_present

app = FastAPI(title="Tyre Wear Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
unet_path = os.path.join(MODEL_DIR, "unet_tire_wear_v3.h5")
cnn_path = os.path.join(MODEL_DIR, "cnn_classifier_rgb_v3.keras")

# Global variables for models
unet = None
cnn = None

@app.on_event("startup")
def load_models():
    global unet, cnn
    if os.path.exists(unet_path) and os.path.exists(cnn_path):
        unet = tf.keras.models.load_model(unet_path, compile=False)
        cnn = tf.keras.models.load_model(cnn_path, compile=False)
        print("Models loaded successfully.")
    else:
        print(f"Warning: Models not found in {MODEL_DIR}. Inference will fail.")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if unet is None or cnn is None:
        return {"error": "Models not loaded. Ensure .h5 and .keras files are in backend/models/"}

    contents = await file.read()
    
    # 1. Preprocess and Segment
    input_tensor = preprocess_image(contents)
    mask_pred = unet.predict(input_tensor)
    print(f"DEBUG: Segmentation complete for {file.filename}")
    
    # 1.5. Validate Tire Presence (Threshold at 15% of image area)
    if not is_tire_present(mask_pred, threshold=0.15):
        return {
            "error": "No tire detected in the image.",
            "code": "NO_TIRE_DETECTED",
            "status": "Invalid"
        }

    # 2. Extract Texture (Image * Mask)
    textured_input = extract_textured_tire(input_tensor, mask_pred)
    
    # 3. Classify with RGB CNN
    pred_prob = float(cnn.predict(textured_input)[0][0])
    
    life_percent = pred_prob * 100
    wear_percent = 100 - life_percent
    
    # Refined Health Logic for Mobile Users
    if life_percent < 15:
        # High likelihood of flat tire or severe damage
        status = "Flat/Damaged"
        remaining_km = 0
    elif life_percent < 50:
        status = "Defective"
        remaining_km = int(pred_prob * 40000)
    else:
        status = "Good"
        remaining_km = int(pred_prob * 40000)

    return {
        "filename": file.filename,
        "life_percentage": round(life_percent, 2),
        "wear_percentage": round(wear_percent, 2),
        "remaining_km": remaining_km,
        "status": status
    }

@app.get("/")
def read_root():
    return {"message": "Tyre Wear Detection API is running."}
