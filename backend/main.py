from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import os
from utils import preprocess_image, is_tire_present, detect_objects_yolo, CBAMLayer, calculate_tire_wear_v2

app = FastAPI(title="Tyre Wear Detection API (V4 - CBAM)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
# The new CBAM model filename
unet_path = os.path.join(MODEL_DIR, "cbam_unet_tire_v4.keras")

# Global variables for models
unet = None

@app.on_event("startup")
def load_models():
    global unet
    if os.path.exists(unet_path):
        # Register CBAMLayer so Keras can load it
        unet = tf.keras.models.load_model(
            unet_path, 
            custom_objects={'CBAMLayer': CBAMLayer},
            compile=False
        )
        print("CBAM-U-Net Model loaded successfully.")
    else:
        print(f"Warning: Model not found at {unet_path}. Inference will fail.")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if unet is None:
        return {"error": "Model not loaded. Ensure cbam_unet_tire_v4.keras is in backend/models/"}

    contents = await file.read()
    
    # 0. YOLOv8 Pre-filtering
    is_filtered, detected = detect_objects_yolo(contents)
    if is_filtered:
        return {
            "error": f"Invalid image detected: {', '.join(detected)}.",
            "code": "FORBIDDEN_OBJECT_DETECTED",
            "status": "Invalid"
        }
    
    # 1. Preprocess and Segment
    input_tensor = preprocess_image(contents)
    mask_pred = unet.predict(input_tensor)
    
    # 1.5. Validate Tire Presence (Lowered to 1% for bald tires)
    if not is_tire_present(mask_pred, threshold=0.01):
        return {
            "error": "The tire appears too worn or no tire was detected.",
            "code": "NO_TIRE_DETECTED",
            "status": "Critical/Worn"
        }

    # 2. Precision Wear Calculation (Edge Density + Groove Analysis)
    wear_percent = calculate_tire_wear_v2(input_tensor, mask_pred)
    life_percent = 100 - wear_percent
    
    # 3. Status Logic
    if life_percent < 20: 
        status = "Critical/Flat"
        remaining_km = 0
    elif life_percent < 45:
        status = "Worn Out"
        remaining_km = int((life_percent/100) * 35000)
    elif life_percent < 75:
        status = "Moderate"
        remaining_km = int((life_percent/100) * 35000)
    else:
        status = "Good Condition"
        remaining_km = int((life_percent/100) * 35000)

    return {
        "filename": file.filename,
        "life_percentage": round(life_percent, 2),
        "wear_percentage": round(wear_percent, 2),
        "remaining_km": remaining_km,
        "status": status,
        "model_version": "v4_cbam_structural"
    }

@app.get("/")
def read_root():
    return {"message": "Tyre Wear Detection API is running."}
