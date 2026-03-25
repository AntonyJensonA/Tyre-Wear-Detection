from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import time
import random

app = FastAPI(title="Tyre Wear Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def predict_wear_mask(image_bytes):
    # PLACEHOLDER: Load your U-Net model from backend/models/
    # Run inference on the image
    time.sleep(1) # simulate inference time
    return {"status": "success", "mask_area_percentage": round(random.uniform(5.0, 35.0), 2)}

def predict_remaining_life(image_bytes):
    # PLACEHOLDER: Load your CNN Regression model from backend/models/
    # Run inference on the image
    time.sleep(1) # simulate inference time
    return {"status": "success", "remaining_life_km": random.randint(5000, 45000)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    
    # Run both model inferences (using placeholders for now)
    wear_result = predict_wear_mask(contents)
    life_result = predict_remaining_life(contents)
    
    return {
        "filename": file.filename,
        "wear_detection": wear_result,
        "remaining_life": life_result
    }

@app.get("/")
def read_root():
    return {"message": "Tyre Wear Detection API is running. Upload images to /predict endpoint."}
