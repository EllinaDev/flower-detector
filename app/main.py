"""
main.py: FastAPI production-grade inference server for Flower Recognition.
Part of the 'app/' directory for Clean Architecture.
"""

import io
import json
import logging
import os
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import models, transforms
from typing import Dict, Optional

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Flower Recognition API", version="1.0.1")

# CORS middleware for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Robust path resolution for Clean Architecture
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "flower_model_v2.pth")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "..", "data", "mappings", "label_mapping.json")

# Device selection (optimized for Mac)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

logger.info(f"Inference device: {DEVICE}")

# Global variables for model and mapping
MODEL: Optional[nn.Module] = None
LABEL_MAP: Optional[Dict[int, str]] = None

def load_model_assets():
    """Loads the model and label mapping from disk."""
    global MODEL, LABEL_MAP
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_MAP_PATH):
        logger.error(f"Model artifacts not found at {MODEL_PATH} or {LABEL_MAP_PATH}. Please run training first.")
        return

    try:
        # Load Label Mapping
        with open(LABEL_MAP_PATH, "r") as f:
            raw_map = json.load(f)
            # JSON keys are strings, convert to int
            LABEL_MAP = {int(k): v for k, v in raw_map.items()}
        
        # Initialize Architecture, MODEL LOADING
        model = models.mobilenet_v2(weights=None) # Start with an empty brain
        
        # Determine number of classes from mapping
        num_classes = len(LABEL_MAP)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)    # Add the 102-flower "head"
        
        # Load State Dict
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE) # Load your saved knowledge
        
        # Handle both raw state_dicts and checkpoint dictionaries
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])       # Fill the brain with knowledge
        else:
            model.load_state_dict(checkpoint)
        model.to(DEVICE)
        model.eval() # IT TELS MODEL STOP LEARNING AND JUST USE WHAT IT KNOWS ALREADY
        
        MODEL = model
        logger.info("Model and label mapping loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

@app.on_event("startup")
async def startup_event():
    load_model_assets()

def get_inference_transforms():
    """Returns standard preprocessing transforms for inference."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "device": str(DEVICE), 
        "model_loaded": MODEL is not None,
        "classes": len(LABEL_MAP) if LABEL_MAP else 0
    }

# THE PREDICTION LOGIC
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        load_model_assets()
        if MODEL is None:
            raise HTTPException(status_code=503, detail="Model not loaded or training artifacts missing.")

    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        logger.error(f"Image load error: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Preprocess
    preprocess = get_inference_transforms()
    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        outputs = MODEL(input_tensor)  # The moment of classification
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, index = torch.max(probabilities, 0)
        
    class_id = index.item()
    conf_score = confidence.item()
    flower_name = LABEL_MAP.get(class_id, f"Unknown (ID: {class_id})").upper()

    return {
        "flower_name": flower_name,
        "confidence": conf_score,
        "class_id": class_id
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
