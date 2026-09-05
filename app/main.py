"""
main.py: FastAPI production-grade inference server for Flower Recognition.
"""

import os
import logging
import asyncio
# 1. Load environment variables IMMEDIATELY before any other imports
try:
    from dotenv import load_dotenv
    # Search for .env in current or parent folder, OVERRIDE ensures new keys are picked up
    env_loaded = load_dotenv(override=True) or load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)
except ImportError:
    pass

import io
import json
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from torchvision import models, transforms
from typing import Dict, Optional

# 2. Now import RAG engine (it will see the environment variables)
try:
    from app.rag_chat import rag_answer
except ImportError:
    from rag_chat import rag_answer

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Flower Recognition API", version="1.0.1")

# CORS middleware for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Robust path resolution for Clean Architecture
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "flower_model_v2.pth")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "..", "data", "mappings", "label_mapping.json")
MIN_RECOGNITION_CONFIDENCE = float(os.getenv("MIN_RECOGNITION_CONFIDENCE", "0.60"))
MIN_RECOGNITION_MARGIN = float(os.getenv("MIN_RECOGNITION_MARGIN", "0.12"))

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
model_lock = asyncio.Lock()

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
    async with model_lock:
        load_model_assets()

def get_inference_transforms():
    """Returns standard preprocessing transforms for inference."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_prediction_payload(class_id: int, flower_name: str, confidence: float, margin: float) -> Dict[str, object]:
    """Return either a recognized flower or an honest unknown result."""
    recognized = confidence >= MIN_RECOGNITION_CONFIDENCE and margin >= MIN_RECOGNITION_MARGIN

    if not recognized:
        return {
            "recognized": False,
            "flower_name": "UNKNOWN",
            "confidence": confidence,
            "class_id": class_id,
            "margin": margin,
            "message": "I cannot confidently recognize this image. The image may show something that is not a flower, or this flower may not be in my database.",
        }

    return {
        "recognized": True,
        "flower_name": flower_name.upper(),
        "confidence": confidence,
        "class_id": class_id,
        "margin": margin,
        "message": "Flower recognized.",
    }

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
    async with model_lock:
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
    async with model_lock:
        with torch.no_grad():
            outputs = MODEL(input_tensor)  # The moment of classification
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top_values, top_indices = torch.topk(probabilities, k=min(3, probabilities.numel()))
            confidence = top_values[0]
            index = top_indices[0]
            runner_up = top_values[1] if top_values.numel() > 1 else torch.tensor(0.0, device=DEVICE)
        
    class_id = index.item()
    conf_score = confidence.item()
    margin = (confidence - runner_up).item()
    flower_name = LABEL_MAP.get(class_id, f"Unknown (ID: {class_id})")
    response = build_prediction_payload(class_id, flower_name, conf_score, margin)
    response["top_predictions"] = [
        {
            "class_id": idx.item(),
            "flower_name": LABEL_MAP.get(idx.item(), f"Unknown (ID: {idx.item()})").upper(),
            "confidence": value.item(),
        }
        for value, idx in zip(top_values, top_indices)
    ]
    return response

# ─── RAG Chatbot Endpoint ───────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    RAG pipeline:
      1. User question received
      2. DuckDuckGo fetches live web data
      3. Context (DuckDuckGo results) combined
      4. Gemini LLM generates the final answer
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = await rag_answer(request.question.strip())
        return result
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate response.")


# Serve static files
STATIC_DIR = os.path.join(BASE_DIR, "..", "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def read_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "BloomIQ API is running. Static files not found."}

@app.get("/styles.css")
async def read_styles():
    styles_path = os.path.join(STATIC_DIR, "styles.css")
    if os.path.exists(styles_path):
        return FileResponse(styles_path, media_type="text/css")
    raise HTTPException(status_code=404, detail="styles.css not found")

@app.get("/app.js")
async def read_js():
    js_path = os.path.join(STATIC_DIR, "app.js")
    if os.path.exists(js_path):
        return FileResponse(js_path, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="app.js not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
