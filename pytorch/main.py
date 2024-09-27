from datetime import datetime
import json
import time
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, status
from PIL import Image

MODEL_PATH = "./mobilenet_v2_jit_pt.pth"
LABELS = {}
model_loading_time = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_PATH, LABELS, model_loading_time
    # Load model on startup
    model_loading_start_time = time.time()
    app.state.model = torch.jit.load(MODEL_PATH)
    with open("imagenet_class_index.json", "r") as f:
        LABELS = json.load(f)
    model_loading_time = time.time() - model_loading_start_time
    yield
    # Clean up resources on shutdown
    del app.state.model


app = FastAPI(lifespan=lifespan, title="Pytorch API")


def preprocess_image(image):
    # Resize and normalize the image
    image = image.resize((256, 256)).crop((16, 16, 240, 240))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std = np.array([0.229, 0.224, 0.225])[:, None, None]
    return torch.from_numpy(((img_array - mean) / std).astype(np.float32)).unsqueeze(0)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    # Read and preprocess the image
    image = Image.open(file.file).convert("RGB")
    input_data = preprocess_image(image)

    # Run inference
    with torch.no_grad():
        results = app.state.model(input_data)

    # Get the predicted class probabilities
    probabilities = torch.nn.functional.softmax(results[0], dim=0)
    top_prob, top_class = torch.max(probabilities, 0)
    label = LABELS[str(top_class.item())][-1]
    score = top_prob.item()

    inference_time = time.time() - start_time

    return {
        "class": label,
        "score": score,
        "inference_time (s)": inference_time,
        "model_loading_time (s)": model_loading_time,
    }


@app.get("/", status_code=status.HTTP_200_OK)
@app.get("/health", status_code=status.HTTP_200_OK)
async def root():
    return {
        "message": "PyTorch Object Detection API",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
    }
