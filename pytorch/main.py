from fastapi import FastAPI, File, UploadFile
from contextlib import asynccontextmanager
import torch
from torchvision.transforms import functional as F
from PIL import Image
import time

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    app.state.model = torch.jit.load("./mobilenet_v2-7ebf99e0.pth")
    app.state.model.eval()
    yield
    # Clean up resources on shutdown
    del app.state.model

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()

    # Read and preprocess the image
    image = Image.open(file.file).convert("RGB")
    image = image.resize((300, 300))  # Adjust size as needed for your model
    image_tensor = F.to_tensor(image).unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        predictions = app.state.model(image_tensor)
    
    # Process results (adjust based on your model's output format)
    boxes = predictions[0].tolist()
    scores = predictions[1].tolist()
    labels = predictions[2].tolist()
    
    inference_time = time.time() - start_time
    
    return {
        "boxes": boxes,
        "classes": labels,
        "scores": scores,
        "inference_time": inference_time
    }

@app.get("/")
async def root():
    return {"message": "PyTorch Object Detection API"}