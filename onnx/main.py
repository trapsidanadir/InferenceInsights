from fastapi import FastAPI, File, UploadFile
from contextlib import asynccontextmanager
import onnxruntime
import numpy as np
from PIL import Image
import time

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    app.state.session = onnxruntime.InferenceSession("path/to/your/mobilenetv2_ssd.onnx")
    yield
    # Clean up resources on shutdown
    del app.state.session

app = FastAPI(lifespan=lifespan)

def preprocess_image(image):
    # Resize and normalize the image
    image = image.resize((300, 300))  # Adjust size as needed for your model
    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = np.transpose(image_array, (2, 0, 1))  # CHW format
    return np.expand_dims(image_array, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    
    # Read and preprocess the image
    image = Image.open(file.file).convert("RGB")
    input_data = preprocess_image(image)
    
    # Get model inputs
    input_name = app.state.session.get_inputs()[0].name
    
    # Run inference
    results = app.state.session.run(None, {input_name: input_data})
    
    # Process results (adjust based on your model's output format)
    boxes, labels, scores = results[:3]
    
    inference_time = time.time() - start_time
    
    return {
        "boxes": boxes.tolist(),
        "classes": labels.tolist(),
        "scores": scores.tolist(),
        "inference_time": inference_time
    }

@app.get("/")
async def root():
    return {"message": "ONNX Object Detection API"}