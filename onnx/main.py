import json
import time
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime
from fastapi import FastAPI, File, UploadFile
from PIL import Image

MODEL_PATH = "./mobilenet_v2_pt.onnx"
LABELS = {}
model_loading_time = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_PATH, LABELS, model_loading_time
    # Load model on startup
    model_loading_start_time = time.time()
    app.state.session = onnxruntime.InferenceSession(MODEL_PATH)
    with open("imagenet_class_index.json", "r") as f:
        LABELS = json.load(f)
    model_loading_time = time.time() - model_loading_start_time
    yield
    # Clean up resources on shutdown
    del app.state.session


app = FastAPI(lifespan=lifespan)


def preprocess_image(image):
    # Resize and normalize the image
    image = image.resize((224, 224))  # Adjust size as needed for your model
    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = np.transpose(image_array, (2, 0, 1))  # CHW format
    return np.expand_dims(image_array, axis=0)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()

    # Read and preprocess the image
    image = Image.open(file.file).convert("RGB")
    input_data = preprocess_image(image)

    # Get model inputs and outputs
    input_name = app.state.session.get_inputs()[0].name
    output_name = app.state.session.get_outputs()[0].name

    # Run inference
    results = app.state.session.run([output_name], {input_name: input_data})

    # Get the predicted class probabilities
    probabilities = np.exp(results[0][0]) / np.sum(np.exp(results[0][0]))

    label = LABELS[str(np.argsort(probabilities)[::-1][0])][-1]
    score = float(np.sort(probabilities)[::-1][0])

    inference_time = time.time() - start_time

    return {
        "class": label,
        "score": score,
        "inference_time (s)": inference_time,
        "model_loading_time (s)": model_loading_time,
    }


@app.get("/")
async def root():
    return {"message": "ONNX Object Detection API"}
