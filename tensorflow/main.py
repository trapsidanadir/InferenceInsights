from datetime import datetime
import json
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, UploadFile, status
from PIL import Image

import tensorflow as tf

MODEL_PATH = "./mobilenet_v2_tf.keras"  # Assuming you have a TensorFlow SavedModel
LABELS = {}
model_loading_time = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_PATH, LABELS, model_loading_time
    # Load model on startup
    model_loading_start_time = time.time()
    app.state.model = tf.keras.models.load_model(MODEL_PATH)
    with open("imagenet_class_index.json", "r") as f:
        LABELS = json.load(f)
    model_loading_time = time.time() - model_loading_start_time
    yield
    # Clean up resources on shutdown
    del app.state.model


app = FastAPI(lifespan=lifespan, title="Tensorflow API")


def preprocess_image(image):
    # Resize and normalize the image
    image = image.resize((224, 224))
    image = image.convert('RGB')
    img_array = np.array(image).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype(np.float32) / 127.5 - 1
    return img_array


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    # Read and preprocess the image
    image = Image.open(file.file).convert("RGB")
    input_data = preprocess_image(image)

    # Run inference
    results = app.state.model.predict(input_data)[0]

    top_prob = np.max(results, axis=0)
    top_class = np.argmax(results, axis=0)

    label = LABELS[str(top_class)][-1]  # Make sure LABELS is indexed correctly
    score = top_prob.item()  # Convert to a standard Python scalar if needed

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
        "message": "TensorFlow Object Detection API",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
    }
