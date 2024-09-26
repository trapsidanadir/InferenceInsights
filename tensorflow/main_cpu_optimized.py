import json
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image

import tensorflow as tf

MODEL_PATH = "./mobilenet_v2_tf"  # Assuming you have a TensorFlow SavedModel
LABELS = {}
model_loading_time = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_PATH, LABELS, model_loading_time
    # Load model on startup
    model_loading_start_time = time.time()

    # Configure TensorFlow to use CPU
    tf.config.set_visible_devices([], "GPU")

    app.state.model = tf.saved_model.load(MODEL_PATH)
    with open("imagenet_class_index.json", "r") as f:
        LABELS = json.load(f)
    model_loading_time = time.time() - model_loading_start_time
    yield
    # Clean up resources on shutdown
    del app.state.model


app = FastAPI(lifespan=lifespan)


def preprocess_image(image):
    # Resize and normalize the image
    image = image.resize((224, 224))  # Adjust size as needed for your model
    image_array = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()

    # Read and preprocess the image
    image = Image.open(file.file).convert("RGB")
    input_data = preprocess_image(image)

    # Run inference
    results = app.state.model(input_data)

    # Get the predicted class probabilities
    probabilities = tf.nn.softmax(results[0]).numpy()
    top_class = np.argmax(probabilities)
    top_prob = probabilities[top_class]

    label = LABELS[str(top_class)][-1]
    score = float(top_prob)

    inference_time = time.time() - start_time

    return {
        "class": label,
        "score": score,
        "inference_time (s)": inference_time,
        "model_loading_time (s)": model_loading_time,
    }


@app.get("/")
async def root():
    return {"message": "TensorFlow Object Detection API"}
