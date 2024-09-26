import io
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
    global LABELS, model_loading_time
    model_loading_start_time = time.time()

    # Optimize for CPU
    session_options = onnxruntime.SessionOptions()
    session_options.intra_op_num_threads = 4  # Adjust based on your CPU cores
    session_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    app.state.session = onnxruntime.InferenceSession(MODEL_PATH, session_options)

    # Load labels
    with open("imagenet_class_index.json", "r") as f:
        LABELS = json.load(f)

    # Pre-fetch input and output names
    app.state.input_name = app.state.session.get_inputs()[0].name
    app.state.output_name = app.state.session.get_outputs()[0].name

    model_loading_time = time.time() - model_loading_start_time
    yield
    del app.state.session


app = FastAPI(lifespan=lifespan)


def preprocess_image(image):
    image = image.resize((224, 224), Image.BILINEAR)
    image_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array.transpose(2, 0, 1), axis=0)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()

    # Read image without saving to disk
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    input_data = preprocess_image(image)

    # Run inference
    results = app.state.session.run(
        [app.state.output_name], {app.state.input_name: input_data}
    )

    # Use softmax for better numerical stability
    probabilities = np.exp(results[0][0] - np.max(results[0][0]))
    probabilities /= np.sum(probabilities)

    # Get top prediction
    top_idx = np.argmax(probabilities)
    label = LABELS[str(top_idx)][-1]
    score = float(probabilities[top_idx])

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
