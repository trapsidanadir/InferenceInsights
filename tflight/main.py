from fastapi import FastAPI, File, UploadFile
from contextlib import asynccontextmanager
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
import time

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    app.state.interpreter = tflite.Interpreter(model_path="path/to/your/mobilenetv2_ssd.tflite")
    app.state.interpreter.allocate_tensors()
    app.state.input_details = app.state.interpreter.get_input_details()
    app.state.output_details = app.state.interpreter.get_output_details()
    yield
    # Clean up resources on shutdown
    del app.state.interpreter

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    
    # Read and preprocess the image
    image = Image.open(file.file).resize((300, 300))
    input_data = np.expand_dims(np.array(image, dtype=np.uint8), axis=0)
    
    # Run inference
    app.state.interpreter.set_tensor(app.state.input_details[0]['index'], input_data)
    app.state.interpreter.invoke()
    
    # Get results
    boxes = app.state.interpreter.get_tensor(app.state.output_details[0]['index'])
    classes = app.state.interpreter.get_tensor(app.state.output_details[1]['index'])
    scores = app.state.interpreter.get_tensor(app.state.output_details[2]['index'])
    
    inference_time = time.time() - start_time
    
    return {
        "boxes": boxes.tolist(),
        "classes": classes.tolist(),
        "scores": scores.tolist(),
        "inference_time": inference_time
    }

@app.get("/")
async def root():
    return {"message": "TFLite Object Detection API"}