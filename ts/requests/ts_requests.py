import pickle
from PIL import Image
from io import BytesIO
import cv2
import torch
import requests
import numpy as np


def preprocess(img_path_or_buf):
    # Check whether image is a path or a buffer
    raw_image = (
        Image.fromarray(cv2.imread(img_path_or_buf))
        if isinstance(img_path_or_buf, str)
        else img_path_or_buf
    )

    # If buffer was np.array instead of PIL.Image, transform it
    if type(raw_image) == np.ndarray:
        raw_image = Image.fromarray(raw_image)

    # Converts the image to RGB
    raw_image = raw_image.convert("RGB")

    # Transform the PIL.Image into a bytes string (required for the inference)
    raw_image_bytes = BytesIO()
    raw_image.save(raw_image_bytes, format="PNG")
    raw_image_bytes.seek(0)

    return raw_image_bytes.read()


def preprocess_yolo(frame: np.ndarray):
    # Convert the bytes object to a BytesIO object
    arr_bytes = pickle.dumps(frame)
    return BytesIO(arr_bytes)


def predict(preprocessed_image_bytes):
    # Send HTTP Post request to TorchServe Inference API
    url = "http://127.0.0.1:8443/predictions/fastrcnn"
    req = requests.post(url, data=preprocessed_image_bytes)
    if req.status_code == 200:
        # Convert the output list into a torch.Tensor
        output = req.json()
        return output
    return None


def predict_yolo(preprocessed_image_bytes):
    req = requests.post(
        "http://localhost:5000/predict",
        files={"file": preprocessed_image_bytes},
    )
    if req.status_code == 200:
        # Convert the output list into a torch.Tensor
        output = req.json()
        return output
    return None
