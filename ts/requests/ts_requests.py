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

    # Here you should uncomment the T.Compose if you are not using
    # TorchServe `image_classifier` or `image_segmenter` handlers.
    # from torchvision import transforms as T
    # image_processing = T.Compose([
    #     T.Resize(256),
    #     T.CenterCrop(224),
    #     T.ToTensor(),
    #     T.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225]
    #     )])
    # raw_image = image_processing(raw_image)

    # Transform the PIL.Image into a bytes string (required for the inference)
    raw_image_bytes = BytesIO()
    raw_image.save(raw_image_bytes, format="PNG")
    raw_image_bytes.seek(0)

    return raw_image_bytes.read()


def predict(preprocessed_image_bytes):
    # Send HTTP Post request to TorchServe Inference API
    url = "http://127.0.0.1:8443/predictions/fastrcnn"
    req = requests.post(url, data=preprocessed_image_bytes)
    if req.status_code == 200:
        # Convert the output list into a torch.Tensor
        output = req.json()
        return output
    return None