import base64
import cv2
import numpy as np


def base64_to_ndarray(b64_data: str):
    """Convert base64 data to numpy array

    Args:
        b64_data (str): Base64 encoded data

    Returns:
        np.ndarray: Numpy array representing the image
    """
    image_bytes = base64.b64decode(b64_data)
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    image_np2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return image_np2


def bytes_to_ndarray(img_bytes: str):
    """Convert bytes to numpy array

    Args:
        img_bytes (str): Image bytes

    Returns:
        np.ndarray: Numpy array representing the image
    """
    image_array = np.frombuffer(img_bytes, dtype=np.uint8)
    image_np2 = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image_np2
