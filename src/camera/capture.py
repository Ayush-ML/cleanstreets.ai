# This Script is responsible for handling the capture of Video from the USB Camera
# It uses OpenCV from Google to handle the Video Capture and Processing
# It uses a Context Manager to handle the Opening and Closing of the Video Capture Object gracefully
# Importing Necessary Libraries
from contextlib import contextmanager
import cv2
from typing import Tuple, Iterable
from pathlib import Path
import numpy as np
from src.core.config import FRAME_WIDTH, FRAME_HEIGHT

@contextmanager
def capture_manager(cap: cv2.VideoCapture) -> Iterable[cv2.VideoCapture]:
    """
    Context Manager for Video Capture
    Args:
        cap: The Video Capture Object to be managed
    Yields:
        cap: The Video Capture Object as an Iterable to loop through
    """
    try:
        yield cap
    except Exception as e:
        print(f"Function: capture_manager, returned an error when yielding Video Capture Object: {e}")
    finally:
        cap.release()
        print("Context Manager Released the Camera Capture")
        
def open_camera(camera_id: int = 0) -> Iterable[Tuple[int, np.ndarray]]:
    """
    Function to open a camera for processing
    Args:
        camera_id: int: The ID of the camera to be opened (default is 0)
    Yields:
        frame_n, frame: An iterator yielding frame numbers and frame data
    """
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    with capture_manager(cap=cap):
        frame_n = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame_n, frame
            frame_n += 1