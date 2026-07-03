# This Script is responsible for handling the capture of Video from the USB Camera
# It also handles the Opening of a Saved Video File for Processing
# It uses OpenCV from Google to handle the Video Capture and Processing
# It uses a Context Manager to handle the Opening and Closing of the Video Capture Object gracefully
# Importing Necessary Libraries
from contextlib import contextmanager
import cv2
from typing import Optional, Tuple, Iterable

# Context Manager for Video Capture
@contextmanager
def capture_manager(cap: cv2.VideoCapture) -> Iterable[cv2.VideoCapture]:
    try:
        yield cap
    finally:
        cap.release()
        print("Context Manager Released the Camera Capture")