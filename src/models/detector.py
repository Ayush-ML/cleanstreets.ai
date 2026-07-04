# This Script is responsible for handling the YOLOv8 Nano Detector Model for Object Detection. 
# It loads the pre-trained YOLOv8 Nano model, performs inference on a frame, and returns the detected objects along with their metadata.
# It uses UltraLytic's model.track to perform object detection and tracking in real time
# Importing Necessary Libraries
from ultralytics import YOLO
import numpy as np
from src.core.config import MODEL_NAME, IOU, CONFIDENCE, CLASSES
from typing import List

# The Detector Class that handles everything
class Detector:
    """
    A Class that handles the YOLOv8 Nano Detector Model for Object Detection
    Attributes:
        model: The YOLOv8 Nano Model for Object Detection
    """
    
    def __init__(self, model_name: str = MODEL_NAME, stream: bool = True) -> None:
        """
        Initializes the Detector Class
        Args:
            model_name: The Name of the Model to be used for Object Detection (default is MODEL_NAME from core/config.py)
            stream: Whether to stream the results or not (default is True)
        """
        self.model = YOLO(model_name)
        self.stream = stream    
        
    def detect(self, frame: np.ndarray) -> List[dict]:
        """
        Performs object detection on a frame and returns the detected objects along with their metadata
        Args:
            frame: The Frame on which object detection is to be performed
        Returns:
            List[dict]: A list of dictionaries containing the detected objects and their metadata
        """
        results = self.model.track(frame, persist=True, stream=self.stream, iou=IOU, conf=CONFIDENCE, verbose=False, classes=CLASSES)