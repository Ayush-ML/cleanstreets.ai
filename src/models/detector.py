# This Script is responsible for handling the YOLOv8 Nano Detector Model for Object Detection. 
# It loads the pre-trained YOLOv8 Nano model, performs inference on a frame, and returns the detected objects along with their metadata.
# It uses UltraLytic's model.track to perform object detection and tracking in real time
# Importing Necessary Libraries
from ultralytics import YOLO
import numpy as np
from src.core.config import MODEL_PATH, IOU, CONFIDENCE, CLASSES, FORMAT, NMS, SIMPLIFY, OPTIMIZE, QUANTIZE, VERBOSE
from typing import List
from src.models.objects import Object 
from pathlib import Path

class Detector:
    """
    A Class that handles the YOLOv8 Nano Detector Model for Object Detection
    Attributes:
        model: The YOLOv8 Nano Model for Object Detection
    """
    
    def __init__(self, model_name: str = MODEL_PATH) -> None:
        """
        Initializes the Detector Class
        Args:
            model_name: The Name of the Model to be used for Object Detection (default is MODEL_NAME from core/config.py)
        """
        self.path = self._get_model_path(model_name=model_name)
        self.model = YOLO(self.path)
        
    @staticmethod
    def _get_model_path(model_name: str) -> str:
        """
        Checks and gives the path of the model
        Args:
            model_name: The Name of the Model to be used for Object Detection 
        Returns:
            str: The path of the model
        """
        path = Path(model_name).with_suffix(f".{FORMAT}")
        if path.exists():
            return str(path)
        return YOLO(model_name).export(format=FORMAT, simplify=SIMPLIFY, dynamic=True, optimize=OPTIMIZE, nms=NMS, quantize=QUANTIZE)
        
    def detect(self, frame: np.ndarray) -> List[dict]:
        """
        Performs object detection on a frame and returns the detected objects along with their metadata
        Args:
            frame: The Frame on which object detection is to be performed
        Returns:
            List[dict]: A list of dictionaries containing the detected objects and their metadata
        """
        results = self.model.track(frame, persist=True, iou=IOU, conf=CONFIDENCE, verbose=VERBOSE, classes=CLASSES)
        objects: List[Object] = []
        if not results:
            return objects
        result = results[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return objects
        
        ids = boxes.id.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy()
        
        for i in range(len(ids)):
            class_id = int(class_ids[i])
            objects.append(Object(tracker_id=int(ids[i]), class_id=class_id, confidence=float(confs[i]), x1=float(xyxy[i][0]), y1=float(xyxy[i][1]), x2=float(xyxy[i][2]), y2=float(xyxy[i][3]), class_name=result.names[class_id]))
        return objects
    
    def reset(self) -> None:
        """
        Resets the model to its initial state
        """
        self.model = YOLO(self.path)
        