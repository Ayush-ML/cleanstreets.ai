# This Script handles The Pose Estimation Model of a Person in the Video
# It can Check whether the person is actually holding the object or not using Hand landmarks
# It uses Google's Mediapipe Library, specifically the Hands Module to draw Hand Landmarks and check if it is actually being hed or not
# This Helps reduce Edge Cases where a person is not actually holding the object but the model thinks it is because of the bounding boxes overlapping
# Importing Necessary Libraries
import mediapipe as mp
from typing import Tuple, Dict, List, Optional
import urllib.request
import numpy as np
from src.models.objects import Object
from pathlib import Path
import threading
from mediapipe.tasks.python import vision as vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from src.core.config import HAND_DETECTION_CONFIDENCE, HAND_MODEL_PATH, HAND_MODEL_URL, HAND_PRESENCE_CONFIDENCE, HAND_TRACKING_CONFIDENCE, PERSON_MATCH_MARGIN_RATIO, WRIST_LANDMARK_IDX, MAX_NUM_HANDS

# the Class that Handles The Pose Estimation
class PoseEstimator:
    """
    Loads MediaPipe's HandLandmarker once and extracts wrist points per
    frame, matched to already-tracked persons by containment.
    """
    
    def __init__(self) -> None:
        """
        Creates The LandMarker and Also the Asynchronus Threading Lock as well as stores the latest result and Timestamp
        This is done because Pipeline needs this two information as it runs in a different thread and frame may not be accessible
        """
        model_path = self._get_model()
        options = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model_path)), running_mode=vision.RunningMode.LIVE_STREAM, num_hands=MAX_NUM_HANDS, min_hand_detection_confidence=HAND_DETECTION_CONFIDENCE,min_hand_presence_confidence=HAND_PRESENCE_CONFIDENCE,min_tracking_confidence=HAND_TRACKING_CONFIDENCE)
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._lock = threading.Lock()
        self._latest_result: Optional[vision.HandLandmarkerResult] = None
        self._latest_ts_ms = Optional[int] = None
        
    def _get_model(self) -> Path:
        """
        Downloads the hand_landmarker.task model bundle once and caches
        it locally and avoids re-downloading on every construction.
        """   
        path = Path(HAND_MODEL_PATH)
        if path.exists():
            return path
        path.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HAND_MODEL_URL, str(path))
        return path
    
    def close(self) -> None:
        """
        Releases The Mediapipe's Landmarker once ALL frames are submitted
        """
        self.landmarker.close()