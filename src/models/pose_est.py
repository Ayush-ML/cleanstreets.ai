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
        self._lock = threading.Lock()
        self._latest_result: Optional[vision.HandLandmarkerResult] = None
        self._latest_ts_ms: Optional[int] = None
        model_path = self._get_model()
        options = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model_path)), running_mode=vision.RunningMode.LIVE_STREAM, num_hands=MAX_NUM_HANDS, min_hand_detection_confidence=HAND_DETECTION_CONFIDENCE,min_hand_presence_confidence=HAND_PRESENCE_CONFIDENCE,min_tracking_confidence=HAND_TRACKING_CONFIDENCE, result_callback=self._on_result)
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        
    @staticmethod
    def _get_model() -> Path:
        """
        Downloads the hand_landmarker.task model bundle once and caches
        it locally and avoids re-downloading on every construction.
        """   
        path = Path(HAND_MODEL_PATH)
        if path.exists():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HAND_MODEL_URL, str(path))
        return path
    
    def close(self) -> None:
        """
        Releases The Mediapipe's Landmarker once ALL frames are submitted
        """
        self.landmarker.close()
       
    def _on_result(self, result: vision.HandLandmarkerResult, output_image: mp.Image, ts_ms: int) -> None:
        """
        Calls on Internal Thread whenever a Frame finishes processing by Mediapipe, modelled as a Callback
        Args:
            result: the Result of the Processing
            output_image: The Image that is returned after processsing
            ts_ms: The Timestamp in Milliseconds
        """
        with self._lock:
            self._latest_result = result
            self._latest_ts_ms = ts_ms
            
    def submit(self, frame: np.ndarray, ts_ms: int) -> None:
        """
        Submit a Request for Detection to the Landmarker
        Args:
            frame: The Frame that is to be detected on
            ts_ms: The Timestamp n Milliseconds of the Frame
        """
        rgb = frame[:, :, ::-1]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.landmarker.detect_async(image, ts_ms)
        
    @staticmethod
    def _match_to_person(point: Tuple[float, float], people: Dict[int, Object], x: float, y: float) -> int | None:
        """
        Finds Which Persons Bounding Box continas the point that was passed in
        Args:
            point: The point that is passed in order to find which Bounding Box its located in
            persons: All The Data of the people and their Bounding Boxes
            x: Margin Expaned x
            y: Margin expaned y
        Returns:
            tid: The Tracker ID of the person who's Bounding box the point is located in or None if no possible candidates are found
        """
        px, py = point
        candidates = []
        for tid, person in people.items():
            if (person.x1 - x <= px <= person.x2 + x and person.y1 - y <= py <= person.y2 + y):
                cx, cy = person.center
                distance = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                candidates.append((distance, tid))
                
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    
    def get_latest_wrists(self, person_history: Dict[int, Dict[int, Object]], width: float, height: float) -> Dict[int, List[Tuple[float, float]]]:
        """
        Returns The Tracker ID and Wrist Points of the Latest Result that has arrived from the Landmarker
        Args:
            person_history: The History of the Person including thier positions at certain frames and their data
            width: The Width of the Frame
            height: The Height of the Frame
        Returns:
            wrists: The Wrist Coordinates of all People in the Frame
        """
        with self._lock:
            result = self._latest_result
            ts = self._latest_ts_ms
            
        if result is None or not result.hand_landmarks:
            return {}
        
        persons_at_ts = person_history.get(ts)
        if persons_at_ts is None:
            return {}
        
        return self._match_hands_to_people(result=result, people=persons_at_ts, width=width, height=height)
        
    def _match_hands_to_people(self, result: vision.HandLandmarkerResult, people: Dict[int, Object], width: float, height: float) -> Dict[int, List[Tuple[float, float]]]:
        """
        Matches The Landmarks of Hands to each person
        Args:
            result: The Result that the landmarker returned
            people: The Info on all the People in the frame
            width: The Width of the Frame
            height: The Height of the Frame
        Returns:
            wrists_by_track: a Dictionary contains all the Wrists matched by the Person's Tracker ID
        """
        wrists_by_track: Dict[int, List[Tuple[float, float]]] = {}
        mx = width * PERSON_MATCH_MARGIN_RATIO
        my = height * PERSON_MATCH_MARGIN_RATIO
        
        for hand_landmarks in result.hand_landmarks:
            wrist_landmark = hand_landmarks[WRIST_LANDMARK_IDX]
            wrist_px = (wrist_landmark.x * width, wrist_landmark.y * height)
            
            matched_tid = self._match_to_person(point=wrist_px, people=people, x=mx, y=my)
            if matched_tid is None:
                continue
            
            wrists_by_track.setdefault(matched_tid, []).append(wrist_px)
            
        return wrists_by_track
        