# This Script is responsible for Wiring together every component of the system like Detector, Event Manager etc to Actually run it
# Its only job is to call each compenent in the right order and make sure that they all align with each other correctly
# This script is what the streamlit dashboard relies on for the reviewer
# It also handling the saving of the incident to the disk
# Importing Necessary Libraries
import threading, numpy as np
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union
from src.camera.buffer import RollingBuffer
from src.camera.capture import get_fps, open_camera
from src.core.config import FPS, INCIDENT_DIR, POSE_HISTORY_WINDOW_SECONDS
from src.models.detector import Detector
from src.models.events import EventChecker
from src.core.incidents import IncidentStorage
from src.models.objects import Object
from src.models.pose_est import PoseEstimator

class Pipeline:
    """
    Handles the Full Pipeline with the Live Camera
    It coordinates each class in the project and makes them work together in order to build the full system
    This is meant to run continously until stop and if stopped, use the reset() command to reset
    """
    
    def __init__(self) -> None:
        """
        Sets up every processing module needed to run with the live camera.
        Args:
            _latest_frame: An Optional Arguement passed by the Stremlit dashboard containing the current frame
        """
        self.fps = FPS
        self.detector = Detector()
        self.pose_estimator = PoseEstimator()
        self.event_detector = EventChecker()
        self.buffer = RollingBuffer()
        self.incidents = IncidentStorage()
        
        self.history: Dict[int, Dict[int, Object]] = {}
        
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_count: int = 0
        
    def reset(self) -> None:
        """
        Clears the Entire State of the Pipeline so it is prepared for a complete fresh start
        ensures that previous pipelines data does not bleed into the new one
        does not redownload the models and resuses them to save storage
        """
        self.detector.reset()
        self.event_detector.reset()
        self.buffer.clear()
        self.history.clear()
        
        with self._lock:
            self._latest_count = 0
            self._latest_frame = None
            
    def delete_persons_history(self, ts_ms: int) -> None:
        """
        Removes a person's position history which exceeds a given time frame, current 2 seconds
        Done to prevent the History from growing larger and larger without stopping
        Args:
            ts_ms: The Current Frame Timestamp in Milliseconds
        """
        window = int(POSE_HISTORY_WINDOW_SECONDS * 1000)
        cutoff = ts_ms - window
        stale = [ts for ts in self.history if ts < cutoff]
        for ts in stale:
            del self.history[ts]
            
    def latest_frame(self) -> Tuple[np.ndarray | None, int]:
        """
        Returns the Latest Frame and that Frame's Count
        Returns:
            _latest_frame and count: The Latest Frame as a np.ndarray and its count
        """
        with self._lock:
            return self._latest_frame, self._latest_count
        
    def run(self, should_stop: Optional[Callable[[], bool]]) -> None:
        """
        Runs the Full Detection, Tracking, Pose Estimation, Event Checking with the live camera
        Args:
            should_stop: An Optional Arguement passed by the Streamlit dashboard that is a function that returns true if pipeline should stop, else false
        """
        source: Iterator[Tuple[int, np.ndarray]] = open_camera()
        
        try:
            for frame_n, frame in source:
                if should_stop is not None and should_stop():
                    break
                
                height, width = frame.shape[0], frame.shape[1]
                
                ts_s = frame_n / self.fps
                ts_ms = int(ts_s * 1000)
                
                self.buffer.add_frame(frame=frame, timestamp=ts_s)
                objects: List[Object] = self.detector.detect(frame=frame)
                persons = {obj.tracker_id: obj for obj in objects if obj.is_person}
                
                with self._lock:
                    self._latest_count = len(objects)
                    self._latest_frame = frame
                    
                try:
                    self.pose_estimator.submit(frame=frame, ts_ms=ts_ms)
                except Exception as e:
                    print(f"Got a Exception when detecting Poses, defaulting to None, {e}")
                    pass
                
                self.history[ts_ms] = persons
                self.delete_persons_history(ts_ms=ts_ms)
                
                try:
                    wrists = self.pose_estimator.get_latest_wrists(person_history=self.history, height=height, width=width)
                except Exception as e:
                    print(f"Got an Exception while Getting Latest Pose Results, defaulting to None, {e}")
                    wrists = {}
                    
                incident = self.event_detector.check(frame_n=frame_n, items=objects, person_wrists=wrists)
                if incident is not None:
                    record = self.incidents.save(incident=incident)
                    path = Path(INCIDENT_DIR) / record['video']
                    self.buffer.save_clip(path=path)
                    self.buffer.clear()
                    print(f"Incident Recorded, ID: {record['id']}, Object: {record['object']}, Confidence: {record['confidence']}")
                    
        finally:
            source.close()
            self.pose_estimator.close()