# This Script is responsible for storing all of the Helper Functions, Classes and any small scripts that the main project needs
# It helps to modularize the Code, Keep it human readable and also sort plain and messy Types into clean variables
# An Example is the Point class in this script is something that this project often uses and i decided to write it in this script instead of keeping it as Tuple[float, float]
# Helper Functions like the Budhayana Pythagorean Distance Formula Between 2 Points are also present here
# Importing Necessary Libraries
from typing import Dict, Tuple, Deque, Optional, Counter, Any
from src.models.objects import Object
from src.core.config import CLASS_VOTE_FRAMES, DOWNWARD_WIDNOW_FRAMES, EDGE_MARGIN_PERCENT
from collections import deque
from dataclasses import dataclass, field
import numpy as np
from src.core.types import Point


@dataclass
class HoldState:
    """
    The Record of How long a Person has 'held' an object for before it actually counts as being held
    """
    frames: int = 0
    missed_count: int = 0
    confirmed_count: int = 0
    release_count: int = 0
@dataclass
class Incident:
    """
    A Candidate for a Littering Event that has been confirmed by Rule Based Decision Logic
    """
    pid: int
    obj_id: int
    class_name: str
    confidence: float
    frame_n: int
    
@dataclass
class DropRecord:
    """
    The Record where the person released the object they were holding and dropping is current in progress
    It is just a Candidate that it is used ot decide whether it actually dropped or not
    """
    pid: int
    obj_id: int
    drop_frame: int
    near_edge: bool
    recent_centers: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=DOWNWARD_WIDNOW_FRAMES))
    settled: Optional[int] = None
    settle_anchor: Optional[Tuple[float, float]] = None
    last_seen: int = 0
    missing_since: Optional[int] = None
@dataclass
class ClassHistory:
    """
    A Rolling Window that consists of the Class ID, Class Name and The Confidence that the model is in its classification
    The Most Common Classification across Multiple Frames is returned as the Standarized Label for it to make an actual Incident
    It uses a Counter to keep track of each Classification the model made and which class it belongs to
    """
    classifications: Deque[Tuple[int, str, float]] = field(default_factory=lambda: deque(maxlen=CLASS_VOTE_FRAMES))
    last_seen: int = 0
    
    def add(self, class_id: int, class_name: str, confidence: float, frame_n: int) -> None:
        """
        Adds a Given Record to the Classifications and sets the Classes Last Seeb Frame to the current Frame
        Args:
            class_id: The ID of the Class
            class_name: The Name of the Class
            confidence: How Confident the model is with its classifications
            frame_n: The Current Frame
        """
        self.classifications.append((class_id, class_name, confidence))
        self.last_seen = frame_n
        
    def stable_classification(self) -> Tuple[str, float]:
        """
        Takes the Most Common classification from all the classification along with the classifications average confidence score
        Returns:
            name, avg_confidence: A Tuple Containing the Name of the Most Common Classification and The Models Average Classification score on that Classification
        """
        if self.classifications is None:
            return "unknown", 0.0
        
        counts = Counter(classification[0] for classification in self.classifications)
        winners_id, _ = counts.most_common(1)[0]
        matching = [classification for classification in self.classifications if classification[0] == winners_id]
        avg_confidence = sum(classification[2] for classification in matching) / len(matching)
        name = matching[-1][1]
        return name, avg_confidence
    
def _iou(a: Object, b: Object) -> float:
    """
    Calculates the IOU or The Overlap/intersection of The Bounding Boxes
    Args:
        a: The First Object
        b: The Second Object
    Returns:
        iou: A Float Value containing how much both the objects Overlap with eachother
    """
    ix1 = max(a.x1, b.x1)
    ix2 = min(a.x2, b.x2)
    iy1 = max(a.y1, b.y1)
    iy2 = min(a.y2, b.y2)
    
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    
    union = a.width * a.height + b.width * b.height - inter
    if union < 0:
        return 0.0
    return inter / union

def _distance(a: Point, b: Point) -> float:
    """
    Returns the Distance between 2 points using the Baudhayana Pythagoras Theorum
    Args:
        a: The First Point
        b: The Second Point
    Returns:
        dist: A float containing the Distance between the 2 points
    """
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

def _near_edge(object: Object, width: float, height: float) -> bool:
    """
    Checks whether an Object is near the Edge of the Frame or not
    Args:
        object: The Object that is checked whether it is near the edge or not
        width: The Width of the Frame
        height: The Height of the Frame
    Returns:
        near_edge: A Boolean containing whether it is near the edge or not
    """
    mx = width * EDGE_MARGIN_PERCENT
    my = height * EDGE_MARGIN_PERCENT
    near_edge = (object.x1 <= mx or object.y1 <= my or object.x2 >= width - mx or object.y2 >= height - my)
    return near_edge

History = Dict[int, Object]

Camera = Tuple[int, np.ndarray]

IncidentDict = Dict[str, Any]