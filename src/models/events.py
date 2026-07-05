# This is the Script handles the Event Detector 
# It combines the Pose Estimation model with a rule based system that checks whether the person has actually littered
# It uses IOU based Bounding Box Overlapping as a fallback if the Pose Estimation model fails
# And it uses specific rules against Edge Cases and also combines multiple frames, not just one frames and requires them to agree before making a decision
# Importing Necessary Libraries
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple
from src.core.config import FRAME_HEIGHT, FRAME_WIDTH, MIN_HOLD_FRAMES, OVERLAP_THRESHOLD, WRIST_PROXIMITY_RATIO, EDGE_MARGIN_PERCENT, DOWNWARD_WIDNOW_FRAMES, DESCENT_THRESHOLD, GROUND_LINE_RATIO, SETTLED_CONFIRMATION_FRAMES, STILLNESS_RADIUS_PERCENT, STALE_HISTORY_FRAMES, PERSON_EXIT_FRAMES, CLASS_VOTE_FRAMES
from src.models.objects import Object

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
    ix2 = max(a.x2, b.x2)
    iy1 = max(a.y1, b.y1)
    iy2 = max(a.y2, b.y2)
    
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    
    union = a.width * a.height + b.width * b.height - inter
    if union < 0:
        return 0.0
    return inter / union

def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
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

@dataclass
class HoldRecord:
    """
    The Record of How long a Person has 'held' an object for before it actually counts as being held
    """
    frames: int = 0
    
@dataclass
class ClassHistory:
    """
    A Rolling Window that consists of the Class ID, Class Name and The Confidence that the model is in its classification
    The Most Common Classification across Multiple Frames is returned as the Standarized Label for it to make an actual Incident
    It uses a Counter to keep track of each Classification the model made and which class it belongs to
    """
    classifications: Deque[Tuple[int, str, float]] = field(default_factory=lambda: deque(CLASS_VOTE_FRAMES))
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
    
class EventChecker:
    """
    A State Based Video Stream Event Checker that checks whether the Littering event has actually occoured using rule based Logic
    It Combined Bounding Box over with Hand Keypoints, proximity and Class Stabilization
    """
    
    def __init__(self) -> None:
        """
        Literally Just calls reset()
        """  
        self.reset()
        
    def reset(self) -> None:
        """
        Resets all Dictionaries containing Data of the Video Stream into thier original Empty State
        Also used on Initialization of EventChecker
        """
        self._holds: Dict[Tuple[int, int], HoldRecord] = {}
        self._associations: Dict[int, Set[int]] = {}
        self._drops: Dict[Tuple[int, int], DropRecord] = {}
        self._person_last_seen: Dict[int, int] = {}
        self._triggered_pairs: Set[Tuple[int, int]] = set()
        self._class_history: Dict[int, ClassHistory] = {}
    
    @staticmethod
    def _is_holding(person: Object, object: Object, wrists: Optional[List[Tuple[float, float]]]) -> bool:
        """Uses Wrist Key Point Distance if given wrists for the person and object 
        else calculates IOU based on the Bounding Boxes of the person and the object to determine whether the person is holding the object
        Args:
            person: The Person holding the object
            object: The Object being held by the person
            wrists: An Optional List of the Coordinates of the Persons Wrists
        Returns:
            bool: A Boolean containing whether or not the person was holding the object
        """
        if wrists:
            object_center = object.center
            diagnol = max((object.width ** 2 + object.height ** 2) ** 0.5, 1.0)
            threshold = diagnol * WRIST_PROXIMITY_RATIO
            return any(_distance(w, object_center) <= threshold for w in wrists)
        
        return _iou(person, object) >= OVERLAP_THRESHOLD