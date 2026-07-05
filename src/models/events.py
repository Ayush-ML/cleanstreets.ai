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
    
    def check(self, frame_n: int, items: List[Object], person_wrists: Optional[Dict[int, List[Tuple[float, float]]]] = None) -> Optional[Incident]:
        """
        Update the State with frames Objects and return a Incident if a Littering Event was detected
        Args:
            frame_n: The Current frame
            objects: All Tracked Objects im the frame
            wrists: An Options Dictionary of a Person's Tracker ID and the Coordinates of their wrists
        Returns:
            Incident: returns an Incident class with all info if an Incident occoured
        """
        person_wrists = person_wrists or {}
        
        people = {obj.tracker_id: obj for obj in items if obj.is_person}
        objects = {obj.tracker_id: obj for obj in items if obj.is_object}
        
        for pid in people:
            self._person_last_seen[pid] = frame_n
            
        for oid, obj in objects.items():
            history = self._class_history.setdefault(oid, ClassHistory())
            history.add(obj.class_id, obj.class_name, obj.confidence, frame_n)
            
        active_pairs: Set[Tuple[int, int]] = set()
        for pid, person in people.items():
            held = self._associations.setdefault(pid, set())
            wrists = person_wrists.get(pid)
            
            for oid, obj in objects.items():
                pair = (pid, oid)
                if self._is_holding(person=person, object=obj, wrists=wrists):
                    active_pairs.add(pair)
                    rec = self._holds.setdefault(pair, HoldRecord())
                    rec.frames += 1
                    if rec.frames >= MIN_HOLD_FRAMES:
                        held.add(oid)
        
        for pair in list(self._holds.keys()):
            if pair not in active_pairs:
                del self._holds[pair]
                
        w, h = FRAME_WIDTH, FRAME_HEIGHT
        for pid, held in list(self._associations.items()):
            person = people.get(pid)
            wrists = person_wrists.get(pid)
            for oid in list(held):
                obj = objects.get(oid)
                if obj is None:
                    held.discard(oid)
                    continue
                    
                touching = person is not None and self._is_holding(person=person, object=obj, wrists=wrists)
                if touching:
                    continue
                
                pair = (pid, oid)
                if pair not in self._drops:
                    self._drops[pair] = DropRecord(pid=pid, obj_id=oid, drop_frame=frame_n, near_edge=_near_edge(object=obj, width=w, height=h))
                held.discard(oid)
                
        result: Optional[Incident] = None
        ground = h * GROUND_LINE_RATIO
        settle_radius = w * STILLNESS_RADIUS_PERCENT
        
        for pair, drop in list(self._drops.items()):
            pid, oid = pair
            obj = objects.get(oid)
            
            if obj is None:
                del self._drops[pair]
                continue
            
            drop.recent_centers.append(obj.center)
            
            if drop.near_edge:
                del self._drops[pair]
                continue
            
            if len(drop.recent_centers) >= 2:
                total_dy = drop.recent_centers[-1][1] - drop.recent_centers[0][1]
                spanned = len(drop.recent_centers) - 1
                avg_dy = (total_dy / spanned) / max(obj.height, 1.0)
                descending = avg_dy > DESCENT_THRESHOLD
            else:
                descending = False
                
            if obj.bottom < ground or not descending:
                drop.settled = None
                drop.settle_anchor = None
                continue
            
            if drop.settle_anchor is None:
                drop.settle_anchor = obj.center
                drop.settled = frame_n
            else:
                moved = _distance(drop.settle_anchor, obj.center)
                if moved > settle_radius:
                    drop.settle_anchor = obj.center
                    drop.settled = frame_n
                    
            frames_settled = frame_n - drop.settled
            if frames_settled < SETTLED_CONFIRMATION_FRAMES:
                continue
            
            if pid in people:
                drop.last_seen = frame_n
                continue
            
            frames_since_last_seen = frame_n - drop.last_seen
            if frames_since_last_seen < PERSON_EXIT_FRAMES:
                continue
            
            if pair in self._triggered_pairs:
                del self._drops[pair]
                continue
            
            self._triggered_pairs.add(pair)
            del(self._drops[pair])
            
            history = self._class_history.get(oid)
            stable_name, stable_conf = (history.stable_classification() if history else (obj.class_name, obj.confidence))
            
            result = Incident(pid=pid, obj_id=oid, class_name=stable_name, confidence=stable_conf, frame_n=frame_n)
            break
        
        stale_people = [pid for pid in self._associations if pid not in people]
        for pid in stale_people:
            del self._associations[pid]
            
        for oid, hist in list(self._class_history.items()):
            if frame_n - hist.last_seen > STALE_HISTORY_FRAMES:
                del self._class_history[oid]
                
        return result