# This is the Script handles the Event Detector 
# It combines the Pose Estimation model with a rule based system that checks whether the person has actually littered
# It uses IOU based Bounding Box Overlapping as a fallback if the Pose Estimation model fails
# And it uses specific rules against Edge Cases and also combines multiple frames, not just one frames and requires them to agree before making a decision
# Importing Necessary Libraries
from typing import Dict, List, Optional, Set, Tuple
from src.core.config import FRAME_HEIGHT, FRAME_WIDTH, MIN_HOLD_FRAMES, OVERLAP_THRESHOLD, WRIST_PROXIMITY_RATIO, DESCENT_THRESHOLD, GROUND_LINE_RATIO, SETTLED_CONFIRMATION_FRAMES, STILLNESS_RADIUS_PERCENT, STALE_HISTORY_FRAMES, PERSON_EXIT_FRAMES, OCCLUSION_GRACE_FRAMES, RELEASE_CONFIRM_FRAMES
from src.models.objects import Object
from src.core.utils import HoldState, Incident, DropRecord, ClassHistory, _iou, _distance, _near_edge
from src.core.types import Point

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
        self._holds: Dict[Tuple[int, int], HoldState] = {}
        self._drops: Dict[Tuple[int, int], DropRecord] = {}
        self._person_last_seen: Dict[int, int] = {}
        self._triggered_pairs: Set[Tuple[int, int]] = set()
        self._class_history: Dict[int, ClassHistory] = {}
    
    @staticmethod
    def _is_holding(person: Object, object: Object, wrists: Optional[List[Point]]) -> bool:
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
        person_wrists = person_wrists or {}

        people = {obj.tracker_id: obj for obj in items if obj.is_person}
        objects = {obj.tracker_id: obj for obj in items if obj.is_object}
        w, h = FRAME_WIDTH, FRAME_HEIGHT

        for pid in people:
            self._person_last_seen[pid] = frame_n

        for oid, obj in objects.items():
            history = self._class_history.setdefault(oid, ClassHistory())
            history.add(obj.class_id, obj.class_name, obj.confidence, frame_n)

        candidate_pairs: Set[Tuple[int, int]] = set(self._holds.keys())
        for pid in people:
            for oid in objects:
                candidate_pairs.add((pid, oid))

        for pid, oid in candidate_pairs:
            pair = (pid, oid)
            person = people.get(pid)
            obj = objects.get(oid)
            wrists = person_wrists.get(pid)
            state = self._holds.get(pair)

            if obj is None:
                if state is not None and state.held:
                    state.missed_count += 1
                    state.release_count = 0
                    if state.missed_count > OCCLUSION_GRACE_FRAMES:
                        del self._holds[pair]
                continue

            touching = person is not None and self._is_holding(person=person, object=obj, wrists=wrists)

            if state is None:
                if not touching:
                    continue
                state = HoldState()
                self._holds[pair] = state

            if touching:
                state.missed_count = 0
                state.release_count = 0
                if not state.held:
                    state.confirm_count += 1
                    if state.confirm_count >= MIN_HOLD_FRAMES:
                        state.held = True
                continue

            state.missed_count = 0
            if state.held:
                state.release_count += 1
                if state.release_count >= RELEASE_CONFIRM_FRAMES:
                    if pair not in self._drops:
                        self._drops[pair] = DropRecord(
                            pid=pid, obj_id=oid, drop_frame=frame_n,
                            near_edge=_near_edge(object=obj, width=w, height=h),
                        )
                    del self._holds[pair]
            else:
                del self._holds[pair]

        result: Optional[Incident] = None
        ground = h * GROUND_LINE_RATIO
        settle_radius = w * STILLNESS_RADIUS_PERCENT

        for pair, drop in list(self._drops.items()):
            pid, oid = pair
            obj = objects.get(oid)

            if obj is None:
                if drop.missing_since is None:
                    drop.missing_since = frame_n
                if frame_n - drop.missing_since > OCCLUSION_GRACE_FRAMES:
                    del self._drops[pair]
                continue
            drop.missing_since = None

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

            if obj.bottom < ground:
                drop.settled = None
                drop.settle_anchor = None
                continue

            if drop.settle_anchor is None:
                if not descending:
                    continue
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
            del self._drops[pair]

            history = self._class_history.get(oid)
            stable_name, stable_conf = (history.stable_classification() if history else (obj.class_name, obj.confidence))

            result = Incident(pid=pid, obj_id=oid, class_name=stable_name, confidence=stable_conf, frame_n=frame_n)
            break

        for oid, hist in list(self._class_history.items()):
            if frame_n - hist.last_seen > STALE_HISTORY_FRAMES:
                del self._class_history[oid]

        return result