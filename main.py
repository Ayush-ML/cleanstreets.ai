# This is the main script that is responsible for acrually running the system
# It uses the Ultralytics Plot Function to draw all the Bounding Boxes

import cv2
from src.camera.capture import open_camera
from src.models.detector import Detector
from src.models.pose_est import PoseEstimator
from src.models.events import EventChecker
from plyer import notification
from datetime import datetime

WRIST_COLOR = (0, 0, 255)       
PERSON_COLOR = (0, 255, 0)      
OBJECT_COLOR = (255, 128, 0)    
HOLD_COLOR = (0, 255, 255)      
INCIDENT_COLOR = (0, 0, 255)    


def draw_object(frame, obj, held: bool) -> None:
    """
    Draws one tracked object's bounding box, track ID, class name, and
    confidence. Boxes for objects the REAL EventChecker currently
    considers 'held' are highlighted yellow instead of the default color.
    Args:
        frame: The Frame that we want to draw the bounding boxes on
        obj: The Data about the object that we want to draw a bounding box for
        held: A Flag of whether that object is currently held or not
    """
    if obj.is_person:
        color = PERSON_COLOR
    else:
        color = HOLD_COLOR if held else OBJECT_COLOR

    p1 = (int(obj.x1), int(obj.y1))
    p2 = (int(obj.x2), int(obj.y2))
    cv2.rectangle(frame, p1, p2, color, 2)

    label = f"#{obj.tracker_id} {obj.class_name} {obj.confidence:.2f}"
    if held:
        label += " [HELD]"
    cv2.putText(frame, label, (p1[0], max(p1[1] - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_wrists(frame, wrists_by_track) -> None:
    """
    Draws every detected wrist point — confirms whether MediaPipe is
    actually returning wrist data at all.
    Args:
        frame: The Frame that we want to draw the wrists for
        wrists_by_track: All wrists in the frame sorted by Tracker ID of the person
    """
    for tracker_id, points in wrists_by_track.items():
        for (x, y) in points:
            cv2.circle(frame, (int(x), int(y)), 8, WRIST_COLOR, -1)
            cv2.putText(frame, f"wrist:{tracker_id}", (int(x) + 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WRIST_COLOR, 1)

def main() -> None:
    """
    The Piece that wires everything Together, Notifies if Incident has been triggered
    """
    detector = Detector()
    pose_estimator = PoseEstimator()
    checker = EventChecker()

    frame_source = open_camera()
    person_history = {}

    print("Running.... Press 'q' to quit.")

    try:
        for frame_number, frame in frame_source:
            frame_height, frame_width = frame.shape[0], frame.shape[1]
            timestamp_ms = int(frame_number / 30 * 1000)

            objects = detector.detect(frame)
            persons = {o.tracker_id: o for o in objects if o.is_person}

            pose_estimator.submit(frame, timestamp_ms)
            person_history[timestamp_ms] = persons
            if len(person_history) > 90:
                del person_history[min(person_history)]

            wrists = pose_estimator.get_latest_wrists(person_history, frame_width, frame_height)

            incident = checker.check(frame_number, objects, wrists)

            currently_held_object_ids = {oid for (pid, oid), state in checker._holds.items() if state.held}
            in_progress_drops = len(checker._drops)

            for obj in objects:
                held = (not obj.is_person) and (obj.tracker_id in currently_held_object_ids)
                draw_object(frame, obj, held)
            draw_wrists(frame, wrists)

            status = (f"objects: {len(objects)}  wrists: {len(wrists)}  "
                      f"held: {len(currently_held_object_ids)}  "
                      f"drops in progress: {in_progress_drops}")
            cv2.putText(frame, status, (10, frame_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if incident is not None:
                cv2.putText(frame, "INCIDENT TRIGGERED", (10, 40),cv2.FONT_HERSHEY_SIMPLEX, 1.0, INCIDENT_COLOR, 3)
                print(f"[INCIDENT] person={incident.pid} object={incident.obj_id} "f"class={incident.class_name} confidence={incident.confidence:.2f}")
                notification.notify(
                    title="Incident Triggered!",
                    message=f"An Incident has been triggered, Confidence: {incident.confidence:.2f}, Timestamp: {datetime.now()}",
                    app_name="CleanStreetsAI",
                    timeout=10 
                )

            cv2.imshow("CleanStreets AI — Diagnostic Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        frame_source.close()
        pose_estimator.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()