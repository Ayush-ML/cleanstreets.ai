# This Script contains all the Hardcoded Configurations for the Project, such as the Model Path, the Camera Resolution, the Camera FPS, and other Configurations
# It is made so that the Configurations can be easily changed in one place, and the changes will be reflected throughout the project
# It also reduces time in development to change the configurations in multiple places, and also reduces the chances of errors due to inconsistent configurations

FRAME_WIDTH: int = 1280 # The Height of The Frame that a 480p camera will capture
FRAME_HEIGHT: int = 780 # The Width of The Frame that a 480p camera will capture
PROP_BUFFERSIZE: int = 1 # Get Latest Frame only from the Camera
FPS: int = 30 # The FPS of the Camera (Frames Per Second)
TIME_WINDOW: float = 10.0 # The Time Window in seconds for which frames are kept in the buffer
MODEL_PATH = "models/yolov8s.pt" # The Name of the Model to be used for Object Detection
IOU = 0.5 # The IOU Threshold for Object Detection
CONFIDENCE = 0.6 # The Confidence Threshold for Object Detection
CLASSES =[
    0,  # person
    39,  # bottle
    40,  # wine glass
    41,  # cup
    42,  # fork
    43,  # knife
    44,  # spoon
    45,  # bowl
    46,  # banana
    47,  # apple
    48,  # sandwich
    49,  # orange
    50,  # broccoli
    51,  # carrot
    52,  # hot dog
    53,  # pizza
    54,  # donut
    55,  # cake
] 
NMS = True # Whether to use Non-Maximum Suppression (NMS) or not
SIMPLIFY = True # Whether to simplify the model or not
OPTIMIZE = True # Whether to optimize the model or not
QUANTIZE = 8 # The Quantization Level for the Model (INT8 Quantization)
FORMAT = "onnx" # The Format in which the Model is to be exported
VERBOSE = False # Whether to print the logs or not
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" # The URL WHere the model is fetched from
HAND_MODEL_PATH = "models/hand_landmarker.task" # Place where model is stored
MAX_NUM_HANDS = 4                  # Up to 2 hands per 2 people in a typical frame
HAND_DETECTION_CONFIDENCE = 0.5 # Defult Confidence
HAND_PRESENCE_CONFIDENCE = 0.5 # Defult Confidence
HAND_TRACKING_CONFIDENCE = 0.5 # Defult Confidence
PERSON_MATCH_MARGIN_RATIO = 0.12   # Expand person bbox by this much when testing hand containment
WRIST_LANDMARK_IDX = 0  # MediaPipe's 21-point hand model, index 0 is the wrist
#
MIN_HOLD_FRAMES = 8 # Number of frames a person has to be holding an object before it is actually considored holding
OVERLAP_THRESHOLD = 0.15 # Minimum IOU Score for Bounding box based Overlapping/Holding
WRIST_PROXIMITY_RATIO = 0.6 #  Distance of the Wrist to the center of the Object Threshold
EDGE_MARGIN_PERCENT = 0.08 # Exclude Drops that happen at this percent of the frames edge
DOWNWARD_WIDNOW_FRAMES = 5 # No of frames in which Downward Motion is averaged
DESCENT_THRESHOLD = 0.15 # Average of dy in a frame as a ratio of the objects height
GROUND_LINE_RATIO = 0.80 # 80 percent of the frame counts as not ground, rest as ground
STILLNESS_RADIUS_PERCENT = 0.03 # Percent Object should stay within of the frames width to count as stopped
SETTLED_CONFIRMATION_FRAMES = 15 # Number of Frames the position of the Object must stop moving in order to register it as settled
PERSON_EXIT_FRAMES = 30 # No of Frames person must be absent before triggering littering
CLASS_VOTE_FRAMES = 15 # No of Frames of History used for voting
STALE_HISTORY_FRAMES = 90 # Prune the Class history for objects that are unseen this many frames
INCIDENT_DIR = "data/incidents" # Directory where All Incident Video's are Stored
INCIDENTS_FILE = "data/incidents.json" # JSON File with Data about all Incidents
DEAFULT_CAMERA = "cam1" # Camera to be used
VALID_STATUS = ("approved", "rejected") # All Status for Incidents, pending means that it is yet to be reviewed