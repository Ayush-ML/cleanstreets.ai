# This Script contains all the Hardcoded Configurations for the Project, such as the Model Path, the Camera Resolution, the Camera FPS, and other Configurations
# It is made so that the Configurations can be easily changed in one place, and the changes will be reflected throughout the project
# It also reduces time in development to change the configurations in multiple places, and also reduces the chances of errors due to inconsistent configurations

FRAME_WIDTH: int = 1280 # The Height of The Frame that a 480p camera will capture
FRAME_HEIGHT: int = 780 # The Width of The Frame that a 480p camera will capture
PROP_BUFFERSIZE: int = 1 # Get Latest Frame only from the Camera
FPS: int = 30 # The FPS of the Camera (Frames Per Second)
TIME_WINDOW: float = 10.0 # The Time Window in seconds for which frames are kept in the buffer
MODEL_NAME = "yolov8s.pt" # The Name of the Model to be used for Object Detection
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