# This Script contains all the Hardcoded Configurations for the Project, such as the Model Path, the Camera Resolution, the Camera FPS, and other Configurations
# It is made so that the Configurations can be easily changed in one place, and the changes will be reflected throughout the project
# It also reduces time in development to change the configurations in multiple places, and also reduces the chances of errors due to inconsistent configurations

FRAME_WIDTH: int = 640 # The Height of The Frame that a 480p camera will capture
FRAME_HEIGHT: int = 480 # The Width of The Frame that a 480p camera will capture
PROP_BUFFERSIZE: int = 1 # Get Latest Frame only from the Camera
FPS: int = 30 # The FPS of the Camera (Frames Per Second)
TIME_WINDOW: float = 10.0 # The Time Window in seconds for which frames are kept in the buffer