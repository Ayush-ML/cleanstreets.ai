# This Script is responsible for Handling the Sliding window Buffer for the Camera Frames. It maintains a buffer of the last N frames captured by the camera, allowing for efficient processing and analysis of video data.
# It does not rely on a Specific Number of Frames, but rather a Time Window, which is more flexible and adaptive to varying frame rates and processing requirements.
# It uses a Deque Object for Efficiently Adding and Removing Frames from the Buffer, ensuring that the buffer always contains the most recent frames within the specified time window.
# It calculates a cutoff timestamp based on current time - time of the given frame, the removes frames starting from the last one, that are older than the cutoff timestamp, until it finds a frame that is within the time window. 
# Importing Necessary Libraries
from src.core.config import FPS, TIME_WINDOW
from typing import Deque, Tuple
from collections import deque
import time, cv2, numpy as np
from pathlib import Path

# The Buffer Class that handles everything
class RollingBuffer:
    """
    A Class that handles the Rolling Buffer for the Camera Frames
    Attributes:
        frames: The Deque Object that holds the frames and their timestamps
        time_window: The Time Window in seconds for which frames are kept in the buffer
    """
    
    def __init__(self, time_window: float = TIME_WINDOW) -> None:
        """
        Initializes the RollingBuffer Class
        Args:
            time_window: The Time Window in seconds for which frames are kept in the buffer (default is TIME_WINDOW)
        """
        self.frames: Deque[Tuple[float, np.ndarray]] = deque()
        self.time_window = time_window
        
    def __len__(self) -> int:
        """
        Returns the number of frames in the buffer
        Returns:
            int: The number of frames in the buffer
        """
        return len(self.frames)
    
    def add_frame(self, frame: np.ndarray, timestamp: float = None) -> None:
        """
        Adds a frame to the buffer and removes old frames outside the time window
        Args:
            frame: The Frame to be added to the buffer
            timestamp: The Timestamp of the frame (default is current time)
        """
        if timestamp is None:
            timestamp = time.monotonic()
        self.frames.append((timestamp, frame))
        cutoff = timestamp - self.time_window
        while self.frames and self.frames[0][0] < cutoff:
            self.frames.popleft()
            
    def save_clip(self, path: Path, fps: int = FPS) -> None:
        """
        Saves the frames in the buffer as a video clip to the specified path
        Args:
            path: The Path to save the video clip
            fps: The Frames Per Second for the video clip (default is FPS from core/config.py)
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.frames:
            print("No frames to save.")
            return None
        
        first_ts, first_frame = self.frames[0]
        last_ts, _ = self.frames[-1]
        height, width = first_frame.shape[:2]
        
        time_taken = last_ts - first_ts
        frame_count = len(self.frames)
        fps = frame_count / time_taken if time_taken > 0 else fps
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(filename=str(path), fourcc=fourcc, fps=fps, frameSize=(width, height))
        try:
            for _, frame in self.frames:
                writer.write(frame)
        except Exception as e:
            print(f"Error while writing frames to video: {e}")
        finally:
            writer.release()
            print(f"Video clip saved to {path} with {frame_count} frames at {fps:.2f} FPS.")
            
    def clear(self) -> None:
        """
        Clears the buffer by removing all frames
        """
        self.frames.clear()