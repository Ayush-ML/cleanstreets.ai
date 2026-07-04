# This Script is Responsible for Creating a Schema for a class to Store an Object and Their metadata
# This Metadata is extracted using the YOLOv8 Nano Model, which is a pre trained model for Object Detection and Classification.
# It mainly classifies the Person and an Object the person is holding in thier hands
# Importing Necessary Libraries
from typing import Tuple
from dataclasses import dataclass

# Schema Class
@dataclass
class  Object:
    """A Class that represents an Object and its metadata"""
    tracker_id: int
    class_name: str
    class_id: int
    x1: float
    x2: float
    y1: float
    y2: float
    confidence: float
    
    @property
    def center(self) -> Tuple[float, float]:
        """Returns the center coordinates of the object"""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def width(self) -> float:
        """Returns the width of the object"""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Returns the height of the object"""
        return self.y2 - self.y1

    @property
    def is_person(self) -> bool:
        """Returns True if the object is a person, False otherwise"""
        return self.class_id == 0
    
    @property
    def is_object(self) -> bool:
        """Returns True if the object is an object, False otherwise"""
        return self.class_id != 0
    
    @property
    def bottom(self) -> float:
        """Returns the bottom coordinate of the object"""
        return self.y2
