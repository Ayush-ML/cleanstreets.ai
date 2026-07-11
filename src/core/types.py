# This Script holds shared type aliases used across the project.
# Kept separate from utils.py specifically to avoid a circular import:
# utils.py needs Object (from objects.py) for its helper functions, and
# objects.py needs Point for its own type hints — if Point lived in
# utils.py, those two files would import each other and neither could
# ever finish loading. This file has ZERO internal project dependencies,
# so both utils.py and objects.py can safely import from here.
from typing import Tuple

Point = Tuple[float, float]