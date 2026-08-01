"""
CleanStreets AI — Realistic Synthetic Littering Test

This test runs the REAL EventChecker without requiring:

    - A camera
    - YOLO
    - ByteTrack
    - MediaPipe

The synthetic inputs imitate common imperfections from the real
pipeline:

    - Bounding-box jitter
    - Confidence variation
    - Wrist-coordinate jitter
    - Brief missing wrist detections
    - Slight bottle-size variation
    - Accelerating bottle motion
    - A small bounce after impact
    - Brief bottle occlusion
    - Natural person movement

IMPORTANT:

This script is designed for the current EventChecker structure.
The bottle remains near the wrist long enough to establish a hold,
then separates, continues descending after release confirmation,
settles, and the person exits.

Controls:

    Q = Quit
    R = Restart
    SPACE = Pause / Resume
"""


import math
import random

import cv2
import numpy as np

from src.models.events import EventChecker
from src.models.objects import Object

from src.core.config import (
    FRAME_WIDTH,
    FRAME_HEIGHT,
    FPS,
    MIN_HOLD_FRAMES,
    RELEASE_CONFIRM_FRAMES,
    SETTLED_CONFIRMATION_FRAMES,
    PERSON_EXIT_FRAMES,
)


# ============================================================
# RANDOMNESS
# ============================================================

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ============================================================
# TRACKING IDs
# ============================================================

PERSON_ID = 1
BOTTLE_ID = 2

PERSON_CLASS_ID = 0
BOTTLE_CLASS_ID = 39


# ============================================================
# SCENARIO TIMING
# ============================================================

# ------------------------------------------------------------
# PHASE 1
#
# Person enters while holding the bottle.
# ------------------------------------------------------------

ENTRY_START = 0
ENTRY_END = 35


# ------------------------------------------------------------
# PHASE 2
#
# Person slows down and lowers their arm.
# The bottle is still held.
# ------------------------------------------------------------

PRE_RELEASE_START = ENTRY_END
PRE_RELEASE_END = 50


# ------------------------------------------------------------
# PHASE 3
#
# The bottle is separated from the wrist.
#
# It remains approximately stationary long enough for the
# current EventChecker to confirm the release.
#
# This is intentionally included because the current event
# logic begins its descent history after release confirmation.
# ------------------------------------------------------------

RELEASE_CONFIRM_START = PRE_RELEASE_END

RELEASE_CONFIRM_END = (
    RELEASE_CONFIRM_START
    + RELEASE_CONFIRM_FRAMES
)


# ------------------------------------------------------------
# PHASE 4
#
# Bottle falls with accelerating downward movement.
# ------------------------------------------------------------

FALL_START = RELEASE_CONFIRM_END

FALL_DURATION = 24

FALL_END = (
    FALL_START
    + FALL_DURATION
)


# ------------------------------------------------------------
# PHASE 5
#
# Bottle bounces slightly after reaching the ground.
# ------------------------------------------------------------

BOUNCE_START = FALL_END

BOUNCE_DURATION = 10

BOUNCE_END = (
    BOUNCE_START
    + BOUNCE_DURATION
)


# ------------------------------------------------------------
# PHASE 6
#
# Bottle remains on the ground.
# Person walks away.
# ------------------------------------------------------------

WALK_AWAY_START = BOUNCE_END

PERSON_EXIT_START = (
    WALK_AWAY_START
    + 30
)


# ------------------------------------------------------------
# PHASE 7
#
# Person is absent from detections.
#
# The bottle remains visible.
# ------------------------------------------------------------

TOTAL_FRAMES = (
    PERSON_EXIT_START
    + PERSON_EXIT_FRAMES
    + 70
)


# ============================================================
# SYNTHETIC DETECTOR SETTINGS
# ============================================================

# Simulated YOLO bounding-box noise.

BBOX_JITTER = 2.0

# Simulated MediaPipe wrist noise.

WRIST_JITTER = 4.0

# Confidence ranges.

PERSON_CONFIDENCE_MIN = 0.83
PERSON_CONFIDENCE_MAX = 0.99

BOTTLE_CONFIDENCE_MIN = 0.70
BOTTLE_CONFIDENCE_MAX = 0.98

# Small variation in detected bottle dimensions.

BOTTLE_SIZE_JITTER = 1.5

# Wrist detections may be missing occasionally.

WRIST_MISSING_PROBABILITY = 0.06

# Brief bottle occlusion during the fall.

OCCLUSION_FRAME_START = (
    FALL_START
    + 8
)

OCCLUSION_FRAME_END = (
    OCCLUSION_FRAME_START
    + 2
)


# ============================================================
# COLOURS
# ============================================================

BACKGROUND_COLOR = (
    35,
    35,
    35,
)

WALL_COLOR = (
    45,
    45,
    45,
)

FLOOR_COLOR = (
    65,
    65,
    65,
)

GRID_COLOR = (
    90,
    90,
    90,
)

PERSON_COLOR = (
    0,
    255,
    0,
)

BOTTLE_COLOR = (
    255,
    150,
    0,
)

HELD_COLOR = (
    0,
    255,
    255,
)

WRIST_COLOR = (
    0,
    0,
    255,
)

TEXT_COLOR = (
    255,
    255,
    255,
)

PANEL_COLOR = (
    20,
    20,
    20,
)

GOOD_COLOR = (
    0,
    220,
    0,
)

WARNING_COLOR = (
    0,
    200,
    255,
)

INCIDENT_COLOR = (
    0,
    0,
    255,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clamp(
    value,
    minimum,
    maximum,
):
    """
    Restricts a value to a range.
    """

    return max(
        minimum,
        min(
            value,
            maximum,
        ),
    )


def noisy(
    value,
    amount,
):
    """
    Adds deterministic random noise.
    """

    return (
        value
        + random.uniform(
            -amount,
            amount,
        )
    )


def smooth_step(
    value,
):
    """
    Smooth interpolation from 0 to 1.
    """

    value = clamp(
        value,
        0.0,
        1.0,
    )

    return (
        value
        * value
        * (
            3.0
            - 2.0 * value
        )
    )


# ============================================================
# REAL OBJECT CREATION
# ============================================================

def create_object(
    tracker_id,
    class_id,
    class_name,
    confidence,
    x1,
    y1,
    x2,
    y2,
):
    """
    Creates the project's real Object instance.

    This provides the properties used by EventChecker:

        tracker_id
        class_id
        class_name
        confidence
        x1
        y1
        x2
        y2
        center
        width
        height
        is_person
        is_object
    """

    return Object(
        tracker_id=tracker_id,
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        x1=float(x1),
        y1=float(y1),
        x2=float(x2),
        y2=float(y2),
    )


# ============================================================
# PERSON MOTION
# ============================================================

def get_person_position(
    frame_number,
):
    """
    Produces natural person movement.

    The person:

        1. Enters from the left
        2. Slows down
        3. Releases the bottle
        4. Walks toward the right
        5. Exits the frame
    """

    person_width = 180

    person_height = 420

    base_y = 165

    # --------------------------------------------------------
    # ENTERING
    # --------------------------------------------------------

    if frame_number < ENTRY_END:

        progress = (
            frame_number
            / ENTRY_END
        )

        progress = smooth_step(
            progress
        )

        x = (
            -120
            + progress * 470
        )

    # --------------------------------------------------------
    # PREPARING TO RELEASE
    # --------------------------------------------------------

    elif (
        frame_number
        < PRE_RELEASE_END
    ):

        progress = (
            (
                frame_number
                - ENTRY_END
            )
            /
            (
                PRE_RELEASE_END
                - ENTRY_END
            )
        )

        x = (
            350
            + progress * 20
        )

    # --------------------------------------------------------
    # RELEASE CONFIRMATION
    # --------------------------------------------------------

    elif (
        frame_number
        < WALK_AWAY_START
    ):

        x = 370

    # --------------------------------------------------------
    # WALKING AWAY
    # --------------------------------------------------------

    elif (
        frame_number
        < PERSON_EXIT_START
    ):

        progress = (
            (
                frame_number
                - WALK_AWAY_START
            )
            /
            (
                PERSON_EXIT_START
                - WALK_AWAY_START
            )
        )

        progress = smooth_step(
            progress
        )

        x = (
            370
            + progress * 900
        )

    # --------------------------------------------------------
    # PERSON HAS EXITED
    # --------------------------------------------------------

    else:

        return None

    # Small walking motion.

    walking_bob = (
        math.sin(
            frame_number
            * 0.35
        )
        * 3
    )

    y = (
        base_y
        + walking_bob
    )

    return (
        x,
        y,
        person_width,
        person_height,
    )


# ============================================================
# WRIST MOTION
# ============================================================

def get_wrist_positions(
    frame_number,
    person_data,
):
    """
    Produces synthetic MediaPipe wrist coordinates.

    The wrist moves naturally with the person and lowers before
    the bottle is released.
    """

    if person_data is None:

        return {}

    person_x = person_data[0]

    person_y = person_data[1]

    # --------------------------------------------------------
    # NORMAL ARM MOVEMENT
    # --------------------------------------------------------

    arm_swing = (
        math.sin(
            frame_number
            * 0.28
        )
        * 7
    )

    left_wrist = (
        person_x
        + 45
        + arm_swing,

        person_y
        + 225,
    )

    right_wrist_x = (
        person_x
        + 145
        + arm_swing
    )

    right_wrist_y = (
        person_y
        + 245
    )

    # --------------------------------------------------------
    # LOWER ARM BEFORE RELEASE
    # --------------------------------------------------------

    if (
        ENTRY_END
        <= frame_number
        < PRE_RELEASE_END
    ):

        progress = (
            (
                frame_number
                - ENTRY_END
            )
            /
            (
                PRE_RELEASE_END
                - ENTRY_END
            )
        )

        right_wrist_y += (
            progress * 70
        )

    # --------------------------------------------------------
    # MOVE ARM AWAY AFTER RELEASE
    # --------------------------------------------------------

    elif (
        frame_number
        >= PRE_RELEASE_END
    ):

        right_wrist_x += 90

        right_wrist_y -= 80

    # --------------------------------------------------------
    # ADD MEDIAPIPE-LIKE NOISE
    # --------------------------------------------------------

    left_wrist = (
        noisy(
            left_wrist[0],
            WRIST_JITTER,
        ),

        noisy(
            left_wrist[1],
            WRIST_JITTER,
        ),
    )

    right_wrist = (
        noisy(
            right_wrist_x,
            WRIST_JITTER,
        ),

        noisy(
            right_wrist_y,
            WRIST_JITTER,
        ),
    )

    # --------------------------------------------------------
    # OCCASIONAL MISSING WRIST
    # --------------------------------------------------------

    if (
        random.random()
        < WRIST_MISSING_PROBABILITY
    ):

        return {
            PERSON_ID: [
                left_wrist,
            ]
        }

    return {
        PERSON_ID: [
            left_wrist,
            right_wrist,
        ]
    }


# ============================================================
# BOTTLE MOTION
# ============================================================

def get_bottle_position(
    frame_number,
    person_data,
):
    """
    Produces realistic bottle movement.

    The bottle:

        1. Moves with the hand
        2. Is released
        3. Falls with acceleration
        4. Bounces slightly
        5. Settles on the ground
    """

    bottle_width = 38

    bottle_height = 82

    ground_y = (
        FRAME_HEIGHT
        - bottle_height
        - 18
    )

    # --------------------------------------------------------
    # HELD
    # --------------------------------------------------------

    if (
        frame_number
        < PRE_RELEASE_END
    ):

        if person_data is None:

            person_x = 370

            person_y = 165

        else:

            person_x = person_data[0]

            person_y = person_data[1]

        # The bottle follows the right hand.

        arm_lowering = 0

        if (
            frame_number
            >= ENTRY_END
        ):

            progress = (
                (
                    frame_number
                    - ENTRY_END
                )
                /
                (
                    PRE_RELEASE_END
                    - ENTRY_END
                )
            )

            arm_lowering = (
                progress * 70
            )

        x = (
            person_x
            + 132
        )

        y = (
            person_y
            + 220
            + arm_lowering
        )

        phase = (
            "HELD"
        )

    # --------------------------------------------------------
    # RELEASE CONFIRMATION
    # --------------------------------------------------------

    elif (
        frame_number
        < RELEASE_CONFIRM_END
    ):

        # The bottle is separated from the wrist.
        #
        # It moves only slightly during this phase so the
        # current EventChecker can complete release
        # confirmation before evaluating the fall.

        progress = (
            (
                frame_number
                - RELEASE_CONFIRM_START
            )
            /
            RELEASE_CONFIRM_FRAMES
        )

        x = (
            500
            + progress * 5
        )

        y = (
            455
            + math.sin(
                progress
                * math.pi
            )
            * 2
        )

        phase = (
            "RELEASE CONFIRMATION"
        )

    # --------------------------------------------------------
    # ACCELERATING FALL
    # --------------------------------------------------------

    elif (
        frame_number
        < FALL_END
    ):

        progress = (
            (
                frame_number
                - FALL_START
            )
            /
            FALL_DURATION
        )

        progress = clamp(
            progress,
            0.0,
            1.0,
        )

        # Quadratic motion imitates acceleration.

        fall_progress = (
            progress
            * progress
        )

        start_y = 455

        x = (
            505
            + progress * 14
        )

        y = (
            start_y
            + (
                ground_y
                - start_y
            )
            * fall_progress
        )

        phase = (
            "FALLING"
        )

    # --------------------------------------------------------
    # SMALL BOUNCE
    # --------------------------------------------------------

    elif (
        frame_number
        < BOUNCE_END
    ):

        progress = (
            (
                frame_number
                - BOUNCE_START
            )
            /
            BOUNCE_DURATION
        )

        bounce = (
            math.sin(
                progress
                * math.pi
            )
            * 18
        )

        x = (
            519
            + progress * 2
        )

        y = (
            ground_y
            - bounce
        )

        phase = (
            "BOUNCING"
        )

    # --------------------------------------------------------
    # SETTLED
    # --------------------------------------------------------

    else:

        x = 521

        y = ground_y

        phase = (
            "SETTLED"
        )

    # --------------------------------------------------------
    # DETECTOR JITTER
    # --------------------------------------------------------

    x = noisy(
        x,
        BBOX_JITTER,
    )

    y = noisy(
        y,
        BBOX_JITTER,
    )

    width = (
        bottle_width
        + noisy(
            0,
            BOTTLE_SIZE_JITTER,
        )
    )

    height = (
        bottle_height
        + noisy(
            0,
            BOTTLE_SIZE_JITTER,
        )
    )

    return (
        x,
        y,
        width,
        height,
        phase,
    )


# ============================================================
# CREATE ONE SYNTHETIC FRAME
# ============================================================

def get_scenario(
    frame_number,
):
    """
    Creates the real Object detections and wrist coordinates
    for one frame.
    """

    items = []

    person_data = (
        get_person_position(
            frame_number
        )
    )

    wrists = (
        get_wrist_positions(
            frame_number,
            person_data,
        )
    )

    # --------------------------------------------------------
    # PERSON DETECTION
    # --------------------------------------------------------

    if person_data is not None:

        person_x = (
            person_data[0]
        )

        person_y = (
            person_data[1]
        )

        person_width = (
            person_data[2]
        )

        person_height = (
            person_data[3]
        )

        person = create_object(
            tracker_id=PERSON_ID,

            class_id=PERSON_CLASS_ID,

            class_name="person",

            confidence=random.uniform(
                PERSON_CONFIDENCE_MIN,
                PERSON_CONFIDENCE_MAX,
            ),

            x1=noisy(
                person_x,
                BBOX_JITTER,
            ),

            y1=noisy(
                person_y,
                BBOX_JITTER,
            ),

            x2=noisy(
                (
                    person_x
                    + person_width
                ),
                BBOX_JITTER,
            ),

            y2=noisy(
                (
                    person_y
                    + person_height
                ),
                BBOX_JITTER,
            ),
        )

        items.append(
            person
        )

    # --------------------------------------------------------
    # BOTTLE DETECTION
    # --------------------------------------------------------

    (
        bottle_x,
        bottle_y,
        bottle_width,
        bottle_height,
        phase,
    ) = get_bottle_position(
        frame_number,
        person_data,
    )

    # --------------------------------------------------------
    # BRIEF BOTTLE OCCLUSION
    # --------------------------------------------------------

    bottle_visible = not (
        OCCLUSION_FRAME_START
        <= frame_number
        < OCCLUSION_FRAME_END
    )

    if bottle_visible:

        bottle = create_object(
            tracker_id=BOTTLE_ID,

            class_id=BOTTLE_CLASS_ID,

            class_name="bottle",

            confidence=random.uniform(
                BOTTLE_CONFIDENCE_MIN,
                BOTTLE_CONFIDENCE_MAX,
            ),

            x1=bottle_x,

            y1=bottle_y,

            x2=(
                bottle_x
                + bottle_width
            ),

            y2=(
                bottle_y
                + bottle_height
            ),
        )

        items.append(
            bottle
        )

    return (
        items,
        wrists,
        phase,
        bottle_visible,
    )


# ============================================================
# DRAW BACKGROUND
# ============================================================

def draw_background(
    frame,
):
    """
    Draws a controlled indoor environment.
    """

    frame[:] = (
        BACKGROUND_COLOR
    )

    horizon_y = int(
        FRAME_HEIGHT
        * 0.72
    )

    # Wall.

    cv2.rectangle(
        frame,

        (0, 0),

        (
            FRAME_WIDTH,
            horizon_y,
        ),

        WALL_COLOR,

        -1,
    )

    # Floor.

    cv2.rectangle(
        frame,

        (
            0,
            horizon_y,
        ),

        (
            FRAME_WIDTH,
            FRAME_HEIGHT,
        ),

        FLOOR_COLOR,

        -1,
    )

    # Perspective floor lines.

    vanishing_x = (
        FRAME_WIDTH
        // 2
    )

    for x in range(
        0,
        FRAME_WIDTH + 1,
        130,
    ):

        cv2.line(
            frame,

            (
                vanishing_x,
                horizon_y,
            ),

            (
                x,
                FRAME_HEIGHT,
            ),

            GRID_COLOR,

            1,
        )

    # Horizontal floor lines.

    for y in range(
        horizon_y,
        FRAME_HEIGHT,
        38,
    ):

        cv2.line(
            frame,

            (
                0,
                y,
            ),

            (
                FRAME_WIDTH,
                y,
            ),

            GRID_COLOR,

            1,
        )

    cv2.putText(
        frame,

        (
            "CLEANSTREETS AI"
        ),

        (
            20,
            38,
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        1.0,

        TEXT_COLOR,

        2,
    )

    cv2.putText(
        frame,

        (
            "REALISTIC SYNTHETIC "
            "LITTERING TEST"
        ),

        (
            20,
            70,
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.58,

        WARNING_COLOR,

        2,
    )


# ============================================================
# DRAW DETECTION
# ============================================================

def draw_object(
    frame,
    obj,
    held,
):
    """
    Draws a detection bounding box.
    """

    if obj.is_person:

        color = (
            PERSON_COLOR
        )

    elif held:

        color = (
            HELD_COLOR
        )

    else:

        color = (
            BOTTLE_COLOR
        )

    x1 = int(
        obj.x1
    )

    y1 = int(
        obj.y1
    )

    x2 = int(
        obj.x2
    )

    y2 = int(
        obj.y2
    )

    cv2.rectangle(
        frame,

        (
            x1,
            y1,
        ),

        (
            x2,
            y2,
        ),

        color,

        3,
    )

    label = (
        f"ID {obj.tracker_id} | "
        f"{obj.class_name} | "
        f"{obj.confidence:.2f}"
    )

    if held:

        label += (
            " | HELD"
        )

    cv2.putText(
        frame,

        label,

        (
            x1,
            max(
                20,
                y1 - 10,
            ),
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        color,

        2,
    )


# ============================================================
# DRAW WRISTS
# ============================================================

def draw_wrists(
    frame,
    wrists,
):
    """
    Draws synthetic MediaPipe wrist positions.
    """

    for points in (
        wrists.values()
    ):

        for index, point in enumerate(
            points
        ):

            x = int(
                point[0]
            )

            y = int(
                point[1]
            )

            cv2.circle(
                frame,

                (
                    x,
                    y,
                ),

                7,

                WRIST_COLOR,

                -1,
            )

            cv2.putText(
                frame,

                (
                    f"W{index + 1}"
                ),

                (
                    x + 9,
                    y - 5,
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.42,

                WRIST_COLOR,

                1,
            )


# ============================================================
# DIAGNOSTIC PANEL
# ============================================================

def draw_diagnostics(
    frame,
    checker,
    frame_number,
    phase,
    bottle_visible,
    triggered,
):
    """
    Displays the internal EventChecker state.
    """

    panel_x1 = (
        FRAME_WIDTH
        - 355
    )

    panel_y1 = 15

    panel_x2 = (
        FRAME_WIDTH
        - 15
    )

    panel_y2 = 285

    overlay = (
        frame.copy()
    )

    cv2.rectangle(
        overlay,

        (
            panel_x1,
            panel_y1,
        ),

        (
            panel_x2,
            panel_y2,
        ),

        PANEL_COLOR,

        -1,
    )

    cv2.addWeighted(
        overlay,

        0.85,

        frame,

        0.15,

        0,

        frame,
    )

    y = (
        panel_y1
        + 28
    )

    line_gap = 27

    def write_line(
        text,
        color=TEXT_COLOR,
    ):

        nonlocal y

        cv2.putText(
            frame,

            text,

            (
                panel_x1
                + 15,
                y,
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.52,

            color,

            1,
        )

        y += (
            line_gap
        )

    held_pairs = []

    for (
        pair,
        state,
    ) in (
        checker._holds.items()
    ):

        if state.held:

            held_pairs.append(
                pair
            )

    write_line(
        "EVENTCHECKER DIAGNOSTICS",
        WARNING_COLOR,
    )

    write_line(
        (
            f"Frame: "
            f"{frame_number}"
        )
    )

    write_line(
        (
            f"Phase: "
            f"{phase}"
        )
    )

    write_line(
        (
            f"Bottle visible: "
            f"{bottle_visible}"
        ),
        (
            GOOD_COLOR
            if bottle_visible
            else WARNING_COLOR
        ),
    )

    write_line(
        (
            f"Confirmed holds: "
            f"{len(held_pairs)}"
        )
    )

    write_line(
        (
            f"Active drops: "
            f"{len(checker._drops)}"
        )
    )

    write_line(
        (
            f"Hold records: "
            f"{len(checker._holds)}"
        )
    )

    write_line(
        (
            "RESULT: "
            + (
                "INCIDENT"
                if triggered
                else "WAITING"
            )
        ),

        (
            INCIDENT_COLOR
            if triggered
            else TEXT_COLOR
        ),
    )


# ============================================================
# MAIN SIMULATION
# ============================================================

def run_simulation():
    """
    Runs the realistic synthetic test.
    """

    random.seed(
        RANDOM_SEED
    )

    checker = (
        EventChecker()
    )

    frame_number = 0

    triggered = False

    paused = False

    trigger_frame = None

    print()
    print(
        "=" * 72
    )

    print(
        "CLEANSTREETS AI"
    )

    print(
        "REALISTIC SYNTHETIC "
        "LITTERING TEST"
    )

    print(
        "=" * 72
    )

    print(
        f"Frame size: "
        f"{FRAME_WIDTH} x "
        f"{FRAME_HEIGHT}"
    )

    print(
        f"FPS: {FPS}"
    )

    print(
        f"Random seed: "
        f"{RANDOM_SEED}"
    )

    print()

    print(
        "Configuration:"
    )

    print(
        f"  MIN_HOLD_FRAMES = "
        f"{MIN_HOLD_FRAMES}"
    )

    print(
        f"  RELEASE_CONFIRM_FRAMES = "
        f"{RELEASE_CONFIRM_FRAMES}"
    )

    print(
        f"  SETTLED_CONFIRMATION_FRAMES = "
        f"{SETTLED_CONFIRMATION_FRAMES}"
    )

    print(
        f"  PERSON_EXIT_FRAMES = "
        f"{PERSON_EXIT_FRAMES}"
    )

    print()

    print(
        "Controls:"
    )

    print(
        "  Q     Quit"
    )

    print(
        "  R     Restart"
    )

    print(
        "  SPACE Pause / Resume"
    )

    print(
        "=" * 72
    )

    while True:

        frame = np.zeros(
            (
                FRAME_HEIGHT,
                FRAME_WIDTH,
                3,
            ),

            dtype=np.uint8,
        )

        draw_background(
            frame
        )

        (
            items,
            wrists,
            phase,
            bottle_visible,
        ) = get_scenario(
            frame_number
        )

        # ----------------------------------------------------
        # RUN THE REAL EVENT DETECTOR
        # ----------------------------------------------------

        incident = (
            checker.check(
                frame_number,
                items,
                wrists,
            )
        )

        # ----------------------------------------------------
        # READ REAL HOLD STATE
        # ----------------------------------------------------

        held_ids = set()

        for (
            pair,
            state,
        ) in (
            checker._holds.items()
        ):

            if state.held:

                object_id = (
                    pair[1]
                )

                held_ids.add(
                    object_id
                )

        # ----------------------------------------------------
        # DRAW DETECTIONS
        # ----------------------------------------------------

        for obj in items:

            is_held = (
                obj.is_object
                and (
                    obj.tracker_id
                    in held_ids
                )
            )

            draw_object(
                frame,
                obj,
                is_held,
            )

        draw_wrists(
            frame,
            wrists,
        )

        # ----------------------------------------------------
        # HANDLE INCIDENT
        # ----------------------------------------------------

        if (
            incident
            is not None
        ):

            if not triggered:

                triggered = True

                trigger_frame = (
                    frame_number
                )

                print()
                print(
                    "=" * 72
                )

                print(
                    "INCIDENT TRIGGERED"
                )

                print(
                    "=" * 72
                )

                print(
                    f"Simulation frame: "
                    f"{frame_number}"
                )

                print(
                    f"Incident frame: "
                    f"{incident.frame_n}"
                )

                print(
                    f"Person ID: "
                    f"{incident.pid}"
                )

                print(
                    f"Object ID: "
                    f"{incident.obj_id}"
                )

                print(
                    f"Object class: "
                    f"{incident.class_name}"
                )

                print(
                    f"Confidence: "
                    f"{incident.confidence:.3f}"
                )

                print(
                    "=" * 72
                )

        # ----------------------------------------------------
        # DRAW INCIDENT MESSAGE
        # ----------------------------------------------------

        if triggered:

            cv2.putText(
                frame,

                (
                    "LITTERING INCIDENT "
                    "DETECTED"
                ),

                (
                    25,
                    115,
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                1.0,

                INCIDENT_COLOR,

                3,
            )

            cv2.putText(
                frame,

                (
                    f"Triggered at "
                    f"frame "
                    f"{trigger_frame}"
                ),

                (
                    25,
                    148,
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                INCIDENT_COLOR,

                2,
            )

        # ----------------------------------------------------
        # DRAW DIAGNOSTICS
        # ----------------------------------------------------

        draw_diagnostics(
            frame,
            checker,
            frame_number,
            phase,
            bottle_visible,
            triggered,
        )

        # ----------------------------------------------------
        # PAUSE MESSAGE
        # ----------------------------------------------------

        if paused:

            cv2.putText(
                frame,

                "PAUSED",

                (
                    FRAME_WIDTH
                    // 2
                    - 90,

                    FRAME_HEIGHT
                    // 2,
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                1.5,

                WARNING_COLOR,

                4,
            )

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        cv2.imshow(
            (
                "CleanStreets AI "
                "- Realistic Test"
            ),

            frame,
        )

        wait_time = (
            30
            if paused
            else max(
                1,
                int(
                    1000
                    / FPS
                ),
            )
        )

        key = (
            cv2.waitKey(
                wait_time
            )
            & 0xFF
        )

        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        if (
            key
            == ord("q")
        ):

            break

        # ----------------------------------------------------
        # RESTART
        # ----------------------------------------------------

        if (
            key
            == ord("r")
        ):

            random.seed(
                RANDOM_SEED
            )

            checker.reset()

            frame_number = 0

            triggered = False

            trigger_frame = None

            paused = False

            print(
                "\nSimulation restarted."
            )

            continue

        # ----------------------------------------------------
        # PAUSE
        # ----------------------------------------------------

        if (
            key
            == ord(" ")
        ):

            paused = (
                not paused
            )

        # ----------------------------------------------------
        # ADVANCE
        # ----------------------------------------------------

        if not paused:

            frame_number += 1

        # ----------------------------------------------------
        # AUTOMATIC RESTART
        # ----------------------------------------------------

        if (
            frame_number
            >= TOTAL_FRAMES
        ):

            print()

            if triggered:

                print(
                    "Simulation completed "
                    "successfully."
                )

                print(
                    f"Trigger frame: "
                    f"{trigger_frame}"
                )

            else:

                print(
                    "Simulation completed "
                    "without an incident."
                )

                print(
                    "This indicates that one "
                    "or more EventChecker "
                    "conditions were not met."
                )

            print(
                "Restarting..."
            )

            random.seed(
                RANDOM_SEED
            )

            checker.reset()

            frame_number = 0

            triggered = False

            trigger_frame = None

            paused = False

    cv2.destroyAllWindows()


if __name__ == "__main__":

    run_simulation()