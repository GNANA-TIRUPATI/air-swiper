import sys
import cv2
import mediapipe as mp
import pyautogui
import time
from pathlib import Path
from collections import deque

# Configure UTF-8 encoding for Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ============================================================
# CONFIGURATION
# ============================================================

# Default to 1 (physical webcam instead of virtual cameras like DroidCam)
CAMERA_INDEX = 1

# Allow passing camera index via command line, e.g. 'python v1.py 0' or 'python v1.py --camera=1'
for arg in sys.argv[1:]:
    if arg.isdigit():
        CAMERA_INDEX = int(arg)
        break
    elif arg.startswith("--camera="):
        try:
            CAMERA_INDEX = int(arg.split("=")[1])
            break
        except ValueError:
            pass

MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"


# ============================================================
# ACTIVATION
# ============================================================

# Show your physical RIGHT PALM for this duration
# when starting the program.
ACTIVATION_HOLD_TIME = 0.7


# ============================================================
# AIR SWIPE SETTINGS
# ============================================================

# Minimum vertical movement.
#
# Smaller = easier triggering.
SWIPE_DISTANCE = 0.035


# Maximum time used for detecting a swipe.
#
# Smaller = faster response.
SWIPE_TIME = 0.20


# ============================================================
# MINIMUM SWIPE SPEED
# ============================================================

# Action is allowed when:
#
# speed >= 0.30
#
MIN_SWIPE_SPEED = 0.30


# ============================================================
# VERTICAL MOVEMENT RATIO
# ============================================================

# Vertical movement must be stronger than horizontal movement.
MIN_VERTICAL_RATIO = 1.25


# ============================================================
# ACTION COOLDOWN
# ============================================================

# Minimum time between two swipe actions.
SWIPE_COOLDOWN = 0.55


# ============================================================
# MOVEMENT HISTORY
# ============================================================

HISTORY_SIZE = 20


# ============================================================
# MEDIAPIPE SETTINGS
# ============================================================

MIN_DETECTION_CONFIDENCE = 0.60

MIN_PRESENCE_CONFIDENCE = 0.60

MIN_TRACKING_CONFIDENCE = 0.60


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():

    print()
    print(
        "ERROR: hand_landmarker.task was not found."
    )
    print()
    print(
        "Expected location:"
    )
    print(MODEL_PATH)
    print()

    raise SystemExit(1)


if MODEL_PATH.stat().st_size == 0:

    print()
    print(
        "ERROR: hand_landmarker.task is EMPTY."
    )
    print()
    print(
        "Download the real Hand Landmarker model."
    )
    print()

    raise SystemExit(1)


# ============================================================
# MEDIAPIPE TASK API
# ============================================================

BaseOptions = mp.tasks.BaseOptions

RunningMode = (
    mp.tasks.vision.RunningMode
)

HandLandmarker = (
    mp.tasks.vision.HandLandmarker
)

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)


options = HandLandmarkerOptions(

    base_options=BaseOptions(

        model_asset_path=str(
            MODEL_PATH
        )

    ),

    running_mode=RunningMode.VIDEO,

    num_hands=1,

    min_hand_detection_confidence=(
        MIN_DETECTION_CONFIDENCE
    ),

    min_hand_presence_confidence=(
        MIN_PRESENCE_CONFIDENCE
    ),

    min_tracking_confidence=(
        MIN_TRACKING_CONFIDENCE
    )
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX
)


if not cap.isOpened():

    print()
    print(
        "ERROR: Camera could not be opened."
    )
    print()

    raise SystemExit(1)


# ============================================================
# GET PALM CENTER
# ============================================================

def get_palm_center(landmarks):

    """
    Calculate a stable palm center using
    multiple palm landmarks.
    """

    palm_ids = [

        0,   # Wrist
        5,   # Index MCP
        9,   # Middle MCP
        13,  # Ring MCP
        17   # Pinky MCP

    ]


    x = sum(

        landmarks[i].x

        for i in palm_ids

    ) / len(palm_ids)


    y = sum(

        landmarks[i].y

        for i in palm_ids

    ) / len(palm_ids)


    return x, y


# ============================================================
# OPEN PALM DETECTION
# ============================================================

def is_open_palm(landmarks):

    index_up = (

        landmarks[8].y
        <
        landmarks[6].y

    )


    middle_up = (

        landmarks[12].y
        <
        landmarks[10].y

    )


    ring_up = (

        landmarks[16].y
        <
        landmarks[14].y

    )


    pinky_up = (

        landmarks[20].y
        <
        landmarks[18].y

    )


    finger_count = sum([

        index_up,
        middle_up,
        ring_up,
        pinky_up

    ])


    return finger_count >= 3


# ============================================================
# DRAW HAND
# ============================================================

def draw_hand(
    frame,
    landmarks
):

    connections = [

        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        (13, 17),
        (17, 18),
        (18, 19),
        (19, 20),

        (0, 17)

    ]


    height, width = (
        frame.shape[:2]
    )


    points = []


    for landmark in landmarks:

        x = int(

            landmark.x
            *
            width

        )


        y = int(

            landmark.y
            *
            height

        )


        points.append(

            (x, y)

        )


        cv2.circle(

            frame,

            (x, y),

            4,

            (0, 255, 0),

            -1

        )


    for start, end in connections:

        cv2.line(

            frame,

            points[start],

            points[end],

            (255, 255, 255),

            2

        )


    # --------------------------------------------------------
    # Draw palm center
    # --------------------------------------------------------

    center_x, center_y = (
        get_palm_center(
            landmarks
        )
    )


    center_x = int(

        center_x
        *
        width

    )


    center_y = int(

        center_y
        *
        height

    )


    cv2.circle(

        frame,

        (center_x, center_y),

        9,

        (0, 0, 255),

        -1

    )


# ============================================================
# AIR SWIPE CONTROLLER
# ============================================================

class AirSwipeController:

    def __init__(self):

        # ----------------------------------------------------
        # Session
        # ----------------------------------------------------

        self.active = False

        self.activation_start = None


        # ----------------------------------------------------
        # Movement history
        # ----------------------------------------------------

        self.history = deque(

            maxlen=HISTORY_SIZE

        )


        # ----------------------------------------------------
        # Action timing
        # ----------------------------------------------------

        self.last_action = 0


        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        self.dx = 0.0

        self.dy = 0.0

        self.speed = 0.0

        self.action = "NONE"


        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        self.message = (

            "Show RIGHT PALM to activate"

        )


    # ========================================================
    # ACTIVATION
    # ========================================================

    def process_activation(
        self,
        palm
    ):

        if self.active:

            return


        if palm:

            if self.activation_start is None:

                self.activation_start = (

                    time.time()

                )


            elapsed = (

                time.time()
                -
                self.activation_start

            )


            self.message = (

                f"Activating... "
                f"{elapsed:.1f}s"

            )


            if (

                elapsed
                >=
                ACTIVATION_HOLD_TIME

            ):

                self.active = True

                self.activation_start = None

                self.history.clear()

                self.message = (

                    "AIR SWIPE ACTIVE"

                )


                print()

                print(
                    "===================================="
                )

                print(
                    " RIGHT-HAND AIR SWIPER ACTIVATED"
                )

                print(
                    "===================================="
                )

                print()

                print(
                    "DOWN = NEXT SHORT / REEL"
                )

                print(
                    "UP   = PREVIOUS SHORT / REEL"
                )

                print()

                print(
                    "Minimum swipe speed = 0.30"
                )

                print()

                print(
                    "Palm activation is required only once."
                )

                print()

                print(
                    "===================================="
                )

                print()


        else:

            self.activation_start = None

            self.message = (

                "Show RIGHT PALM to activate"

            )


    # ========================================================
    # TRACK MOVEMENT
    # ========================================================

    def track_movement(
        self,
        x,
        y
    ):

        now = time.time()


        # ----------------------------------------------------
        # Store current position
        # ----------------------------------------------------

        self.history.append(

            (
                now,
                x,
                y
            )

        )


        # ----------------------------------------------------
        # Need enough history
        # ----------------------------------------------------

        if len(self.history) < 3:

            return


        # ----------------------------------------------------
        # Current point
        # ----------------------------------------------------

        current_time = (

            self.history[-1][0]

        )


        current_x = (

            self.history[-1][1]

        )


        current_y = (

            self.history[-1][2]

        )


        # ----------------------------------------------------
        # Find an older point inside SWIPE_TIME
        # ----------------------------------------------------

        start_point = None


        for point in reversed(

            self.history

        ):

            point_time = point[0]


            if (

                current_time
                -
                point_time

                <=

                SWIPE_TIME

            ):

                start_point = point

            else:

                break


        if start_point is None:

            return


        # ----------------------------------------------------
        # Starting point
        # ----------------------------------------------------

        start_time = (
            start_point[0]
        )

        start_x = (
            start_point[1]
        )

        start_y = (
            start_point[2]
        )


        # ----------------------------------------------------
        # Elapsed time
        # ----------------------------------------------------

        elapsed = (

            current_time
            -
            start_time

        )


        if elapsed <= 0:

            return


        # ----------------------------------------------------
        # Movement
        # ----------------------------------------------------

        self.dx = (

            current_x
            -
            start_x

        )


        self.dy = (

            current_y
            -
            start_y

        )


        # ----------------------------------------------------
        # SPEED
        # ----------------------------------------------------

        self.speed = (

            abs(self.dy)
            /
            elapsed

        )


        # ====================================================
        # SPEED FILTER
        #
        # ACTION IS ALLOWED WHEN:
        #
        # speed >= 0.30
        # ====================================================

        if (

            self.speed
            <
            MIN_SWIPE_SPEED

        ):

            return


        # ----------------------------------------------------
        # COOLDOWN
        # ----------------------------------------------------

        if (

            current_time
            -
            self.last_action

            <

            SWIPE_COOLDOWN

        ):

            return


        # ----------------------------------------------------
        # Vertical movement
        # ----------------------------------------------------

        vertical = abs(

            self.dy

        )


        horizontal = abs(

            self.dx

        )


        # ----------------------------------------------------
        # Distance filter
        # ----------------------------------------------------

        if (

            vertical
            <
            SWIPE_DISTANCE

        ):

            return


        # ----------------------------------------------------
        # Vertical dominance
        # ----------------------------------------------------

        if horizontal > 0:

            vertical_ratio = (

                vertical
                /
                horizontal

            )

        else:

            vertical_ratio = 999


        if (

            vertical_ratio
            <
            MIN_VERTICAL_RATIO

        ):

            return


        # ====================================================
        # DOWN
        # ====================================================

        if self.dy > 0:

            self.perform_down()

            return


        # ====================================================
        # UP
        # ====================================================

        if self.dy < 0:

            self.perform_up()

            return


    # ========================================================
    # PERFORM DOWN
    # ========================================================

    def perform_down(self):

        now = time.time()


        pyautogui.scroll(

            -7

        )


        self.last_action = now


        self.action = (

            "DOWN"

        )


        self.message = (

            "↓ NEXT"

        )


        print()

        print(
            ">>> RIGHT HAND DOWN → NEXT"
        )


        self.history.clear()


    # ========================================================
    # PERFORM UP
    # ========================================================

    def perform_up(self):

        now = time.time()


        pyautogui.scroll(

            7

        )


        self.last_action = now


        self.action = (

            "UP"

        )


        self.message = (

            "↑ PREVIOUS"

        )


        print()

        print(
            ">>> RIGHT HAND UP → PREVIOUS"
        )


        self.history.clear()


    # ========================================================
    # RESET MOVEMENT
    # ========================================================

    def reset_history(self):

        self.history.clear()

        self.dx = 0.0

        self.dy = 0.0

        self.speed = 0.0


# ============================================================
# CREATE CONTROLLER
# ============================================================

controller = (
    AirSwipeController()
)


# ============================================================
# START
# ============================================================

print()

print(
    "=============================================="
)

print(
    " LANDSCAPE AIR SWIPER V6"
)

print(
    "=============================================="
)

print()

print(
    "Physical RIGHT HAND mode"
)

print(
    f"Camera device index: {CAMERA_INDEX}"
)

print()

print(
    "1. Show your RIGHT PALM once."
)

print(
    "2. Hold for approximately 0.7 seconds."
)

print()

print(
    "After activation:"
)

print(
    "   RIGHT HAND ↓ = NEXT"
)

print(
    "   RIGHT HAND ↑ = PREVIOUS"
)

print()

print(
    "Minimum swipe speed: 0.30"
)

print()

print(
    "Press ESC to exit."
)

print()


# ============================================================
# MEDIAPIPE LOOP
# ============================================================

with HandLandmarker.create_from_options(
    options
) as landmarker:

    timestamp_ms = 0


    failed_frames = 0

    while True:

        # ====================================================
        # CAMERA
        # ====================================================

        success, frame = (
            cap.read()
        )

        if not success:
            failed_frames += 1
            if failed_frames > 15:
                print()
                print("ERROR: Could not read camera.")
                print("If another application (or another instance of this script) is using the webcam, please close it and try again.")
                print()
                break
            time.sleep(0.05)
            continue

        failed_frames = 0


        # ====================================================
        # MIRROR CAMERA
        # ====================================================

        frame = cv2.flip(

            frame,

            1

        )


        # ====================================================
        # BGR → RGB
        # ====================================================

        rgb_frame = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2RGB

        )


        # ====================================================
        # MEDIAPIPE IMAGE
        # ====================================================

        mp_image = mp.Image(

            image_format=(
                mp.ImageFormat.SRGB
            ),

            data=rgb_frame

        )


        timestamp_ms += 1


        # ====================================================
        # DETECT HAND
        # ====================================================

        result = (

            landmarker.detect_for_video(

                mp_image,

                timestamp_ms

            )

        )


        hand_text = (
            "NO HAND"
        )


        # ====================================================
        # HAND FOUND
        # ====================================================

        if result.hand_landmarks:

            landmarks = (

                result.hand_landmarks[0]

            )


            # ------------------------------------------------
            # HANDEDNESS
            #
            # Because the camera view is mirrored:
            #
            # MediaPipe LEFT  -> Physical RIGHT
            # MediaPipe RIGHT -> Physical LEFT
            # ------------------------------------------------

            physical_right = True


            detected_label = (
                "UNKNOWN"
            )


            if result.handedness:

                detected_label = (

                    result
                    .handedness[0][0]
                    .category_name
                    .lower()

                )


                if detected_label == "left":

                    physical_right = True

                else:

                    physical_right = False


            # ------------------------------------------------
            # Display physical hand
            # ------------------------------------------------

            if physical_right:

                hand_text = (
                    "PHYSICAL RIGHT HAND"
                )

            else:

                hand_text = (
                    "PHYSICAL LEFT HAND"
                )


            # ------------------------------------------------
            # Draw hand
            # ------------------------------------------------

            draw_hand(

                frame,

                landmarks

            )


            # =================================================
            # RIGHT HAND
            # =================================================

            if physical_right:

                # ---------------------------------------------
                # Palm
                # ---------------------------------------------

                palm = (

                    is_open_palm(

                        landmarks

                    )

                )


                # ---------------------------------------------
                # Palm center
                # ---------------------------------------------

                x, y = (

                    get_palm_center(

                        landmarks

                    )

                )


                # =============================================
                # ACTIVATION
                # =============================================

                if not controller.active:

                    controller.process_activation(

                        palm

                    )


                # =============================================
                # ACTIVE AIR SWIPE
                # =============================================

                else:

                    controller.track_movement(

                        x,

                        y

                    )


            # =================================================
            # LEFT HAND
            # =================================================

            else:

                controller.reset_history()


                controller.message = (

                    "LEFT HAND — USE RIGHT HAND"

                )


        # ====================================================
        # NO HAND
        # ====================================================

        else:

            controller.reset_history()


            hand_text = (
                "NO HAND"
            )


            if not controller.active:

                controller.activation_start = None

                controller.message = (

                    "Show RIGHT PALM to activate"

                )

            else:

                controller.message = (
                    "Ready"
                )


        # ====================================================
        # UI
        # ====================================================

        height, width = (
            frame.shape[:2]
        )


        # ----------------------------------------------------
        # Header background
        # ----------------------------------------------------

        cv2.rectangle(

            frame,

            (0, 0),

            (width, 185),

            (0, 0, 0),

            -1

        )


        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        status = (

            "ACTIVE"

            if controller.active

            else

            "NOT ACTIVE"

        )


        cv2.putText(

            frame,

            f"Status: {status}",

            (20, 30),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (0, 255, 0),

            2

        )


        # ----------------------------------------------------
        # Hand
        # ----------------------------------------------------

        cv2.putText(

            frame,

            f"Hand: {hand_text}",

            (20, 60),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.58,

            (255, 255, 255),

            2

        )


        # ----------------------------------------------------
        # DY
        # ----------------------------------------------------

        cv2.putText(

            frame,

            f"DY: {controller.dy:.4f}",

            (20, 90),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.52,

            (255, 255, 255),

            2

        )


        # ----------------------------------------------------
        # SPEED
        # ----------------------------------------------------

        cv2.putText(

            frame,

            f"Speed: {controller.speed:.2f}",

            (190, 90),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.52,

            (255, 255, 255),

            2

        )


        # ----------------------------------------------------
        # MINIMUM SPEED
        # ----------------------------------------------------

        cv2.putText(

            frame,

            f"Min Speed: {MIN_SWIPE_SPEED:.2f}",

            (350, 90),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.52,

            (255, 255, 255),

            2

        )


        # ----------------------------------------------------
        # ACTION
        # ----------------------------------------------------

        cv2.putText(

            frame,

            f"Action: {controller.action}",

            (20, 120),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.52,

            (255, 255, 255),

            2

        )


        # ----------------------------------------------------
        # MESSAGE
        # ----------------------------------------------------

        cv2.putText(

            frame,

            controller.message,

            (20, 150),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.52,

            (0, 255, 255),

            2

        )


        # ----------------------------------------------------
        # THRESHOLD
        # ----------------------------------------------------

        cv2.putText(

            frame,

            "Swipe speed threshold: 0.30",

            (20, 175),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.45,

            (180, 180, 180),

            1

        )


        # ====================================================
        # INSTRUCTIONS
        # ====================================================

        cv2.putText(

            frame,

            "RIGHT HAND  ↓  NEXT",

            (20, height - 42),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (255, 255, 255),

            2

        )


        cv2.putText(

            frame,

            "RIGHT HAND  ↑  PREVIOUS",

            (20, height - 15),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (255, 255, 255),

            2

        )


        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(

            "Landscape Air Swiper V6",

            frame

        )


        # ====================================================
        # ESC
        # ====================================================

        key = (

            cv2.waitKey(1)
            &
            0xFF

        )


        if key == 27:

            break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()


print()

print(
    "Landscape Air Swiper stopped."
)

