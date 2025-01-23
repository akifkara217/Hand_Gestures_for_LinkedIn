# Import necessary libraries
import cv2  # For video capture and image processing
import mediapipe as mp  # For hand landmark detection
import time  # For time-based delay handling
import subprocess  # To execute ADB (Android Debug Bridge) commands
import pyautogui  # For screen resolution retrieval

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands  # Load the MediaPipe Hands solution
hands = mp_hands.Hands()  # Initialize the Hands module
mp_drawing = mp.solutions.drawing_utils  # Drawing utility for landmarks

# Get screen dimensions for gesture scaling
screen_width, screen_height = pyautogui.size()

# Function to execute ADB commands
def adb_command(command):
    try:
        subprocess.run(command, check=True)  # Execute the ADB command
        print(f"ADB Command Successfully Executed: {command}")
    except subprocess.CalledProcessError:
        print(f"ADB Command Error: {command}")

# Last detection times to handle gesture delays
last_detection_times = {
    "fingers_up_1": None,
    "fingers_up_2": None,
    "fingers_up_3": None,
    "fingers_up_4": None,
    "fingers_up_5": None,
    "single_tap": None,
    "like": None,
    "scroll_up": None,
    "scroll_down": None,
    "swipe_right": None,
    "swipe_left": None,
    "go_back": None,
}

# First detection flag to avoid immediate repetition
first_detection = {
    "fingers_up_1": True,
    "fingers_up_2": True,
    "fingers_up_3": True,
    "fingers_up_4": True,
    "fingers_up_5": True,
    "single_tap": True,
    "like": True,
    "scroll_up": True,
    "scroll_down": True,
    "swipe_right": True,
    "swipe_left": True,
    "go_back": True,
}

# Detection delay settings for each gesture
detection_delays = {
    "fingers_up_1":1,
    "fingers_up_2": 1,
    "fingers_up_3": 1,
    "fingers_up_4": 1,
    "fingers_up_5": 1,
    "single_tap": 0.3,
    "like": 2,
    "scroll_up": 1,
    "scroll_down": 0.2,
    "swipe_right": 1.3,
    "swipe_left": 1.3,
    "go_back": 2,
}

# Function to check if a gesture can be detected (based on time constraints)
def can_detect(action_name):
    current_time = time.time()  # Get current time
    if first_detection[action_name]:
        first_detection[action_name] = False
        last_detection_times[action_name] = current_time
        return True

    # Check if enough time has passed since last detection
    if last_detection_times[action_name] is None or current_time - last_detection_times[action_name] >= detection_delays[action_name]:
        last_detection_times[action_name] = current_time
        return True
    return False


def detect_single_tap(hand_landmarks):
    # Get the positions of the thumb tip, index tip, and wrist landmarks
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    wrist = hand_landmarks[0]

    # Get the positions of additional landmarks used for determining hand orientation
    point_5 = hand_landmarks[5]
    point_17 = hand_landmarks[17]

    # Check if the hand is facing the back (not palm up)
    is_hand_back_facing = point_5.x < point_17.x
    if not is_hand_back_facing:
        return False

    # Calculate the distance between the thumb and index tip
    distance = ((index_tip.x - thumb_tip.x) ** 2 + (index_tip.y - thumb_tip.y) ** 2) ** 0.5

    # Check if the other fingers are extended
    other_fingers_extended = all(
        hand_landmarks[i].y < hand_landmarks[i - 2].y for i in [12, 16, 20]
    )

    # Check if all fingers are above the wrist
    fingers_above_wrist = all(hand_landmarks[i].y < wrist.y for i in [4, 8, 12, 16, 20])

    # Return True if the distance is small, other fingers are extended, and all fingers are above the wrist
    if distance < 0.05 and other_fingers_extended and fingers_above_wrist:
        return True

    return False

def detect_like(hand_landmarks):
    # Loop through hand landmarks to check for the 'like' gesture
    for fingerNum, landmark in enumerate(hand_landmarks):
        # Break if any finger beyond the index is lower than the middle finger
        if fingerNum > 4 and landmark.y < hand_landmarks[2].y:
            break

        # Return False if the middle finger is above the wrist
        if hand_landmarks[2].y > hand_landmarks[0].y:
            return False

        # Check if the pinky (fingerNum == 20) is above the middle finger to detect the 'like' gesture
        if fingerNum == 20 and landmark.y > hand_landmarks[2].y:
            return True

    return False

def detect_fingers_up(hand_landmarks, expected_fingers_up):
    # Get the positions of finger tips and bases to detect extended fingers
    index_tip = hand_landmarks[8]
    index_base = hand_landmarks[6]
    middle_tip = hand_landmarks[12]
    middle_base = hand_landmarks[10]
    ring_tip = hand_landmarks[16]
    ring_base = hand_landmarks[14]
    pinky_tip = hand_landmarks[20]
    pinky_base = hand_landmarks[18]
    thumb_tip = hand_landmarks[4]
    thumb_base = hand_landmarks[3]

    point_5 = hand_landmarks[5]
    point_17 = hand_landmarks[17]

    # Check if the hand is facing the back
    is_hand_back_facing = point_5.x > point_17.x
    if not is_hand_back_facing:
        return False

    # Detect which fingers are up (extended)
    fingers_up = [
        index_tip.y < index_base.y,
        middle_tip.y < middle_base.y,
        ring_tip.y < ring_base.y,
        pinky_tip.y < pinky_base.y,
        thumb_tip.x > thumb_base.x,
    ]

    # If no fingers are up, return False
    if all(not finger for finger in fingers_up):
        return False

    # Count the number of fingers up
    num_fingers_up = sum(fingers_up)

    # Return True if the number of fingers up matches the expected value
    return num_fingers_up == expected_fingers_up

def detect_scroll_up(hand_landmarks):
    # Get hand landmarks to detect if the hand is facing up (palm facing up)
    point_5 = hand_landmarks[5]
    point_17 = hand_landmarks[17]

    # Check if the palm is facing up
    is_palm_facing = point_5.x < point_17.x
    if not is_palm_facing:
        return False

    # Check if only the index finger is up (other fingers should be down)
    fingers_up = [
        hand_landmarks[8].y < hand_landmarks[6].y,
        hand_landmarks[12].y < hand_landmarks[10].y,
        hand_landmarks[16].y < hand_landmarks[14].y,
        hand_landmarks[20].y < hand_landmarks[18].y,
    ]

    # Return True if index finger is up and others are down (scroll up gesture)
    if fingers_up == [True, False, False, False]:
        return True

    return False

def detect_scroll_down(hand_landmarks):
    # Check for back-facing hand (palm facing down)
    point_5 = hand_landmarks[5]
    point_17 = hand_landmarks[17]

    is_back_facing = point_5.x < point_17.x
    if not is_back_facing:
        return False

    # Check if the index finger is pointing up
    is_index_finger_up = hand_landmarks[8].y > hand_landmarks[6].y

    # Check if the index finger is pointing downward
    is_pointing_down = hand_landmarks[8].y > hand_landmarks[0].y

    # Return True if both conditions are met (scroll down gesture)
    if is_index_finger_up and is_pointing_down:
        return True

    return False

def detect_swipe_right(hand_landmarks):
    # Check if the hand is facing up (palm facing up)
    point_5 = hand_landmarks[5]
    point_17 = hand_landmarks[17]

    is_palm_facing = point_5.x < point_17.x
    if not is_palm_facing:
        detect_swipe_right.previous_x = None
        return False

    # Check if the index and middle fingers are up, and the ring and pinky are down
    fingers_up = [
        hand_landmarks[8].y < hand_landmarks[6].y,
        hand_landmarks[12].y < hand_landmarks[10].y,
        hand_landmarks[16].y < hand_landmarks[14].y,
        hand_landmarks[20].y < hand_landmarks[18].y,
    ]
    if fingers_up != [True, True, False, False]:
        detect_swipe_right.previous_x = None
        return False

    # Calculate horizontal movement based on the index finger's x position
    current_x = hand_landmarks[12].x * screen_width

    if detect_swipe_right.previous_x is None:
        detect_swipe_right.previous_x = current_x
        return False

    # Return True if the movement is more than 75 pixels to detect swipe right
    if current_x - detect_swipe_right.previous_x > 75:
        detect_swipe_right.previous_x = current_x
        return True

    detect_swipe_right.previous_x = current_x
    return False

def detect_swipe_left(hand_landmarks):
    # Check if the hand is facing up (palm facing up)
    point_5 = hand_landmarks[5]
    point_17 = hand_landmarks[17]

    is_palm_facing = point_5.x < point_17.x
    if not is_palm_facing:
        detect_swipe_left.previous_x = None
        return False

    # Check if the index and middle fingers are up, and the ring and pinky are down
    fingers_up = [
        hand_landmarks[8].y < hand_landmarks[6].y,
        hand_landmarks[12].y < hand_landmarks[10].y,
        hand_landmarks[16].y < hand_landmarks[14].y,
        hand_landmarks[20].y < hand_landmarks[18].y,
    ]

    if fingers_up != [True, True, False, False]:
        detect_swipe_left.previous_x = None
        return False

    # Calculate horizontal movement based on the index finger's x position
    current_x = hand_landmarks[12].x * screen_width

    if detect_swipe_left.previous_x is None:
        detect_swipe_left.previous_x = current_x
        return False

    # Return True if the movement is more than 75 pixels to detect swipe left
    if detect_swipe_left.previous_x - current_x > 75:
        detect_swipe_left.previous_x = current_x
        return True

    detect_swipe_left.previous_x = current_x
    return False

def go_back(hand_landmarks):
    # Check the positions of thumb, middle finger, and palm to detect 'go back' gesture
    thumb_tip = hand_landmarks[4]
    thumb_middle = hand_landmarks[3]
    palm_middle = hand_landmarks[9]

    palm_facing_camera = hand_landmarks[5].x < hand_landmarks[17].x

    # Check if the hand is facing the camera (palm facing camera)
    if not palm_facing_camera:
        return False

    # Check if other fingers are closed
    other_fingers_closed = all(
        hand_landmarks[i].y > hand_landmarks[i - 2].y
        for i in [8, 12, 16, 20]
    )

    # Check if the thumb is open and pointing to the left
    thumb_open_left = thumb_tip.x < thumb_middle.x and thumb_tip.x < palm_middle.x

    return palm_facing_camera and other_fingers_closed and thumb_open_left

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Check if the camera is working
    if not ret:
        print("Unable to read camera source!")
        break

    # Flip the frame horizontally for a mirror view
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB as required by the hand tracking model
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand landmarks
    results = hands.process(frame_rgb)

    # If landmarks are detected, iterate over each hand
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Draw the landmarks and connections on the frame
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            hand_landmarks = landmarks.landmark

            # Detect and perform actions based on the number of fingers up
            if can_detect("fingers_up_1") and detect_fingers_up(hand_landmarks, 1):
                print("Number 1 detected (1 finger raised).")
                adb_command(["adb", "shell", "input", "tap", "108", "2183"])

            if can_detect("fingers_up_2") and detect_fingers_up(hand_landmarks, 2):
                print("Number 2 detected (2 fingers raised).")
                adb_command(["adb", "shell", "input", "tap", "324", "2183"])

            if can_detect("fingers_up_3") and detect_fingers_up(hand_landmarks, 3):
                print("Number 3 detected (3 fingers raised).")
                adb_command(["adb", "shell", "input", "tap", "540", "2183"])

            if can_detect("fingers_up_4") and detect_fingers_up(hand_landmarks, 4):
                print("Number 4 detected (4 fingers raised).")
                adb_command(["adb", "shell", "input", "tap", "756", "2183"])

            if can_detect("fingers_up_5") and detect_fingers_up(hand_landmarks, 5):
                print("Number 5 detected (all fingers raised).")
                adb_command(["adb", "shell", "input", "tap", "972", "2183"])

            # Detect single tap gesture
            if can_detect("single_tap") and detect_single_tap(hand_landmarks):
                print("Single tap detected.")
                adb_command(["adb", "shell", "input", "tap", "540", "1200"])

            # Detect like gesture (thumb up)
            if can_detect("like") and detect_like(hand_landmarks):
                print("Like detected.")
                adb_command(["adb", "shell", "input", "tap", "540", "1200"])
                time.sleep(0.5)
                adb_command(["adb", "shell", "input", "tap", "108", "2183"])

            # Detect scroll up gesture
            if can_detect("scroll_up") and detect_scroll_up(hand_landmarks):
                print("Scrolling page up.")
                adb_command(["adb", "shell", "input", "swipe", "500", "300", "500", "900"])

            # Detect scroll down gesture
            if can_detect("scroll_down") and detect_scroll_down(hand_landmarks):
                print("Scrolling page down.")
                adb_command(["adb", "shell", "input", "swipe", "500", "700", "500", "300"])

            # Detect swipe right gesture
            if can_detect("swipe_right") and detect_swipe_right(hand_landmarks):
                print("Swiping page to the right.")
                adb_command(["adb", "shell", "input", "swipe", "200", "860", "1000", "860", "300"])

            # Detect swipe left gesture
            if can_detect("swipe_left") and detect_swipe_left(hand_landmarks):
                print("Swiping page to the left.")
                adb_command(["adb", "shell", "input", "swipe", "800", "860", "200", "860", "300"])

            # Detect go back gesture (thumb open left)
            if can_detect("go_back") and go_back(hand_landmarks):
                print("Going back.")
                adb_command(["adb", "shell", "input", "tap", "90", "150"])

    # Display the frame with drawn landmarks and connections
    cv2.imshow("Hand Gesture Control", frame)

    # Wait for a key event (1 ms) and continue
    cv2.waitKey(1)

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

