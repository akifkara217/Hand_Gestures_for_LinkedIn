import cv2
import mediapipe as mp
import time
import subprocess
import pyautogui

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()

def adb_command(command):
    try:
        subprocess.run(command, check=True)
        print(f"ADB Komutu Başarıyla Çalıştırıldı: {command}")
    except subprocess.CalledProcessError:
        print(f"ADB Komutu Hatası: {command}")

last_detection_times = {
    "fingers_up_1": None,
    "fingers_up_2": None,
    "fingers_up_3": None,
    "fingers_up_4": None,
    "fingers_up_5": None,
    "ok_sign": None,
    "thumbs_up": None,
    "scroll_up": None,
    "point_down": None,
    "swipe_right": None,
    "scroll_left": None,
    "go_back": None,
}

first_detection = {
    "fingers_up_1": True,
    "fingers_up_2": True,
    "fingers_up_3": True,
    "fingers_up_4": True,
    "fingers_up_5": True,
    "ok_sign": True,
    "thumbs_up": True,
    "scroll_up": True,
    "point_down": True,
    "swipe_right": True,
    "scroll_left": True,
    "go_back": True,
}

detection_delays = {
    "fingers_up_1":1,
    "fingers_up_2": 1,
    "fingers_up_3": 1,
    "fingers_up_4": 1,
    "fingers_up_5": 1,
    "ok_sign": 0.3,
    "thumbs_up": 2,
    "scroll_up": 1,
    "point_down": 0.2,
    "swipe_right": 1.3,
    "scroll_left": 1.3,
    "go_back": 2,
}

def can_detect(action_name):
    current_time = time.time()

    if first_detection[action_name]:
        first_detection[action_name] = False
        last_detection_times[action_name] = current_time
        return True


    if last_detection_times[action_name] is None or current_time - last_detection_times[action_name] >= detection_delays[action_name]:
        last_detection_times[action_name] = current_time
        return True

    return False


def detect_ok_sign(hand_landmarks):

    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]

    wrist = hand_landmarks[0]


    point_5 = hand_landmarks[5]
    point_17 = hand_landmarks[17]


    is_hand_back_facing = point_5.x < point_17.x
    if not is_hand_back_facing:
        return False


    distance = ((index_tip.x - thumb_tip.x) ** 2 + (index_tip.y - thumb_tip.y) ** 2) ** 0.5


    other_fingers_extended = all(
        hand_landmarks[i].y < hand_landmarks[i - 2].y for i in [12, 16, 20]
    )

    fingers_above_wrist = all(hand_landmarks[i].y < wrist.y for i in [4, 8, 12, 16, 20])

    if distance < 0.05 and other_fingers_extended and fingers_above_wrist:
        return True

    return False


def detect_thumbs_up(hand_landmarks):

    for fingerNum, landmark in enumerate(hand_landmarks):

        if fingerNum > 4 and landmark.y < hand_landmarks[2].y:
            break

        if hand_landmarks[2].y > hand_landmarks[0].y:
            return False

        if fingerNum == 20 and landmark.y > hand_landmarks[2].y:
            return True

    return False


def detect_fingers_up(hand_landmarks, expected_fingers_up):

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

    is_hand_back_facing = point_5.x > point_17.x
    if not is_hand_back_facing:
        return False

    fingers_up = [
        index_tip.y < index_base.y,
        middle_tip.y < middle_base.y,
        ring_tip.y < ring_base.y,
        pinky_tip.y < pinky_base.y,
        thumb_tip.x > thumb_base.x,
    ]

    if all(not finger for finger in fingers_up):
        return False

    num_fingers_up = sum(fingers_up)

    return num_fingers_up == expected_fingers_up


def detect_scroll_up(hand_landmarks):

    point_5 = hand_landmarks[5]
    point_17 = hand_landmarks[17]

    is_palm_facing = point_5.x < point_17.x
    if not is_palm_facing:
        return False

    fingers_up = [
        hand_landmarks[8].y < hand_landmarks[6].y,
        hand_landmarks[12].y < hand_landmarks[10].y,
        hand_landmarks[16].y < hand_landmarks[14].y,
        hand_landmarks[20].y < hand_landmarks[18].y,
    ]

    if fingers_up == [True, False, False, False]:
        return True

    return False

def detect_scroll_down(hand_landmarks):

    point_5 = hand_landmarks[5]
    point_17 = hand_landmarks[17]

    is_back_facing = point_5.x < point_17.x
    if not is_back_facing:
        return False

    is_index_finger_up = hand_landmarks[8].y > hand_landmarks[6].y

    is_pointing_down = hand_landmarks[8].y > hand_landmarks[0].y

    if is_index_finger_up and is_pointing_down:
        return True

    return False

def detect_swipe_right(hand_landmarks):

    point_5 = hand_landmarks[5]
    point_17 = hand_landmarks[17]

    is_palm_facing = point_5.x < point_17.x
    if not is_palm_facing:
        detect_swipe_right.previous_x = None
        return False

    fingers_up = [
        hand_landmarks[8].y < hand_landmarks[6].y,
        hand_landmarks[12].y < hand_landmarks[10].y,
        hand_landmarks[16].y < hand_landmarks[14].y,
        hand_landmarks[20].y < hand_landmarks[18].y,
    ]
    if fingers_up != [True, True, False, False]:
        detect_swipe_right.previous_x = None
        return False


    current_x = hand_landmarks[12].x * screen_width

    if detect_swipe_right.previous_x is None:
        detect_swipe_right.previous_x = current_x
        return False

    if current_x - detect_swipe_right.previous_x > 75:
        detect_swipe_right.previous_x = current_x
        return True

    detect_swipe_right.previous_x = current_x
    return False




def detect_swipe_left(hand_landmarks):
    point_5 = hand_landmarks[5]
    point_17 = hand_landmarks[17]

    is_palm_facing = point_5.x < point_17.x
    if not is_palm_facing:
        detect_swipe_left.previous_x = None
        return False

    fingers_up = [
        hand_landmarks[8].y < hand_landmarks[6].y,
        hand_landmarks[12].y < hand_landmarks[10].y,
        hand_landmarks[16].y < hand_landmarks[14].y,
        hand_landmarks[20].y < hand_landmarks[18].y,
    ]

    if fingers_up != [True, True, False, False]:
        detect_swipe_left.previous_x = None
        return False

    current_x = hand_landmarks[12].x * screen_width

    if detect_swipe_left.previous_x is None:
        detect_swipe_left.previous_x = current_x
        return False

    if detect_swipe_left.previous_x - current_x > 75:
        detect_swipe_left.previous_x = current_x
        return True

    detect_swipe_left.previous_x = current_x
    return False


def go_back(hand_landmarks):

    thumb_tip = hand_landmarks[4]
    thumb_middle = hand_landmarks[3]
    palm_middle = hand_landmarks[9]

    palm_facing_camera = hand_landmarks[5].x < hand_landmarks[17].x

    if not palm_facing_camera:
        return False

    other_fingers_closed = all(
        hand_landmarks[i].y > hand_landmarks[i - 2].y
        for i in [8, 12, 16, 20]
    )

    thumb_open_left = thumb_tip.x < thumb_middle.x and thumb_tip.x < palm_middle.x

    return palm_facing_camera and other_fingers_closed and thumb_open_left

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Unable to read camera source!")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            hand_landmarks = landmarks.landmark

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

            if can_detect("ok_sign") and detect_ok_sign(hand_landmarks):
                print("Single tap detected.")
                adb_command(["adb", "shell", "input", "tap", "540", "1200"])

            if can_detect("thumbs_up") and detect_thumbs_up(hand_landmarks):
                print("Like detected.")
                adb_command(["adb", "shell", "input", "tap", "540", "1200"])
                time.sleep(0.5)
                adb_command(["adb", "shell", "input", "tap", "108", "2183"])

            if can_detect("scroll_up") and detect_scroll_up(hand_landmarks):
                print("Scrolling page up.")
                adb_command(["adb", "shell", "input", "swipe", "500", "300", "500", "900"])

            if can_detect("point_down") and detect_scroll_down(hand_landmarks):
                print("Scrolling page down.")
                adb_command(["adb", "shell", "input", "swipe", "500", "700", "500", "300"])

            if can_detect("swipe_right") and detect_swipe_right(hand_landmarks):
                print("Swiping page to the right.")
                adb_command(["adb", "shell", "input", "swipe", "200", "860", "1000", "860", "300"])

            if can_detect("scroll_left") and detect_swipe_left(hand_landmarks):
                print("Swiping page to the left.")
                adb_command(["adb", "shell", "input", "swipe", "800", "860", "200", "860", "300"])

            if can_detect("go_back") and go_back(hand_landmarks):
                print("Going back.")
                adb_command(["adb", "shell", "input", "tap", "90", "150"])

    cv2.imshow("Hand Gesture Control", frame)

    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()
