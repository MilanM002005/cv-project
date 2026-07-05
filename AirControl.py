

import time
import math

import cv2
import mediapipe as mp
import pyautogui


# --------------------------------------------------------------------------
# Hand detection wrapper around MediaPipe
# --------------------------------------------------------------------------
class HandDetector:
    """Thin wrapper over MediaPipe Hands that returns landmark pixel coords."""

    # Landmark indices for each fingertip and the joint below it (PIP / IP).
    TIP_IDS = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky tips

    def __init__(self, max_hands=1, detection_conf=0.7, track_conf=0.6):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=track_conf,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None

    def find_hands(self, frame, draw=True):
        """Run inference on a BGR frame; optionally draw the skeleton."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)
        if draw and self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_lms, self.mp_hands.HAND_CONNECTIONS
                )
        return frame

    def get_landmarks(self, frame):
        """Return (landmark_list, handedness) for the first hand, or (None, None).

        landmark_list is [(id, x_px, y_px), ...] in pixel coordinates.
        handedness is 'Left' or 'Right'.
        """
        if not self.results or not self.results.multi_hand_landmarks:
            return None, None

        h, w = frame.shape[:2]
        hand_lms = self.results.multi_hand_landmarks[0]
        handedness = self.results.multi_handedness[0].classification[0].label

        landmarks = [
            (idx, int(lm.x * w), int(lm.y * h))
            for idx, lm in enumerate(hand_lms.landmark)
        ]
        return landmarks, handedness

    def fingers_up(self, landmarks, handedness):
        """Return a list of 5 booleans: [thumb, index, middle, ring, pinky]."""
        if landmarks is None:
            return [False] * 5

        fingers = []

        # Thumb: compare tip x to the joint x. Direction depends on which hand
        # and the fact that the webcam image is mirrored (selfie view).
        thumb_tip_x = landmarks[self.TIP_IDS[0]][1]
        thumb_joint_x = landmarks[self.TIP_IDS[0] - 1][1]
        if handedness == "Right":
            fingers.append(thumb_tip_x < thumb_joint_x)
        else:
            fingers.append(thumb_tip_x > thumb_joint_x)

        # Other four fingers: a finger is "up" if the tip is above its PIP joint
        # (smaller y). Image y grows downward, so tip_y < pip_y == extended.
        for tip_id in self.TIP_IDS[1:]:
            tip_y = landmarks[tip_id][2]
            pip_y = landmarks[tip_id - 2][2]
            fingers.append(tip_y < pip_y)

        return fingers


# --------------------------------------------------------------------------
# Gesture -> action controller
# --------------------------------------------------------------------------
class GestureController:
    """Maps a stable finger count to a system action, with debounce + cooldown."""

    ACTIONS = {
        0: ("Mute",          lambda: pyautogui.press("volumemute")),
        1: ("Volume Up",     lambda: pyautogui.press("volumeup")),
        2: ("Volume Down",   lambda: pyautogui.press("volumedown")),
        3: ("Next Track",    lambda: pyautogui.press("nexttrack")),
        4: ("Prev Track",    lambda: pyautogui.press("prevtrack")),
        5: ("Play / Pause",  lambda: pyautogui.press("playpause")),
    }

    def __init__(self, hold_frames=6, cooldown_sec=1.2):
        self.hold_frames = hold_frames      # how long a gesture must be steady
        self.cooldown_sec = cooldown_sec    # min gap between fired actions
        self._candidate = None
        self._streak = 0
        self._last_fire_time = 0.0
        self.last_action = ""

    def update(self, finger_count, armed):
        """Feed the current finger count; fire an action when stable + cooled down.

        Returns the action name if one fired this frame, else None.
        """
        # Debounce: require the same count for `hold_frames` consecutive frames.
        if finger_count == self._candidate:
            self._streak += 1
        else:
            self._candidate = finger_count
            self._streak = 1

        if self._streak < self.hold_frames:
            return None

        # Cooldown so a held gesture doesn't spam the same action.
        now = time.time()
        if now - self._last_fire_time < self.cooldown_sec:
            return None

        action = self.ACTIONS.get(finger_count)
        if not action:
            return None

        name, fn = action
        self._last_fire_time = now
        self.last_action = name

        if armed:
            try:
                fn()
            except Exception as exc:  # media keys can be flaky on some OSes
                print(f"[warn] could not send key for '{name}': {exc}")
        return name


# --------------------------------------------------------------------------
# On-screen heads-up display
# --------------------------------------------------------------------------
def draw_hud(frame, finger_count, armed, last_action, fps):
    h, w = frame.shape[:2]
    green, red, white = (0, 255, 0), (0, 0, 255), (255, 255, 255)

    # Status bar background
    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)

    status = "ARMED" if armed else "DISARMED (press 'a')"
    cv2.putText(frame, f"AirControl  |  {status}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, green if armed else red, 2)

    cv2.putText(frame, f"Fingers: {finger_count}   FPS: {int(fps)}", (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 1)

    if last_action:
        cv2.putText(frame, f">> {last_action}", (w - 260, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
    return frame


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Is another app using it?")

    detector = HandDetector(max_hands=1)
    controller = GestureController()

    armed = False           # start disarmed so you can test gestures safely
    prev_time = time.time()
    last_action_display = ""
    action_shown_at = 0.0

    print("AirControl running. Press 'a' to arm, 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # mirror for an intuitive selfie view
        frame = detector.find_hands(frame, draw=True)
        landmarks, handedness = detector.get_landmarks(frame)

        finger_count = 0
        if landmarks is not None:
            fingers = detector.fingers_up(landmarks, handedness)
            finger_count = sum(fingers)

            fired = controller.update(finger_count, armed)
            if fired:
                last_action_display = fired
                action_shown_at = time.time()
        else:
            controller.update(-1, armed)  # reset debounce when no hand

        # Fade the action label after 1.5s
        if last_action_display and time.time() - action_shown_at > 1.5:
            last_action_display = ""

        # FPS
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        frame = draw_hud(frame, finger_count, armed, last_action_display, fps)
        cv2.imshow("AirControl", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("a"):
            armed = not armed
            print(f"[info] automation {'ARMED' if armed else 'DISARMED'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
