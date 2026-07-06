

import os
import time
import threading
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp


# ------------------------- Config -----------------------------------------
MODEL = "____"      # or "claude-haiku-4-5" for faster/cheaper
SHOW_WINDOW = False            # True works on Windows/Linux; may be unstable on macOS
DEBUG = False                  # print raw facial feature values for tuning
SMOOTH_N = 15                  # frames to majority-vote over
# --------------------------------------------------------------------------

# Face Mesh landmark indices
L_EYE_OUT, R_EYE_OUT = 33, 263        # outer eye corners (inter-ocular scale)
MOUTH_L, MOUTH_R = 61, 291            # mouth corners
LIP_TOP, LIP_BOT = 13, 14             # inner lip center (openness)
R_BROW, R_EYE_TOP = 105, 159          # right brow / right eye top
L_BROW, L_EYE_TOP = 334, 386          # left brow / left eye top


def dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


class EmotionDetector:
    """Runs Face Mesh in a background thread; exposes a smoothed mood."""

    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._mood = "Neutral"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._history = deque(maxlen=SMOOTH_N)

    @property
    def mood(self):
        with self._lock:
            return self._mood

    def stop(self):
        self._stop.set()

    def _features(self, lms, w, h):
        """Return scale-normalized facial features from landmarks."""
        p = [(lm.x * w, lm.y * h) for lm in lms]
        scale = dist(p[L_EYE_OUT], p[R_EYE_OUT]) or 1.0   # inter-ocular distance

        mouth_open = dist(p[LIP_TOP], p[LIP_BOT]) / scale
        mouth_width = dist(p[MOUTH_L], p[MOUTH_R]) / scale

        # Smile: mouth corners raised above the lip's vertical center.
        lip_center_y = (p[LIP_TOP][1] + p[LIP_BOT][1]) / 2.0
        corner_y = (p[MOUTH_L][1] + p[MOUTH_R][1]) / 2.0
        smile = (lip_center_y - corner_y) / scale          # +ve = corners up

        # Brow raise: brow-to-eye vertical gap (larger = raised, e.g. surprise).
        brow_raise = (dist(p[R_BROW], p[R_EYE_TOP]) +
                      dist(p[L_BROW], p[L_EYE_TOP])) / (2.0 * scale)

        return dict(mouth_open=mouth_open, mouth_width=mouth_width,
                    smile=smile, brow_raise=brow_raise)

    def _classify(self, f):
        """Coarse rule-based mood from features. Tune thresholds with DEBUG=True."""
        if f["mouth_open"] > 0.28 and f["brow_raise"] > 0.42:
            return "Surprised"
        if f["smile"] > 0.06 and f["mouth_width"] > 0.95:
            return "Happy"
        if f["smile"] < -0.02:
            return "Sad"
        return "Neutral"

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[error] Could not open webcam; mood stays Neutral.")
            return

        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            results = self.face_mesh.process(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            label = "Neutral"
            if results.multi_face_landmarks:
                feats = self._features(
                    results.multi_face_landmarks[0].landmark, w, h)
                label = self._classify(feats)
                if DEBUG:
                    print("  ".join(f"{k}={v:.2f}" for k, v in feats.items()),
                          "->", label)

            self._history.append(label)
            smoothed = Counter(self._history).most_common(1)[0][0]
            with self._lock:
                self._mood = smoothed

            if SHOW_WINDOW:
                cv2.putText(frame, f"Mood: {smoothed}", (12, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
                cv2.imshow("Emotion sensor", frame)
                cv2.waitKey(1)

            time.sleep(0.02)

        cap.release()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()


TONE_GUIDE = {
    "Happy": "The user looks happy. Match their upbeat energy warmly.",
    "Sad": "The user looks sad or low. Be gentle, kind, and supportive; "
           "don't be relentlessly cheerful.",
    "Surprised": "The user looks surprised or confused. Be clear, calm, "
                 "and reassuring.",
    "Neutral": "The user's expression is neutral. Respond naturally.",
}


def build_system_prompt(mood):
    guide = TONE_GUIDE.get(mood, TONE_GUIDE["Neutral"])
    return (
        "You are a warm, emotionally attuned conversational assistant. "
        f"{guide} Do not explicitly announce that you detected their emotion "
        "unless it's clearly relevant. Keep replies concise and natural."
    )


def main():
    try:
        from anthropic import Anthropic
    except ImportError:
        print("Install the SDK first:  pip install anthropic")
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set your API key first:  export ANTHROPIC_API_KEY=sk-ant-...")
        return

    client = Anthropic()

    detector = EmotionDetector()
    cam_thread = threading.Thread(target=detector.run, daemon=True)
    cam_thread.start()
    time.sleep(1.5)   # let the camera warm up and mood stabilize

    print("Emotion-Aware Assistant ready. Look at your camera and chat.")
    print("Type 'quit' to exit.\n")

    history = []
    try:
        while True:
            mood = detector.mood
            user = input(f"[you · looks {mood}] > ").strip()
            if not user:
                continue
            if user.lower() in {"quit", "exit"}:
                break

            history.append({"role": "user", "content": user})
            try:
                resp = client.messages.create(
                    model=MODEL,
                    max_tokens=500,
                    system=build_system_prompt(mood),
                    messages=history,
                )
                reply = "".join(
                    b.text for b in resp.content if b.type == "text")
            except Exception as exc:
                print(f"[api error] {exc}\n")
                history.pop()          # drop the unanswered turn
                continue

            history.append({"role": "assistant", "content": reply})
            print(f"\nClaude: {reply}\n")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        detector.stop()
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
