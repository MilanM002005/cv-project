
import time
import csv
from collections import defaultdict

import cv2
import numpy as np
import mediapipe as mp

import matplotlib
matplotlib.use("Agg")            # save charts without needing a display
import matplotlib.pyplot as plt


# ------------------------- Tunable thresholds -----------------------------
EAR_CLOSED = 0.20        # eye-aspect-ratio below this = eye considered closed
DROWSY_SEC = 1.2         # eyes closed longer than this = DROWSY (not a blink)
YAW_THRESH = 16          # |yaw| under this degrees = facing forward
PITCH_THRESH = 14        # |pitch| under this degrees = facing forward
# --------------------------------------------------------------------------

# Face Mesh landmark indices
RIGHT_EYE = [33, 160, 158, 133, 153, 144]     # for EAR (person's right eye)
LEFT_EYE = [362, 385, 387, 263, 373, 380]     # for EAR (person's left eye)
POSE_IDS = [33, 263, 1, 61, 291, 199]         # for head-pose solvePnP

STATE_COLORS = {
    "FOCUSED": "#2ecc71",
    "DISTRACTED": "#e67e22",
    "DROWSY": "#9b59b6",
    "AWAY": "#95a5a6",
}


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(landmarks_px, eye_ids):
    """EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|). Low value = closed eye."""
    p1, p2, p3, p4, p5, p6 = [landmarks_px[i] for i in eye_ids]
    vertical = euclidean(p2, p6) + euclidean(p3, p5)
    horizontal = 2.0 * euclidean(p1, p4)
    return vertical / horizontal if horizontal else 0.0


def head_pose_angles(landmarks_norm, w, h):
    """Estimate (pitch, yaw, roll) in degrees via solvePnP.

    Uses the landmark's own 3D position as the model — a simple, widely used
    technique that is robust for the 'facing forward vs away' decision.
    """
    face_2d, face_3d = [], []
    for idx in POSE_IDS:
        lm = landmarks_norm[idx]
        x, y = lm.x * w, lm.y * h
        face_2d.append([x, y])
        face_3d.append([x, y, lm.z])

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    focal = w
    cam_matrix = np.array([[focal, 0, w / 2],
                           [0, focal, h / 2],
                           [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)

    ok, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist)
    if not ok:
        return None
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    pitch, yaw, roll = angles[0] * 360, angles[1] * 360, angles[2] * 360
    return pitch, yaw, roll


def classify(landmarks_norm, w, h, eyes_closed_since):
    """Return (state, ear, angles) for the current frame."""
    landmarks_px = [(lm.x * w, lm.y * h) for lm in landmarks_norm]

    ear = (eye_aspect_ratio(landmarks_px, RIGHT_EYE) +
           eye_aspect_ratio(landmarks_px, LEFT_EYE)) / 2.0

    angles = head_pose_angles(landmarks_norm, w, h)
    if angles is None:
        return "DISTRACTED", ear, None
    pitch, yaw, roll = angles

    facing_forward = abs(yaw) < YAW_THRESH and abs(pitch) < PITCH_THRESH

    if not facing_forward:
        return "DISTRACTED", ear, angles

    # Eyes closed: brief = blink (still FOCUSED); sustained = DROWSY.
    if ear < EAR_CLOSED and eyes_closed_since is not None:
        if time.time() - eyes_closed_since > DROWSY_SEC:
            return "DROWSY", ear, angles

    return "FOCUSED", ear, angles


def draw_hud(frame, state, focus_pct, elapsed, blinks, ear):
    h, w = frame.shape[:2]
    hexcol = STATE_COLORS[state].lstrip("#")
    bgr = tuple(int(hexcol[i:i + 2], 16) for i in (4, 2, 0))

    cv2.rectangle(frame, (0, 0), (w, 78), (20, 20, 20), -1)
    cv2.putText(frame, f"STATE: {state}", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr, 2)
    mins, secs = divmod(int(elapsed), 60)
    cv2.putText(frame,
                f"Focus: {focus_pct:4.1f}%   Time: {mins:02d}:{secs:02d}"
                f"   Blinks: {blinks}   EAR: {ear:.2f}",
                (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)

    # focus bar
    bar_w = int((w - 24) * min(focus_pct, 100) / 100)
    cv2.rectangle(frame, (12, h - 22), (w - 12, h - 8), (60, 60, 60), -1)
    cv2.rectangle(frame, (12, h - 22), (12 + bar_w, h - 8), (46, 204, 113), -1)
    return frame


def save_report(samples, state_seconds, blinks, total_time):
    ts = time.strftime("%Y%m%d_%H%M%S")
    focus_pct = 100.0 * state_seconds["FOCUSED"] / total_time if total_time else 0

    # --- CSV log ---
    csv_name = f"focus_log_{ts}.csv"
    with open(csv_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["elapsed_sec", "state"])
        writer.writerows(samples)

    # --- PNG chart ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: focus ribbon over time
    if samples:
        times = [s[0] / 60.0 for s in samples]           # minutes
        colors = [STATE_COLORS[s[1]] for s in samples]
        ax1.scatter(times, [1] * len(times), c=colors, marker="s", s=80)
    ax1.set_yticks([])
    ax1.set_xlabel("Time (minutes)")
    ax1.set_title("Attention timeline")
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", color=c,
                          label=st) for st, c in STATE_COLORS.items()]
    ax1.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False)

    # Right: time spent per state
    states = list(STATE_COLORS.keys())
    minutes = [state_seconds[s] / 60.0 for s in states]
    ax2.bar(states, minutes, color=[STATE_COLORS[s] for s in states])
    ax2.set_ylabel("Minutes")
    ax2.set_title("Time per state")

    fig.suptitle(f"Study Session Report  —  Focus {focus_pct:.1f}%   "
                 f"Duration {total_time/60:.1f} min   Blinks {blinks}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    png_name = f"focus_report_{ts}.png"
    fig.savefig(png_name, dpi=120)
    plt.close(fig)
    return csv_name, png_name, focus_pct


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    state_seconds = defaultdict(float)
    samples = []                 # (elapsed_sec, state) sampled ~1/sec
    blinks = 0
    eyes_closed = False
    eyes_closed_since = None

    start = time.time()
    last_frame = start
    last_sample_sec = -1

    print("Focus Analytics running. Press 'q' to stop and get your report.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        now = time.time()
        dt = now - last_frame
        last_frame = now
        elapsed = now - start

        ear = 0.0
        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0].landmark

            # blink bookkeeping (needs current EAR first)
            lm_px = [(p.x * w, p.y * h) for p in lms]
            ear = (eye_aspect_ratio(lm_px, RIGHT_EYE) +
                   eye_aspect_ratio(lm_px, LEFT_EYE)) / 2.0

            if ear < EAR_CLOSED and not eyes_closed:
                eyes_closed = True
                eyes_closed_since = now
            elif ear >= EAR_CLOSED and eyes_closed:
                eyes_closed = False
                eyes_closed_since = None
                blinks += 1

            state, ear, _ = classify(lms, w, h, eyes_closed_since)
        else:
            state = "AWAY"
            eyes_closed = False
            eyes_closed_since = None

        state_seconds[state] += dt

        # sample once per second for the timeline/CSV
        if int(elapsed) != last_sample_sec:
            last_sample_sec = int(elapsed)
            samples.append((round(elapsed, 1), state))

        total = max(elapsed, 1e-6)
        focus_pct = 100.0 * state_seconds["FOCUSED"] / total

        frame = draw_hud(frame, state, focus_pct, elapsed, blinks, ear)
        cv2.imshow("Focus Analytics", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    total_time = time.time() - start
    if total_time < 1:
        print("Session too short to report.")
        return

    csv_name, png_name, focus_pct = save_report(
        samples, state_seconds, blinks, total_time)

    print("\n================ SESSION REPORT ================")
    print(f"Duration      : {total_time/60:.1f} min")
    print(f"Focus score   : {focus_pct:.1f}%")
    print(f"Blinks        : {blinks}")
    for st in STATE_COLORS:
        print(f"  {st:<11}: {state_seconds[st]/60:5.1f} min")
    print(f"Chart saved   : {png_name}")
    print(f"CSV log saved : {csv_name}")
    print("================================================")


if __name__ == "__main__":
    main()
