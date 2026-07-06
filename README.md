# Autonomous Driving Perception Engine


The Autonomous Driving Perception Engine processes road videos frame-by-frame and extracts meaningful environmental information required for autonomous navigation.

The pipeline detects and tracks dynamic road participants, estimates scene depth from monocular images, predicts future trajectories, identifies hazardous regions, and presents all information through an interactive driving intelligence dashboard.

This project is designed for:

- Autonomous Driving Research
- Advanced Driver Assistance Systems (ADAS)
- Intelligent Transportation Systems
- Robotics & Computer Vision
- AI Perception Studies

---

"""
AirControl - Gesture-Controlled Desktop Automation

Control media playback and volume with hand gestures using your webcam.

Vision (MediaPipe) -> Intent (finger counting) -> Action (pyautogui).
This "perceive-decide-act" loop is the same skeleton used by AI agents.

Gestures (number of fingers held up):
    0 (fist)          -> Mute / unmute
    1 (index)         -> Volume up
    2 (peace)         -> Volume down
    3               -> Next track
    4               -> Previous track
    5 (open palm)     -> Play / pause

Keys:
    a  -> Arm / disarm automation (starts DISARMED so you can test safely)
    q  -> Quit

Setup:
    pip install mediapipe opencv-python pyautogui
    python aircontrol.py

"""


"""
Emotion-Aware AI Assistant
==========================
Your webcam reads your facial expression from MediaPipe Face Mesh landmarks,
infers a coarse mood, and an LLM (Claude) adapts its tone to how you look.
Perceive (vision) -> infer (mood) -> reason (LLM) -> respond: a small
multimodal agent loop.

How it runs:
  - A background thread watches your face and keeps a smoothed current mood.
  - You chat in the terminal. Each reply is tuned to your detected mood.

Detected moods (heuristic, from geometry — see notes below):
  Happy, Surprised, Sad, Neutral

Setup:
  pip install mediapipe opencv-python numpy anthropic
  export ANTHROPIC_API_KEY="sk-ant-..."     # (Windows: set ANTHROPIC_API_KEY=...)
  python emotion_assistant.py

Type 'quit' to exit.


"""
