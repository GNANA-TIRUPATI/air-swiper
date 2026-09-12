# 🖐️ Air Swiper

> **Scroll hands-free through YouTube Shorts, Instagram Reels, TikTok, documents, and web pages using intuitive air gestures and computer vision.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks%20Vision-orange.svg)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌟 Overview

**Air Swiper** turns your webcam into a touchless controller. Using Google's **MediaPipe Hand Landmarker Task API**, it tracks hand movements in real-time with sub-millisecond responsiveness. Swipe your hand in the air to scroll through short-form video feeds without touching your keyboard, mouse, or screen.

---

## ✨ Features

- 🔒 **Accidental Trigger Protection**: Requires a 0.7-second **Right Palm Hold** to activate the session, ensuring everyday movements and typing don't trigger unwanted scrolling.
- 🎯 **Palm-Center Tracking**: Calculates a stabilized palm centroid using 5 key palm landmarks (wrist + metacarpophalangeal joints) rather than erratic fingertip tracking.
- ⚡ **Dynamic Velocity & Ratio Filters**:
  - **Speed threshold**: Filters out slow, casual hand repositioning.
  - **Vertical dominance ratio**: Ensures vertical motions are at least 1.25× stronger than horizontal drift.
  - **Cooldown buffer**: Configurable debounce timer (0.55s) to avoid multi-triggering from a single gesture.
- 🖐️ **Physical Hand Orientation Aware**: Automatically accounts for webcam mirroring, isolating controls to your physical right hand and rejecting left-hand inputs.
- 🖥️ **Live Diagnostic HUD**: On-screen overlay displaying activation status, real-time displacement (`dy`), velocity (`speed`), threshold limits, action logs, and a full hand skeleton visualization.

---

## 🎮 Gestures & Controls

| Gesture | Movement | Action |
| :--- | :--- | :--- |
| **Open Right Palm** | Hold open facing camera for **0.7s** | **Activate** gesture session |
| **Swipe Down** | Swift flick downward (physical right hand) | **Next Short / Reel** (`Scroll Down`) |
| **Swipe Up** | Swift flick upward (physical right hand) | **Previous Short / Reel** (`Scroll Up`) |
| **Left Hand** | Any movement | **Ignored** (Prompts right hand) |
| <kbd>ESC</kbd> | Press on keyboard | **Exit** application |

---

## 📁 Repository Structure

```text
air-swiper/
├── hand_landmarker.task   # MediaPipe pretrained Hand Landmarker model
├── v1.py                  # Main application script & gesture controller
├── requirements.txt       # Python dependencies
├── .gitignore             # Standard git ignore file
└── README.md              # Documentation
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/GNANA-TIRUPATI/air-swiper.git
cd air-swiper
```

### 2. Set Up Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python v1.py
```

---

## ⚙️ Configuration

You can easily fine-tune gesture sensitivity by editing the constants near the top of [`v1.py`](v1.py):

| Variable | Default | Description |
| :--- | :---: | :--- |
| `CAMERA_INDEX` | `0` | Camera device index (change to `1`, `2` for external webcams). |
| `ACTIVATION_HOLD_TIME` | `0.7` | Seconds to hold right palm to activate. |
| `SWIPE_DISTANCE` | `0.035` | Minimum vertical movement required. |
| `SWIPE_TIME` | `0.20` | Movement detection time window in seconds. |
| `MIN_SWIPE_SPEED` | `0.30` | Speed threshold to distinguish intentional flicks from casual drift. |
| `MIN_VERTICAL_RATIO`| `1.25` | Ratio of vertical to horizontal displacement required. |
| `SWIPE_COOLDOWN` | `0.55` | Debounce time (in seconds) between consecutive swipes. |

---

## 🛠️ Tech Stack

- **[Python](https://www.python.org/)** (3.8+)
- **[Google MediaPipe](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)** - 21 3D hand landmarks tracking using vision task bundles
- **[OpenCV](https://opencv.org/)** - Real-time camera feed capture, frame mirroring, and HUD rendering
- **[PyAutoGUI](https://pyautogui.readthedocs.io/)** - Programmatic native OS mouse scrolling

---

## 💡 Usage Tips

1. **Window Focus**: Make sure your browser tab (YouTube Shorts, Instagram Reels, TikTok) or reader application is active in the background.
2. **Lighting**: Ensure good front lighting so your hand is clearly visible against the background.
3. **Camera Placement**: Position your webcam so your upper chest and hand can be comfortably held in view without straining.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) - feel free to use, modify, and contribute!
