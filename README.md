# Air Drawing System 🎨

An Augmented Reality (AR) application that allows you to draw in the air using hand gestures. The system uses your webcam to track your hand movements and translates them into digital art on a virtual canvas in real-time.

---

## 🚀 Features

- **Gesture-Based Drawing**: No mouse or keyboard needed.
- **Color Palette**: Choose between Red, Green, and Blue.
- **Intelligent Modes**: Switch between drawing and navigation automatically based on your hand posture.
- **Full Canvas Erase**: Clear the entire screen with a simple palm gesture.
- **Save Work**: Export your drawings as `.png` files directly to your machine.
- **Real-time Performance**: Optimized for smooth tracking and low latency.

---

## 🛠️ Tech Stack

- **Python**: Core programming language.
- **OpenCV**: Image processing and video capture.
- **MediaPipe**: Hand tracking and landmark detection (using Google AI).
- **NumPy**: Canvas management and mathematical operations.

---

## 🎮 How to Use

| Action | Gesture | Description |
| :--- | :--- | :--- |
| **Draw** | ☝️ Index finger up | Draw on the screen with the selected color. |
| **Select/Move** | ✌️ Index + Middle up | Move over the top menu to change colors or save. |
| **Clear Screen** | 🖐️ Full Palm open | Hold for 0.5s to erase everything. |
| **Quick Actions** | Keyboard shortcut | Press **'s'** to Save or **'c'** to Clear. |

---

## 📦 Installation

1. **Clone or Download** this repository.
2. **Install Dependencies**:
   ```bash
   pip install opencv-python mediapipe numpy
   ```
3. **Ensure Model File**: Make sure `hand_landmarker.task` is in the project root.
4. **Run the Application**:
   ```bash
   python air_draw.py
   ```

---

## 📂 Project Structure

- `air_draw.py`: The main application script.
- `hand_landmarker.task`: The MediaPipe AI model file.
- `drawings/`: A folder where your saved masterpieces are stored.
- `README.md`: This file!

---

## 📝 License

This project is open-source. Feel free to modify and build upon it!
