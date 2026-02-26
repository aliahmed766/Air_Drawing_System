import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os
from datetime import datetime

# ==========================================
# Optimized Hand Detector for Streamlit
# ==========================================
class HandDetector:
    def __init__(self, model_path='hand_landmarker.task'):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            running_mode=vision.RunningMode.VIDEO
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.results = None
        self.lm_list = []
        self.tip_ids = [4, 8, 12, 16, 20]

    def detect(self, img, timestamp_ms):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.results = self.detector.detect_for_video(mp_image, int(timestamp_ms))
        self.lm_list = []
        if self.results.hand_landmarks:
            h, w, c = img.shape
            hand_lms = self.results.hand_landmarks[0]
            for id, lm in enumerate(hand_lms):
                px, py = int(lm.x * w), int(lm.y * h)
                self.lm_list.append([id, px, py])
        return self.lm_list

    def fingers_up(self):
        if not self.lm_list: return [0, 0, 0, 0, 0]
        fingers = []
        # Thumb
        if self.lm_list[4][1] < self.lm_list[2][1]: fingers.append(1)
        else: fingers.append(0)
        # Fingers
        for id in range(1, 5):
            if self.lm_list[self.tip_ids[id]][2] < self.lm_list[self.tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

# ==========================================
# Video Processor for WebRTC
# ==========================================
class AirDrawingProcessor(VideoTransformerBase):
    def __init__(self):
        self.detector = HandDetector()
        self.canvas = None
        self.prev_x, self.prev_y = 0, 0
        self.start_time = time.time()
        
        # Settings (will be updated from Streamlit UI)
        self.brush_color = (0, 0, 255)
        self.brush_thickness = 15
        self.clear_request = False

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        if self.canvas is None:
            self.canvas = np.zeros((h, w, 3), np.uint8)

        if self.clear_request:
            self.canvas = np.zeros((h, w, 3), np.uint8)
            self.clear_request = False

        timestamp_ms = int((time.time() - self.start_time) * 1000)
        lm_list = self.detector.detect(img, timestamp_ms)

        if lm_list:
            fingers = self.detector.fingers_up()
            x1, y1 = lm_list[8][1:] # Index Tip

            # Drawing Mode (Only index finger up)
            if fingers[1] and not fingers[2]:
                cv2.circle(img, (x1, y1), self.brush_thickness // 2, self.brush_color, cv2.FILLED)
                if self.prev_x == 0 and self.prev_y == 0:
                    self.prev_x, self.prev_y = x1, y1
                cv2.line(self.canvas, (self.prev_x, self.prev_y), (x1, y1), self.brush_color, self.brush_thickness)
                self.prev_x, self.prev_y = x1, y1
            else:
                self.prev_x, self.prev_y = 0, 0

        # Merge Canvas and Image
        img_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, img_inv)
        img = cv2.bitwise_or(img, self.canvas)

        return img

# ==========================================
# Streamlit UI
# ==========================================
def main():
    st.set_page_config(page_title="Air Drawing Pro Web", layout="wide")
    
    st.title("🎨 Air Drawing Pro - Web Edition")
    st.markdown("Draw in the air using your webcam! Use your **Index Finger** to draw.")

    sidebar = st.sidebar
    sidebar.title("Settings")
    
    color_options = {
        "Red": (0, 0, 255),
        "Green": (0, 255, 0),
        "Blue": (255, 0, 0),
        "Yellow": (0, 255, 255),
        "White": (255, 255, 255)
    }
    
    selected_color_name = sidebar.selectbox("Select Color", list(color_options.keys()), index=0)
    brush_color = color_options[selected_color_name]
    
    brush_thickness = sidebar.slider("Brush Thickness", 5, 50, 15)
    
    if sidebar.button("Clear Canvas"):
        st.session_state["clear_flag"] = True

    # WebRTC Streamer
    ctx = webrtc_streamer(
        key="air-draw",
        video_transformer_factory=AirDrawingProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
    )

    if ctx.video_transformer:
        ctx.video_transformer.brush_color = brush_color
        ctx.video_transformer.brush_thickness = brush_thickness
        if st.session_state.get("clear_flag", False):
            ctx.video_transformer.clear_request = True
            st.session_state["clear_flag"] = False

    st.info("💡 **Tip:** Only raise your index finger to draw. Raise multiple fingers to move without drawing.")

if __name__ == "__main__":
    main()
