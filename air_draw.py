
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os
from datetime import datetime

# ==========================================
# Hand Tracking Module (Updated for MediaPipe Tasks)
# ==========================================
class HandDetector:
    """
    Handles hand detection and gesture recognition using MediaPipe Tasks.
    """
    def __init__(self, model_path='hand_landmarker.task'):
        # Initialize the Hand Landmarker
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
        self.tip_ids = [4, 8, 12, 16, 20] # Thumb, Index, Middle, Ring, Pinky

    def detect(self, img, timestamp_ms):
        """Processes a frame to detect hand landmarks."""
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.results = self.detector.detect_for_video(mp_image, int(timestamp_ms))
        
        self.lm_list = []
        bbox = []
        
        if self.results.hand_landmarks:
            h, w, c = img.shape
            # We only use the first hand detected
            hand_lms = self.results.hand_landmarks[0]
            
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            
            for id, lm in enumerate(hand_lms):
                px, py = int(lm.x * w), int(lm.y * h)
                self.lm_list.append([id, px, py])
                
                # Bounding Box calculation
                x_min, y_min = min(x_min, px), min(y_min, py)
                x_max, y_max = max(x_max, px), max(y_max, py)
            
            bbox = (x_min - 20, y_min - 20, x_max + 20, y_max + 20)
            
        return self.lm_list, bbox

    def fingers_up(self):
        """Returns a list of 5 integers (0 or 1) representing if each finger is up."""
        fingers = []
        if not self.lm_list:
            return [0, 0, 0, 0, 0]
            
        # Thumb: Check distance between tip and palm-side joint
        # More robust: check if tip is further from pinky than the base of the thumb
        # In a mirrored view, we can check relative X position
        if self.lm_list[4][1] < self.lm_list[2][1]:
            fingers.append(1)
        else:
            fingers.append(0)
            
        # 4 Fingers: Check if tip (id) is above the middle joint (id-2)
        # Tip IDs: 8, 12, 16, 20. Joints: 6, 10, 14, 18
        for id in range(1, 5):
            if self.lm_list[self.tip_ids[id]][2] < self.lm_list[self.tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

# ==========================================
# Air Drawing Application
# ==========================================
class AirDrawingSystem:
    def __init__(self):
        self.width, self.height = 1280, 720
        self.header_height = 120
        
        # Updated tools: Remove Eraser button (index 3)
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        self.color_names = ["BLUE", "GREEN", "RED", "SAVE"]
        self.draw_color = self.colors[2] # Default Red
        self.brush_thickness = 15
        
        self.canvas = np.zeros((self.height, self.width, 3), np.uint8)
        self.prev_x, self.prev_y = 0, 0
        self.save_cooldown = 0
        
        self.fps = 0
        self.prev_time = 0
        self.clear_counter = 0 
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("[WARNING] Could not open camera 0, trying camera 1...")
            self.cap = cv2.VideoCapture(1)
            
        if not self.cap.isOpened():
            print("[ERROR] No camera found at index 0 or 1. Please check connection.")
            return

        self.cap.set(3, self.width)
        self.cap.set(4, self.height)
        
        print(f"[INFO] Camera initialized successfully ({int(self.cap.get(3))}x{int(self.cap.get(4))})")
        print("[INFO] Loading Hand Landmarker model...")
        self.detector = HandDetector()
        print("[INFO] System Ready. Starting main loop...")

    def draw_ui(self, img):
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.header_height), (30, 30, 30), cv2.FILLED)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        num_btns = len(self.colors)
        w_btn = self.width // num_btns
        for i, color in enumerate(self.colors):
            x1, x2 = i * w_btn + 10, (i + 1) * w_btn - 10
            y1, y2 = 15, self.header_height - 15
            
            if i == 3: # Save button
                cv2.rectangle(img, (x1, y1), (x2, y2), (100, 100, 100), cv2.FILLED)
                cv2.putText(img, "SAVE", (x1+w_btn//4, y2-40), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, cv2.FILLED)
                cv2.putText(img, self.color_names[i], (x1+w_btn//4, y2-40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 2)
            
            # Selection Highlight
            if self.draw_color == color and i < 3:
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 4)

    def save_canvas(self):
        if not os.path.exists("drawings"): os.makedirs("drawings")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.abspath(os.path.join("drawings", f"air_draw_{ts}.png"))
        cv2.imwrite(path, self.canvas)
        print(f"[SUCCESS] Drawing saved to: {path}")
        return path

    def run(self):
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            print("[ERROR] Cannot run system without camera access.")
            return
            
        start_time = time.time()
        cv2.namedWindow("Air Drawing Pro", cv2.WINDOW_AUTOSIZE)
        cv2.setWindowProperty("Air Drawing Pro", cv2.WND_PROP_TOPMOST, 1)
        
        while True:
            success, img = self.cap.read()
            if not success: break
            img = cv2.flip(img, 1)
            
            if self.save_cooldown > 0:
                self.save_cooldown -= 1
                cv2.putText(img, "SAVED!", (self.width//2 - 50, self.header_height + 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            timestamp_ms = int((time.time() - start_time) * 1000)
            lm_list, bbox = self.detector.detect(img, timestamp_ms)
            
            if lm_list:
                x1, y1 = lm_list[8][1:]   # Index
                x2, y2 = lm_list[12][1:]  # Middle
                fingers = self.detector.fingers_up()
                
                # 1. ERASE MODE (Palm - requires 4 main fingers up)
                # Check this FIRST so it's not overridden by Selection mode
                if fingers[1] and fingers[2] and fingers[3] and fingers[4]:
                    self.prev_x, self.prev_y = 0, 0
                    self.clear_counter += 1
                    cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
                    prog = min(self.clear_counter / 15.0, 1.0) # Faster (0.5s)
                    cv2.circle(img, (cx, cy), 65, (0, 0, 255), 3)
                    cv2.putText(img, "KEEP HOLDING TO CLEAR", (cx - 100, cy - 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    
                    if self.clear_counter >= 15:
                        self.canvas = np.zeros((self.height, self.width, 3), np.uint8)
                        self.clear_counter = 0
                        self.save_cooldown = 20
                
                # 2. SELECTION MODE (Index + Middle)
                elif fingers[1] and fingers[2]:
                    self.prev_x, self.prev_y = 0, 0
                    self.clear_counter = 0
                    cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), self.draw_color, cv2.FILLED)
                    
                    if y1 < self.header_height:
                        num_btns = len(self.colors)
                        w_btn = self.width // num_btns
                        idx = x1 // w_btn
                        if idx < 3:
                            self.draw_color = self.colors[idx]
                        elif idx == 3 and self.save_cooldown == 0:
                            self.save_canvas()
                            self.save_cooldown = 30
                
                # 3. DRAWING MODE (Only Index)
                elif fingers[1]:
                    self.clear_counter = 0
                    cv2.circle(img, (x1, y1), self.brush_thickness // 2, self.draw_color, cv2.FILLED)
                    if self.prev_x == 0 and self.prev_y == 0: self.prev_x, self.prev_y = x1, y1
                    cv2.line(self.canvas, (self.prev_x, self.prev_y), (x1, y1), self.draw_color, self.brush_thickness)
                    self.prev_x, self.prev_y = x1, y1
                
                else: 
                    self.prev_x, self.prev_y = 0, 0
                    self.clear_counter = 0

            # UI and Merge
            img_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, img_inv = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY_INV)
            img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
            img = cv2.bitwise_and(img, img_inv)
            img = cv2.bitwise_or(img, self.canvas)
            self.draw_ui(img)
            
            # Show finger status for debugging clear screen
            if lm_list:
                cv2.putText(img, f"Fingers: {fingers}", (20, self.height - 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # FPS
            curr_tp = time.time()
            self.fps = 1 / (curr_tp - self.prev_time)
            self.prev_time = curr_tp
            cv2.putText(img, f"FPS: {int(self.fps)}", (self.width - 150, self.height - 20), 0, 0.8, (0, 255, 255), 2)
            
            cv2.imshow("Air Drawing Pro", img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break
            elif key == ord('s'): self.save_canvas()
            elif key == ord('c'): self.canvas = np.zeros((self.height, self.width, 3), np.uint8)

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = AirDrawingSystem()
    app.run()
