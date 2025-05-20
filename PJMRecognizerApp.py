# -*- coding: utf-8 -*-
# gesture_recognition_app.py

import sys
import os
import csv
import time # Import time module
import pickle
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
# Import QKeySequence needed for displaying the key name in labels
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QStatusBar, QGridLayout,
    QScrollArea, QComboBox, QTextEdit, QTabWidget # Added QTabWidget
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QColor # Added QPainter etc.
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPoint # Added QPoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- Constants ---
APP_NAME = "Gesture Recognition App (Tabbed)"
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 850
CAMERA_FEED_WIDTH = 640
CAMERA_FEED_HEIGHT = 480
STATIC_CONFIDENCE_THRESHOLD = 0.90 # Threshold for triggering path recording
STATIC_HOLD_DURATION_SEC = 1.0 # Hold static gesture for 1 sec
PATH_MAX_DURATION_SEC = 3.0 # Max path recording time

# Static Gesture Constants
STATIC_CSV_FILE = 'gesture_data.csv'
STATIC_MODEL_FILE = 'gesture_model.h5'
STATIC_ENCODER_FILE = 'label_encoder.pkl'
STATIC_NUM_LANDMARKS = 21
STATIC_NUM_FEATURES = STATIC_NUM_LANDMARKS * 3 # x, y, z
STATIC_ALLOWED_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'W', 'Y', 'Z']
STATIC_NUM_CLASSES = len(STATIC_ALLOWED_LABELS)

# Path Gesture Constants
PATH_CSV_FILE = 'path_data.csv'
PATH_MODEL_FILE = 'path_model.h5'
PATH_ENCODER_FILE = 'path_label_encoder.pkl'
PATH_SCALER_FILE = 'path_scaler.pkl'
PATH_ALLOWED_LABELS = ['J', 'left', 'right', 'down', 'tail', 'zigzagl', 'zigzagr', 'zkropka']
PATH_NUM_CLASSES = len(PATH_ALLOWED_LABELS)
PATH_LENGTH = 100
PATH_NUM_FEATURES = PATH_LENGTH * 2 # x, y
PATH_TRACKING_LANDMARK = 8 # Index finger tip
PATH_RECORD_KEY = Qt.Key_Q # Key to press for path collection/classification modes

# Combined Mode Rules
COMBINATION_RULES = {
    ('A', 'tail'): 'Ą', ('C', 'down'): 'Ć', ('I', 'J'): 'J', ('N', 'down'): 'Ń',
    ('O', 'down'): 'Ó', ('R', 'zigzagl'): 'RZ', ('S', 'down'): 'Ś', ('L', 'right'): 'Ł',
    ('B', 'left'): 'SZ', ('Z', 'zigzagl'): 'Ź', ('Z', 'zkropka'): 'Ż',
}

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1,
    min_detection_confidence=0.7, min_tracking_confidence=0.5, model_complexity=1
)

# --- Helper Functions ---

def process_static_landmarks(hand_landmarks):
    """Processes landmarks for static gestures."""
    if not hand_landmarks: return None
    lm_list = []
    base_x, base_y, base_z = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z
    for lm in hand_landmarks.landmark: lm_list.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
    return lm_list if len(lm_list) == STATIC_NUM_FEATURES else None

def save_static_to_csv(fp, lbl, lms):
    """Saves static gesture data."""
    exists = os.path.isfile(fp)
    header = ['label'] + [f'{ax}{i}' for i in range(STATIC_NUM_LANDMARKS) for ax in 'xyz']
    try:
        with open(fp, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if not exists or os.path.getsize(fp) == 0: w.writerow(header)
            if lms and len(lms) == STATIC_NUM_FEATURES: w.writerow([lbl] + lms); return True
            else: return False
    except IOError as e: print(f"Err static CSV: {e}"); return False

def standardize_path(pts, length=PATH_LENGTH):
    """Standardizes a path to a fixed number of points using interpolation."""
    if len(pts) < 2: return None
    pts_np = np.array(pts)
    dist = np.cumsum(np.sqrt(np.sum(np.diff(pts_np, axis=0)**2, axis=1)))
    dist = np.insert(dist, 0, 0)
    if dist[-1] == 0: return np.array([pts_np[0]] * length).flatten()
    alpha = np.linspace(0, dist[-1], length)
    interp_x = interp1d(dist, pts_np[:, 0], kind='linear', bounds_error=False, fill_value="extrapolate")
    interp_y = interp1d(dist, pts_np[:, 1], kind='linear', bounds_error=False, fill_value="extrapolate")
    return np.vstack((interp_x(alpha), interp_y(alpha))).T.flatten()

def save_path_to_csv(fp, lbl, coords):
    """Saves standardized path data."""
    exists = os.path.isfile(fp)
    header = ['label'] + [f'{ax}{i}' for i in range(PATH_LENGTH) for ax in 'xy']
    try:
        with open(fp, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if not exists or os.path.getsize(fp) == 0: w.writerow(header)
            if coords is not None and len(coords) == PATH_NUM_FEATURES: w.writerow([lbl] + coords.tolist()); return True
            else: print(f"Warn: Invalid path data {lbl}. Len: {len(coords) if coords is not None else 'None'}!={PATH_NUM_FEATURES}"); return False
    except IOError as e: print(f"Err path CSV: {e}"); return False

# Moved outside classes <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def _convert_cv_qt(cv_img):
    """Converts OpenCV image (BGR) to QImage (RGB888)."""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return convert_to_Qt_format
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# --- PyQt Threads ---

class CameraThread(QThread):
    """Handles camera capture and processing in a separate thread."""
    frame_signal = pyqtSignal(QImage)
    static_landmarks_signal = pyqtSignal(list)
    path_point_signal = pyqtSignal(tuple)
    combined_path_ready_signal = pyqtSignal(list, str)
    path_recording_started_signal = pyqtSignal() # Signal drawing start
    path_recording_stopped_signal = pyqtSignal() # Signal drawing stop
    status_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, mode='idle', parent=None):
        super().__init__(parent)
        self.running = False
        self.mode = mode
        self.current_static_label = None
        self.static_model = None
        self.static_label_encoder = None
        self.path_model = None
        self.path_label_encoder = None
        self.path_scaler = None
        self.is_recording_path = False # Internal flag for thread logic
        self.combined_trigger_label = None
        self.current_path_points = []
        # Timing state variables
        self.high_conf_start_time = None
        self.high_conf_label = None
        self.path_record_start_time = None

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): self.error_signal.emit("Error: Could not open camera."); self.running = False; return
        self.status_signal.emit(f"Camera started in {self.mode} mode.")

        while self.running and cap.isOpened():
            current_time = time.time() # Get current time at start of loop
            ret, frame = cap.read()
            if not ret: time.sleep(0.1); continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = hands.process(rgb_frame)
            frame.flags.writeable = True

            static_landmarks_data = None
            path_point_data = None
            static_pred_label = None
            static_pred_conf = 0.0
            display_prediction_text = ""
            hand_detected = results.multi_hand_landmarks is not None

            if hand_detected:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing_styles.get_default_hand_landmarks_style(),
                                          mp_drawing_styles.get_default_hand_connections_style())
                processed_static_landmarks = process_static_landmarks(hand_landmarks)

                # --- Static Classification (always run if model loaded and hand detected) ---
                if self.mode in ['classify_static', 'combined'] and processed_static_landmarks and self.static_model and self.static_label_encoder:
                    try:
                        input_data = np.array([processed_static_landmarks], dtype=np.float32)
                        prediction = self.static_model.predict(input_data, verbose=0)[0]
                        idx = np.argmax(prediction)
                        static_pred_conf = prediction[idx]
                        static_pred_label = self.static_label_encoder.inverse_transform([idx])[0]
                        if self.mode == 'classify_static': display_prediction_text = f"Static: {static_pred_label} ({static_pred_conf:.2f})"
                    except Exception as e: self.status_signal.emit(f"Static Classif. Error: {e}"); static_pred_label = None; static_pred_conf = 0.0

                # --- Path Point Tracking (always run if needed and hand detected) ---
                if self.mode in ['collect_path', 'classify_path', 'combined']:
                    lm = hand_landmarks.landmark[PATH_TRACKING_LANDMARK]
                    path_point_data = (lm.x, lm.y)

                # --- Mode-Specific Logic ---
                # 1. Static Collection
                if self.mode == 'collect_static' and self.current_static_label and processed_static_landmarks:
                    self.static_landmarks_signal.emit([self.current_static_label] + processed_static_landmarks)
                    self.status_signal.emit(f"Captured static '{self.current_static_label}'. Press key or Stop.")
                    self.current_static_label = None

                # 2. Path Collection/Classification
                elif self.mode in ['collect_path', 'classify_path'] and self.is_recording_path and path_point_data:
                    self.path_point_signal.emit(path_point_data) # Emit for drawing

                # 3. Combined Mode Logic with Time Constraints
                elif self.mode == 'combined':
                    # Check static confidence threshold
                    if static_pred_conf >= STATIC_CONFIDENCE_THRESHOLD:
                        # Track how long high confidence has been held for the *same* label
                        if self.high_conf_label != static_pred_label:
                            # Label changed or first time high conf detected
                            self.high_conf_label = static_pred_label
                            self.high_conf_start_time = current_time
                            # If recording was active for a different label, stop it
                            if self.is_recording_path:
                                self.status_signal.emit(f"Static label changed mid-hold. Stopping path for '{self.combined_trigger_label}'.")
                                if len(self.current_path_points) > 1: self.combined_path_ready_signal.emit(list(self.current_path_points), self.combined_trigger_label)
                                self.is_recording_path = False
                                self.current_path_points = []
                                self.combined_trigger_label = None
                                self.path_record_start_time = None
                                self.path_recording_stopped_signal.emit() # Signal stop for drawing
                        # else: label is the same, continue checking hold duration

                        # Check if static hold duration is met AND not already recording path
                        if not self.is_recording_path and self.high_conf_label == static_pred_label and \
                           self.high_conf_start_time is not None and \
                           (current_time - self.high_conf_start_time >= STATIC_HOLD_DURATION_SEC):
                            # Start recording path
                            self.is_recording_path = True
                            self.combined_trigger_label = static_pred_label # The label that met the hold time
                            self.current_path_points = []
                            self.path_record_start_time = current_time # Record path start time
                            self.status_signal.emit(f"Held '{self.combined_trigger_label}' >= {STATIC_HOLD_DURATION_SEC:.1f}s. Recording path...")
                            self.path_recording_started_signal.emit() # Signal start for drawing

                        # If currently recording path
                        if self.is_recording_path:
                            # Check if path duration exceeded limit
                            if self.path_record_start_time is not None and \
                               (current_time - self.path_record_start_time > PATH_MAX_DURATION_SEC):
                                self.status_signal.emit(f"Path time limit ({PATH_MAX_DURATION_SEC:.1f}s) exceeded for '{self.combined_trigger_label}'. Processing.")
                                if len(self.current_path_points) > 1: self.combined_path_ready_signal.emit(list(self.current_path_points), self.combined_trigger_label)
                                self.is_recording_path = False
                                self.current_path_points = []
                                self.combined_trigger_label = None
                                self.path_record_start_time = None
                                self.high_conf_start_time = None
                                self.high_conf_label = None # Reset all relevant state
                                self.path_recording_stopped_signal.emit() # Signal stop for drawing
                            # Else, if time limit not exceeded, append point if valid
                            elif self.combined_trigger_label == static_pred_label and path_point_data: # Ensure static label hasn't changed while appending
                                self.current_path_points.append(path_point_data)
                                self.path_point_signal.emit(path_point_data) # Emit point for drawing
                            elif self.combined_trigger_label != static_pred_label: # Static label changed during recording
                                 self.status_signal.emit(f"Static label changed during path recording. Stopping path for '{self.combined_trigger_label}'.")
                                 if len(self.current_path_points) > 1: self.combined_path_ready_signal.emit(list(self.current_path_points), self.combined_trigger_label)
                                 self.is_recording_path = False
                                 self.current_path_points = []
                                 self.combined_trigger_label = None
                                 self.path_record_start_time = None
                                 self.high_conf_start_time = None
                                 self.high_conf_label = None # Reset all
                                 self.path_recording_stopped_signal.emit() # Signal stop for drawing

                    else: # Static confidence dropped below threshold
                        # Reset high confidence tracking
                        self.high_conf_start_time = None
                        self.high_conf_label = None
                        # If we were recording, stop and process
                        if self.is_recording_path:
                            self.status_signal.emit(f"Static conf low. Stopping path for '{self.combined_trigger_label}'.")
                            if len(self.current_path_points) > 1: self.combined_path_ready_signal.emit(list(self.current_path_points), self.combined_trigger_label)
                            self.is_recording_path = False
                            self.current_path_points = []
                            self.combined_trigger_label = None
                            self.path_record_start_time = None
                            self.path_recording_stopped_signal.emit() # Signal stop for drawing

            # --- Handle Hand Lost Event ---
            if not hand_detected:
                # Reset high confidence tracking if hand is lost
                self.high_conf_start_time = None
                self.high_conf_label = None
                # Stop recording if active in combined mode
                if self.is_recording_path and self.mode == 'combined':
                    self.status_signal.emit(f"Hand lost. Stopping path for '{self.combined_trigger_label}'.")
                    if len(self.current_path_points) > 1: self.combined_path_ready_signal.emit(list(self.current_path_points), self.combined_trigger_label)
                    self.is_recording_path = False
                    self.current_path_points = []
                    self.combined_trigger_label = None
                    self.path_record_start_time = None
                    self.path_recording_stopped_signal.emit() # Signal stop for drawing

            # --- Display Text & Emit Frame ---
            if display_prediction_text: cv2.putText(frame, display_prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # Call the standalone conversion function <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MODIFIED
            qt_image = _convert_cv_qt(frame)
            self.frame_signal.emit(qt_image)

        cap.release()
        self.status_signal.emit("Camera stopped.")
        self.running = False

    def stop(self): # Modified stop logic
        self.running = False
        if self.is_recording_path and self.mode == 'combined' and self.combined_trigger_label:
             self.status_signal.emit(f"Camera stopping. Processing path for '{self.combined_trigger_label}'.")
             if len(self.current_path_points) > 1: self.combined_path_ready_signal.emit(list(self.current_path_points), self.combined_trigger_label)
             self.path_recording_stopped_signal.emit()
        # Reset all relevant states on stop
        self.is_recording_path = False
        self.current_path_points = []
        self.combined_trigger_label = None
        self.high_conf_start_time = None
        self.high_conf_label = None
        self.path_record_start_time = None
        self.status_signal.emit("Stopping camera...")

    # --- Other Methods ---
    def set_static_label(self, label):
        if self.mode == 'collect_static' and label in STATIC_ALLOWED_LABELS: self.current_static_label = label; self.status_signal.emit(f"Ready for static '{label}'. Show gesture.")
        elif self.mode != 'collect_static': self.status_signal.emit("Not in static data collection mode.")
    def set_path_recording(self, is_recording):
        if self.mode in ['collect_path', 'classify_path']: self.is_recording_path = is_recording
    def load_static_model(self, model, encoder):
        self.static_model = model; self.static_label_encoder = encoder; msg = "Static model loaded." if model and encoder else "Failed to load static model/encoder."
        if model and encoder: self.status_signal.emit(msg)
        else: self.error_signal.emit(msg)
    def load_path_model(self, model, encoder, scaler):
        self.path_model = model; self.path_label_encoder = encoder; self.path_scaler = scaler; msg = "Path model loaded." if model and encoder and scaler else "Failed to load path model/encoder/scaler."
        if model and encoder and scaler: self.status_signal.emit(msg)
        else: self.error_signal.emit(msg)

    # REMOVED _convert_cv_qt from CameraThread

# --- Training Threads ---
class StaticTrainingThread(QThread):
    progress_signal = pyqtSignal(str); finished_signal = pyqtSignal(bool, str)
    def __init__(self, csv_path, model_path, encoder_path, parent=None): super().__init__(parent); self.csv_path = csv_path; self.model_path = model_path; self.encoder_path = encoder_path
    def run(self):
        try:
            self.progress_signal.emit("Starting static model training..."); self.progress_signal.emit(f"Loading static data from {self.csv_path}...")
            if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0: raise FileNotFoundError(f"Static CSV '{self.csv_path}' not found or empty.")
            df = pd.read_csv(self.csv_path); df.dropna(inplace=True); assert not df.empty, "Static CSV is empty after dropping NaNs."
            self.progress_signal.emit(f"Static data loaded: {df.shape[0]} samples."); self.progress_signal.emit("Preprocessing static data...")
            X = df.iloc[:, 1:].values; y_labels = df.iloc[:, 0].values; label_encoder = LabelEncoder(); label_encoder.fit(STATIC_ALLOWED_LABELS)
            try: y_encoded = label_encoder.transform(y_labels)
            except ValueError as e: raise ValueError(f"Unknown static label(s) in CSV: {set(y_labels) - set(label_encoder.classes_)}. Error: {e}")
            y_categorical = to_categorical(y_encoded, num_classes=STATIC_NUM_CLASSES)
            with open(self.encoder_path, 'wb') as f: pickle.dump(label_encoder, f); self.progress_signal.emit(f"Static label encoder saved to {self.encoder_path}")
            assert len(df) >= 10 and df['label'].nunique() >= 2, "Need >= 10 samples and >= 2 classes for static training."
            try: X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical)
            except ValueError: X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
            self.progress_signal.emit(f"Static data split: Train={X_train.shape[0]}, Val={X_val.shape[0]}"); self.progress_signal.emit("Building static Keras model...")
            model = Sequential([ Dense(128, activation='relu', input_shape=(STATIC_NUM_FEATURES,)), Dropout(0.3), Dense(64, activation='relu'), Dropout(0.3), Dense(STATIC_NUM_CLASSES, activation='softmax') ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']); model.summary(print_fn=lambda x: self.progress_signal.emit(x)); self.progress_signal.emit("Starting static model training...")
            cbs = [EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True), ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)]
            history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=cbs, verbose=0)
            val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0); self.progress_signal.emit(f"Static training finished. Val Accuracy: {val_accuracy:.4f}")
            self.progress_signal.emit(f"Saving static model to {self.model_path}..."); model.save(self.model_path); self.finished_signal.emit(True, f"Static training complete. Model saved. Val Acc: {val_accuracy:.4f}")
        except (FileNotFoundError, ValueError, AssertionError) as e: self.finished_signal.emit(False, f"Static Training Error: {e}")
        except Exception as e: import traceback; traceback.print_exc(); self.finished_signal.emit(False, f"Unexpected static training error: {e}")

class PathTrainingThread(QThread):
    progress_signal = pyqtSignal(str); finished_signal = pyqtSignal(bool, str)
    def __init__(self, csv_path, model_path, encoder_path, scaler_path, parent=None): super().__init__(parent); self.csv_path, self.model_path, self.encoder_path, self.scaler_path = csv_path, model_path, encoder_path, scaler_path
    def run(self):
        try:
            self.progress_signal.emit("Starting path model training..."); self.progress_signal.emit(f"Loading path data from {self.csv_path}...")
            if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0: raise FileNotFoundError(f"Path CSV '{self.csv_path}' not found or empty.")
            df = pd.read_csv(self.csv_path); df.dropna(inplace=True); assert not df.empty, "Path CSV is empty after dropping NaNs."
            self.progress_signal.emit(f"Path data loaded: {df.shape[0]} samples."); self.progress_signal.emit("Preprocessing path data...")
            X = df.iloc[:, 1:].values; y_labels = df.iloc[:, 0].values; scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
            with open(self.scaler_path, 'wb') as f: pickle.dump(scaler, f); self.progress_signal.emit(f"Path feature scaler saved to {self.scaler_path}")
            label_encoder = LabelEncoder(); label_encoder.fit(PATH_ALLOWED_LABELS)
            try: y_encoded = label_encoder.transform(y_labels)
            except ValueError as e: raise ValueError(f"Unknown path label(s) in CSV: {set(y_labels) - set(label_encoder.classes_)}. Error: {e}")
            y_categorical = to_categorical(y_encoded, num_classes=PATH_NUM_CLASSES)
            with open(self.encoder_path, 'wb') as f: pickle.dump(label_encoder, f); self.progress_signal.emit(f"Path label encoder saved to {self.encoder_path}")
            assert len(df) >= 10 and df['label'].nunique() >= 2, "Need >= 10 samples and >= 2 classes for path training."
            try: X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical)
            except ValueError: X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)
            self.progress_signal.emit(f"Path data split: Train={X_train.shape[0]}, Val={X_val.shape[0]}"); self.progress_signal.emit("Building path Keras model (MLP)...")
            model = Sequential([ Dense(256, activation='relu', input_shape=(PATH_NUM_FEATURES,)), Dropout(0.4), Dense(128, activation='relu'), Dropout(0.4), Dense(64, activation='relu'), Dropout(0.4), Dense(PATH_NUM_CLASSES, activation='softmax') ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']); model.summary(print_fn=lambda x: self.progress_signal.emit(x)); self.progress_signal.emit("Starting path model training...")
            cbs = [EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True), ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=0.00001)]
            history = model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_val, y_val), callbacks=cbs, verbose=0)
            val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0); self.progress_signal.emit(f"Path training finished. Val Accuracy: {val_accuracy:.4f}")
            self.progress_signal.emit(f"Saving path model to {self.model_path}..."); model.save(self.model_path); self.finished_signal.emit(True, f"Path training complete. Model saved. Val Acc: {val_accuracy:.4f}")
        except (FileNotFoundError, ValueError, AssertionError) as e: self.finished_signal.emit(False, f"Path Training Error: {e}")
        except Exception as e: import traceback; traceback.print_exc(); self.finished_signal.emit(False, f"Unexpected path training error: {e}")

# --- Main Application Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)
        # --- State Variables ---
        self.camera_thread = None; self.static_training_thread = None; self.path_training_thread = None; self.current_mode = 'idle'
        self.static_model = None; self.static_label_encoder = None; self.static_data_counts = {}
        self.path_model = None; self.path_label_encoder = None; self.path_scaler = None; self.path_data_counts = {}
        self.current_path_points = []; self.is_recording_path = False # Controls drawing in main window
        self.selected_path_label = PATH_ALLOWED_LABELS[0] if PATH_ALLOWED_LABELS else None; self.current_recording_path_label = None
        self.combined_result = "---"
        # --- GUI Elements ---
        self.central_widget = QWidget(); self.setCentralWidget(self.central_widget); self.layout = QVBoxLayout(self.central_widget)
        self.top_layout = QHBoxLayout(); self.layout.addLayout(self.top_layout)
        self.camera_label = QLabel("Camera Feed"); self.camera_label.setAlignment(Qt.AlignCenter); self.camera_label.setFixedSize(CAMERA_FEED_WIDTH, CAMERA_FEED_HEIGHT); self.camera_label.setStyleSheet("border: 1px solid black; background-color: #f0f0f0;"); self.top_layout.addWidget(self.camera_label)
        self.tab_widget = QTabWidget(); self.top_layout.addWidget(self.tab_widget)
        self.static_tab = QWidget(); self.path_tab = QWidget(); self.combined_tab = QWidget()
        self.tab_widget.addTab(self.static_tab, "Static Gestures"); self.tab_widget.addTab(self.path_tab, "Path Gestures"); self.tab_widget.addTab(self.combined_tab, "Combined Mode")
        self.static_layout = QVBoxLayout(self.static_tab); self.path_layout = QVBoxLayout(self.path_tab); self.combined_layout = QVBoxLayout(self.combined_tab)
        # Static Tab
        self.static_collect_label = QLabel("1. Static Gesture Collection"); self.static_collect_label.setFont(QFont("Arial", 11, QFont.Bold)); self.static_layout.addWidget(self.static_collect_label)
        self.start_static_collect_button = QPushButton("Start Collecting Static"); self.start_static_collect_button.clicked.connect(self.start_static_collection_mode); self.static_layout.addWidget(self.start_static_collect_button)
        self.stop_static_collect_button = QPushButton("Stop Collecting Static"); self.stop_static_collect_button.clicked.connect(self.stop_camera); self.static_layout.addWidget(self.stop_static_collect_button)
        self.static_collect_instructions = QLabel(f"Press keys ({', '.join(STATIC_ALLOWED_LABELS)})"); self.static_layout.addWidget(self.static_collect_instructions)
        self.static_data_count_label = QLabel("Static Samples: 0"); self.static_layout.addWidget(self.static_data_count_label)
        self.static_layout.addSpacing(15)
        self.static_train_label = QLabel("2. Static Model Training"); self.static_train_label.setFont(QFont("Arial", 11, QFont.Bold)); self.static_layout.addWidget(self.static_train_label)
        self.static_train_button = QPushButton("Train Static Model"); self.static_train_button.clicked.connect(self.start_static_training); self.static_layout.addWidget(self.static_train_button)
        self.static_layout.addSpacing(15)
        self.static_classify_label = QLabel("3. Static Classification"); self.static_classify_label.setFont(QFont("Arial", 11, QFont.Bold)); self.static_layout.addWidget(self.static_classify_label)
        self.start_static_classify_button = QPushButton("Start Real-time Static Classif."); self.start_static_classify_button.clicked.connect(self.start_static_classification_mode); self.static_layout.addWidget(self.start_static_classify_button)
        self.stop_static_classify_button = QPushButton("Stop Static Classif."); self.stop_static_classify_button.clicked.connect(self.stop_camera); self.static_layout.addWidget(self.stop_static_classify_button)
        self.classify_static_image_button = QPushButton("Classify Static from Image"); self.classify_static_image_button.clicked.connect(self.classify_static_from_image); self.static_layout.addWidget(self.classify_static_image_button)
        self.static_layout.addStretch()
        # Path Tab
        self.path_collect_label = QLabel("4. Path Gesture Collection"); self.path_collect_label.setFont(QFont("Arial", 11, QFont.Bold)); self.path_layout.addWidget(self.path_collect_label)
        self.start_path_collect_button = QPushButton("Start Collecting Paths"); self.start_path_collect_button.clicked.connect(self.start_path_collection_mode); self.path_layout.addWidget(self.start_path_collect_button)
        self.stop_path_collect_button = QPushButton("Stop Collecting Paths"); self.stop_path_collect_button.clicked.connect(self.stop_camera); self.path_layout.addWidget(self.stop_path_collect_button)
        self.path_label_select_layout = QHBoxLayout(); self.path_label_select_label = QLabel("Select Path Label:"); self.path_label_select_combo = QComboBox(); self.path_label_select_combo.addItems(PATH_ALLOWED_LABELS)
        if PATH_ALLOWED_LABELS: self.selected_path_label = PATH_ALLOWED_LABELS[0]; self.path_label_select_combo.setCurrentText(self.selected_path_label)
        self.path_label_select_combo.currentTextChanged.connect(self._update_selected_path_label); self.path_label_select_layout.addWidget(self.path_label_select_label); self.path_label_select_layout.addWidget(self.path_label_select_combo); self.path_layout.addLayout(self.path_label_select_layout)
        record_key_name = QKeySequence(PATH_RECORD_KEY).toString(QKeySequence.NativeText)
        self.path_collect_instructions = QLabel(f"Select label, then Press/Hold '{record_key_name}' to record."); self.path_layout.addWidget(self.path_collect_instructions)
        self.path_data_count_label = QLabel("Path Samples: 0"); self.path_layout.addWidget(self.path_data_count_label)
        self.path_layout.addSpacing(15)
        self.path_train_label = QLabel("5. Path Model Training"); self.path_train_label.setFont(QFont("Arial", 11, QFont.Bold)); self.path_layout.addWidget(self.path_train_label)
        self.path_train_button = QPushButton("Train Path Model"); self.path_train_button.clicked.connect(self.start_path_training); self.path_layout.addWidget(self.path_train_button)
        self.path_layout.addSpacing(15)
        self.path_classify_label = QLabel("6. Path Classification"); self.path_classify_label.setFont(QFont("Arial", 11, QFont.Bold)); self.path_layout.addWidget(self.path_classify_label)
        self.start_path_classify_button = QPushButton("Start Real-time Path Classif."); self.start_path_classify_button.clicked.connect(self.start_path_classification_mode); self.path_layout.addWidget(self.start_path_classify_button)
        self.stop_path_classify_button = QPushButton("Stop Path Classif."); self.stop_path_classify_button.clicked.connect(self.stop_camera); self.path_layout.addWidget(self.stop_path_classify_button)
        self.path_classify_instructions = QLabel(f"Press/Hold '{record_key_name}' to record path for classif."); self.path_layout.addWidget(self.path_classify_instructions)
        self.path_prediction_label = QLabel("Path Prediction: ---"); self.path_layout.addWidget(self.path_prediction_label)
        self.path_layout.addStretch()
        # Combined Tab
        self.combined_label = QLabel("7. Combined Classification"); self.combined_label.setFont(QFont("Arial", 11, QFont.Bold)); self.combined_layout.addWidget(self.combined_label)
        self.start_combined_button = QPushButton("Start Combined Mode"); self.start_combined_button.clicked.connect(self.start_combined_mode); self.combined_layout.addWidget(self.start_combined_button)
        self.stop_combined_button = QPushButton("Stop Combined Mode"); self.stop_combined_button.clicked.connect(self.stop_camera); self.combined_layout.addWidget(self.stop_combined_button)
        self.combined_result_label = QLabel("Combined Result: ---"); font = self.combined_result_label.font(); font.setPointSize(14); font.setBold(True); self.combined_result_label.setFont(font); self.combined_layout.addWidget(self.combined_result_label)
        # History Text Box
        self.history_label = QLabel("Recognized Sequence:"); self.combined_layout.addWidget(self.history_label) # Add to combined layout
        self.history_text_edit = QTextEdit(); self.history_text_edit.setReadOnly(True); self.history_text_edit.setFixedHeight(150); self.combined_layout.addWidget(self.history_text_edit) # Add to combined layout
        self.combined_layout.addStretch()
        # Status Bar
        self.setStatusBar(QStatusBar(self)); self.statusBar().showMessage("Ready.")
        # Load initial state
        self._load_initial_counts(); self._load_models_and_encoders(); self._update_button_states(); self._update_data_count_display()

    # --- Data Count & Loading ---
    def _load_initial_counts(self):
        self.static_data_counts = {label: 0 for label in STATIC_ALLOWED_LABELS}
        if os.path.exists(STATIC_CSV_FILE) and os.path.getsize(STATIC_CSV_FILE) > 0:
            try: df = pd.read_csv(STATIC_CSV_FILE); [self.static_data_counts.update({l:c}) for l,c in df['label'].value_counts().to_dict().items() if l in self.static_data_counts]
            except Exception as e: self.show_error("Static CSV Load Error", f"{e}")
        self.path_data_counts = {label: 0 for label in PATH_ALLOWED_LABELS}
        if os.path.exists(PATH_CSV_FILE) and os.path.getsize(PATH_CSV_FILE) > 0:
            try: df = pd.read_csv(PATH_CSV_FILE); [self.path_data_counts.update({l:c}) for l,c in df['label'].value_counts().to_dict().items() if l in self.path_data_counts]
            except Exception as e: self.show_error("Path CSV Load Error", f"{e}")
    def _update_data_count_display(self):
        self.static_data_count_label.setText(f"Static Samples: {sum(self.static_data_counts.values())}"); self.path_data_count_label.setText(f"Path Samples: {sum(self.path_data_counts.values())}")
    def _load_models_and_encoders(self):
        static_loaded, path_loaded = False, False
        if os.path.exists(STATIC_MODEL_FILE) and os.path.exists(STATIC_ENCODER_FILE):
            try: self.static_model = load_model(STATIC_MODEL_FILE); self.static_label_encoder = pickle.load(open(STATIC_ENCODER_FILE, 'rb')); static_loaded = True if len(self.static_label_encoder.classes_) == STATIC_NUM_CLASSES and self.static_model.output_shape[-1] == STATIC_NUM_CLASSES else False; assert static_loaded, "Static model/encoder mismatch."
            except Exception as e: self.show_error("Static Load Error", f"{e}"); self.static_model = self.static_label_encoder = None; static_loaded = False
        if os.path.exists(PATH_MODEL_FILE) and os.path.exists(PATH_ENCODER_FILE) and os.path.exists(PATH_SCALER_FILE):
             try: self.path_model = load_model(PATH_MODEL_FILE); self.path_label_encoder = pickle.load(open(PATH_ENCODER_FILE, 'rb')); self.path_scaler = pickle.load(open(PATH_SCALER_FILE, 'rb')); path_loaded = True if len(self.path_label_encoder.classes_) == PATH_NUM_CLASSES and self.path_model.output_shape[-1] == PATH_NUM_CLASSES else False; assert path_loaded, "Path model/encoder mismatch."
             except Exception as e: self.show_error("Path Load Error", f"{e}"); self.path_model = self.path_label_encoder = self.path_scaler = None; path_loaded = False
        status = []; [status.append(s) for s, l in [("Static model loaded.", static_loaded), ("Path model loaded.", path_loaded)] if l]; self.statusBar().showMessage(" ".join(status) if status else "No models loaded. Collect data and train.")
        self._update_button_states(); return static_loaded, path_loaded

    # --- Mode Control & Button States --- (Modified to clear history)
    def _set_mode(self, new_mode):
        if self.camera_thread and self.camera_thread.isRunning(): self.show_message("Camera Busy", "Stop the current camera process first."); return False
        self.combined_result = "---"; self.combined_result_label.setText(f"Combined Result: {self.combined_result}")
        self.history_text_edit.clear() # Clear history on mode change
        self.current_mode = new_mode; self.is_recording_path = False; self.current_path_points = []; self.current_recording_path_label = None; self.path_prediction_label.setText("Path Prediction: ---")
        if new_mode == 'idle': self.stop_camera()
        else:
            models_ok = True
            if new_mode == 'classify_static' and (not self.static_model or not self.static_label_encoder): self.show_message("Model Error", "Static model not loaded."); models_ok = False
            elif new_mode == 'classify_path' and (not self.path_model or not self.path_label_encoder or not self.path_scaler): self.show_message("Model Error", "Path model/scaler not loaded."); models_ok = False
            elif new_mode == 'combined' and (not self.static_model or not self.static_label_encoder or not self.path_model or not self.path_label_encoder or not self.path_scaler): self.show_message("Model Error", "Both Static and Path models/etc must be loaded."); models_ok = False
            if not models_ok: self.current_mode = 'idle'; self._update_button_states(); return False
            self.camera_thread = CameraThread(mode=new_mode); self._connect_camera_signals()
            if new_mode in ['classify_static', 'combined']: self.camera_thread.load_static_model(self.static_model, self.static_label_encoder)
            if new_mode in ['classify_path', 'combined']: self.camera_thread.load_path_model(self.path_model, self.path_label_encoder, self.path_scaler)
            self.camera_thread.start(); self.statusBar().showMessage(f"Mode set to: {new_mode}")
            if new_mode in ['collect_static', 'collect_path', 'classify_path']: self.setFocus()
        self._update_button_states(); return True
    def _update_button_states(self):
        mode = self.current_mode; is_idle = mode == 'idle'; is_cs = mode == 'collect_static'; is_cls = mode == 'classify_static'; is_cp = mode == 'collect_path'; is_clp = mode == 'classify_path'; is_comb = mode == 'combined'
        is_training = (self.static_training_thread and self.static_training_thread.isRunning()) or (self.path_training_thread and self.path_training_thread.isRunning())
        static_loaded = self.static_model is not None and self.static_label_encoder is not None; path_loaded = self.path_model is not None and self.path_label_encoder is not None and self.path_scaler is not None; both_loaded = static_loaded and path_loaded
        cam_running = self.camera_thread is not None and self.camera_thread.isRunning()
        # Static Tab
        self.start_static_collect_button.setEnabled(is_idle and not is_training and not cam_running)
        self.stop_static_collect_button.setEnabled(is_cs)
        self.static_train_button.setEnabled(is_idle and not is_training and not cam_running)
        self.start_static_classify_button.setEnabled(is_idle and not is_training and static_loaded and not cam_running)
        self.stop_static_classify_button.setEnabled(is_cls)
        self.classify_static_image_button.setEnabled(is_idle and not is_training and static_loaded and not cam_running)
        # Path Tab
        self.start_path_collect_button.setEnabled(is_idle and not is_training and not cam_running)
        self.stop_path_collect_button.setEnabled(is_cp)
        self.path_train_button.setEnabled(is_idle and not is_training and not cam_running)
        self.start_path_classify_button.setEnabled(is_idle and not is_training and path_loaded and not cam_running)
        self.stop_path_classify_button.setEnabled(is_clp)
        self.path_label_select_combo.setEnabled(is_idle or is_cp or is_clp)
        # Combined Tab
        self.start_combined_button.setEnabled(is_idle and not is_training and both_loaded and not cam_running)
        self.stop_combined_button.setEnabled(is_comb)

        # Disable train buttons if camera running
        if cam_running:
            self.static_train_button.setEnabled(False)
            self.path_train_button.setEnabled(False)

    def stop_camera(self):
        if self.camera_thread and self.camera_thread.isRunning(): self.camera_thread.stop(); self.camera_thread = None
        self.current_mode = 'idle'; self.is_recording_path = False; self.current_path_points = []
        self.camera_label.setText("Camera Stopped."); self.camera_label.setPixmap(QPixmap()); self.statusBar().showMessage("Camera stopped. Mode set to Idle.")
        self.combined_result_label.setText("Combined Result: ---")
        self.history_text_edit.clear() # Clear history on stop
        self._update_button_states()

    # --- Signal Connections & Handling --- (No changes)
    def _connect_camera_signals(self):
        if not self.camera_thread: return
        connections = [
            (self.camera_thread.frame_signal, self.update_frame),
            (self.camera_thread.static_landmarks_signal, self.save_static_landmarks),
            (self.camera_thread.path_point_signal, self.append_path_point),
            (self.camera_thread.combined_path_ready_signal, self.handle_combined_path),
            (self.camera_thread.path_recording_started_signal, self._on_path_recording_started),
            (self.camera_thread.path_recording_stopped_signal, self._on_path_recording_stopped),
            (self.camera_thread.status_signal, self.update_status),
            (self.camera_thread.error_signal, self.show_error),
            (self.camera_thread.finished, self._on_camera_thread_finished)
        ]
        for signal, _ in connections:
            try: signal.disconnect()
            except TypeError: pass
        for signal, slot in connections:
            signal.connect(slot)
    def _on_camera_thread_finished(self):
        if self.current_mode != 'idle': self.statusBar().showMessage("Camera thread finished unexpectedly.")
        self.camera_thread = None; self._update_button_states()

    # --- Static Gesture Methods --- (No changes)
    def start_static_collection_mode(self): self._set_mode('collect_static')
    def start_static_classification_mode(self): self._set_mode('classify_static')
    def save_static_landmarks(self, data):
        if len(data) > 1: label, landmarks = data[0], data[1:];
        if save_static_to_csv(STATIC_CSV_FILE, label, landmarks): self.static_data_counts[label] = self.static_data_counts.get(label, 0) + 1; self._update_data_count_display()
        else: self.show_error("Save Error", f"Failed to save static data for '{label}'.")
    def start_static_training(self):
        if self.static_training_thread and self.static_training_thread.isRunning(): self.show_message("Training Busy", "Static training already running."); return
        if self.current_mode != 'idle': self.show_message("Action Required", "Stop camera before static training."); return
        if not os.path.exists(STATIC_CSV_FILE) or os.path.getsize(STATIC_CSV_FILE) == 0: self.show_error("Training Error", f"Static CSV '{STATIC_CSV_FILE}' not found/empty."); return
        self.statusBar().showMessage("Starting static training..."); self.static_training_thread = StaticTrainingThread(STATIC_CSV_FILE, STATIC_MODEL_FILE, STATIC_ENCODER_FILE)
        self.static_training_thread.progress_signal.connect(self.update_status); self.static_training_thread.finished_signal.connect(self.on_static_training_finished)
        self.static_training_thread.start(); self._update_button_states()
    def on_static_training_finished(self, success, message):
        if success: self.show_message("Static Training Complete", message); self._load_models_and_encoders()
        else: self.show_error("Static Training Failed", message)
        self.static_training_thread = None; self._update_button_states(); self.statusBar().showMessage("Static training finished. Ready.")
    def classify_static_from_image(self):
        if not self.static_model or not self.static_label_encoder: self.show_message("Model Not Loaded", "Static model not available."); return
        options = QFileDialog.Options(); options |= QFileDialog.DontUseNativeDialog; filepath, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        if not filepath: return
        self.statusBar().showMessage(f"Classifying static gesture in image...")
        try:
            img = cv2.imread(filepath); assert img is not None, "Could not read image."
            img_display = img.copy(); rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); rgb_image.flags.writeable = False; results = hands.process(rgb_image); rgb_image.flags.writeable = True
            prediction_text = "No hand detected."
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]; processed_landmarks = process_static_landmarks(hand_landmarks)
                if processed_landmarks:
                    input_data = np.array([processed_landmarks], dtype=np.float32); prediction = self.static_model.predict(input_data, verbose=0)[0]
                    idx = np.argmax(prediction); conf = prediction[idx]; label = self.static_label_encoder.inverse_transform([idx])[0]; prediction_text = f"Static: {label} (Conf: {conf:.2f})"
                    mp_drawing.draw_landmarks(img_display, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
                else: prediction_text = "Hand detected, landmarks invalid."
            cv2.putText(img_display, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            qt_image = _convert_cv_qt(img_display); scaled_pixmap = QPixmap.fromImage(qt_image).scaled(self.camera_label.width(), self.camera_label.height(), Qt.KeepAspectRatio)
            self.camera_label.setPixmap(scaled_pixmap); self.statusBar().showMessage(f"Image classification: {prediction_text}")
        except Exception as e: self.show_error("Static Image Classification Error", f"{e}"); self.statusBar().showMessage("Static image classification failed.")

    # --- Path Gesture Methods --- (No changes)
    def start_path_collection_mode(self): self._set_mode('collect_path')
    def start_path_classification_mode(self): self._set_mode('classify_path')
    def _update_selected_path_label(self, label): self.selected_path_label = label
    def append_path_point(self, point):
        if self.is_recording_path: self.current_path_points.append(point); self.camera_label.update()
    def process_and_save_path(self):
        if not self.current_recording_path_label: print("Warning: Path recorded without a label assigned."); return
        if len(self.current_path_points) < 5: self.statusBar().showMessage(f"Path '{self.current_recording_path_label}' too short, discarded."); return
        self.statusBar().showMessage(f"Processing path '{self.current_recording_path_label}' ({len(self.current_path_points)} points)...")
        standardized_flat = standardize_path(self.current_path_points)
        if standardized_flat is not None:
            if save_path_to_csv(PATH_CSV_FILE, self.current_recording_path_label, standardized_flat):
                self.path_data_counts[self.current_recording_path_label] = self.path_data_counts.get(self.current_recording_path_label, 0) + 1
                self._update_data_count_display(); self.statusBar().showMessage(f"Path '{self.current_recording_path_label}' saved.")
            else: self.show_error("Save Error", f"Failed to save path data for '{self.current_recording_path_label}'.")
        else: self.show_warning("Path Processing Error", f"Could not standardize path for '{self.current_recording_path_label}'.")
    def classify_live_path(self):
        if not self.path_model or not self.path_label_encoder or not self.path_scaler: self.statusBar().showMessage("Path model/encoder/scaler not loaded."); return
        if len(self.current_path_points) < 5: self.statusBar().showMessage("Path too short to classify."); self.path_prediction_label.setText("Path Prediction: Too short"); return
        self.statusBar().showMessage(f"Classifying path ({len(self.current_path_points)} points)...")
        standardized_flat = standardize_path(self.current_path_points)
        if standardized_flat is not None:
            try:
                input_data = self.path_scaler.transform(standardized_flat.reshape(1, -1)); prediction = self.path_model.predict(input_data, verbose=0)[0]
                idx = np.argmax(prediction); conf = prediction[idx]; label = self.path_label_encoder.inverse_transform([idx])[0]
                pred_text = f"Path Prediction: {label} ({conf:.2f})"
                self.path_prediction_label.setText(pred_text); self.statusBar().showMessage(f"Classification result: {label} ({conf:.2f})")
            except Exception as e: self.show_error("Path Classification Error", f"{e}"); self.path_prediction_label.setText("Path Prediction: Error"); self.statusBar().showMessage("Path classification error.")
        else: self.show_warning("Path Processing Error", "Could not standardize path for classification."); self.path_prediction_label.setText("Path Prediction: Proc. Error"); self.statusBar().showMessage("Path processing error during classification.")
    def start_path_training(self):
        if self.path_training_thread and self.path_training_thread.isRunning(): self.show_message("Training Busy", "Path training already running."); return
        if self.current_mode != 'idle': self.show_message("Action Required", "Stop camera before path training."); return
        if not os.path.exists(PATH_CSV_FILE) or os.path.getsize(PATH_CSV_FILE) == 0: self.show_error("Training Error", f"Path CSV '{PATH_CSV_FILE}' not found/empty."); return
        self.statusBar().showMessage("Starting path training..."); self.path_training_thread = PathTrainingThread(PATH_CSV_FILE, PATH_MODEL_FILE, PATH_ENCODER_FILE, PATH_SCALER_FILE)
        self.path_training_thread.progress_signal.connect(self.update_status); self.path_training_thread.finished_signal.connect(self.on_path_training_finished)
        self.path_training_thread.start(); self._update_button_states()
    def on_path_training_finished(self, success, message):
        if success: self.show_message("Path Training Complete", message); self._load_models_and_encoders()
        else: self.show_error("Path Training Failed", message)
        self.path_training_thread = None; self._update_button_states(); self.statusBar().showMessage("Path training finished. Ready.")

    # --- Combined Mode Methods ---
    def start_combined_mode(self): self._set_mode('combined')

    def handle_combined_path(self, path_points, static_label): # Modified
        """Receives path data from thread, classifies path, applies rules, updates history."""
        self.statusBar().showMessage(f"Received path for static '{static_label}'. Classifying path...")
        # Path drawing is stopped by _on_path_recording_stopped signal

        final_result = static_label # Default result is the static label

        if not self.path_model or not self.path_label_encoder or not self.path_scaler:
             self.statusBar().showMessage("Path model/encoder/scaler not loaded for combined mode.")
        elif len(path_points) < 5:
            self.statusBar().showMessage("Received path too short to classify.")
        else:
            standardized_flat = standardize_path(path_points)
            if standardized_flat is not None:
                try:
                    input_data = self.path_scaler.transform(standardized_flat.reshape(1, -1))
                    prediction = self.path_model.predict(input_data, verbose=0)[0]
                    idx = np.argmax(prediction); conf = prediction[idx]
                    path_label = self.path_label_encoder.inverse_transform([idx])[0]
                    self.statusBar().showMessage(f"Static: {static_label}, Path: {path_label} ({conf:.2f}). Applying rules...")
                    combination_key = (static_label, path_label)
                    final_result = COMBINATION_RULES.get(combination_key, static_label) # Use default if no rule
                except Exception as e:
                    self.show_error("Combined Path Classification Error", f"{e}")
                    self.statusBar().showMessage("Combined path classification error.")
            else:
                self.show_warning("Combined Path Processing Error", "Could not standardize path.")
                self.statusBar().showMessage("Combined path processing error.")

        # Update results and history
        self.combined_result = final_result
        self.combined_result_label.setText(f"Combined Result: {self.combined_result}")
        self.history_text_edit.append(self.combined_result) # Append to history <<<<< RE-ADDED

        # Clear path points *after* processing (drawing already stopped)
        self.current_path_points = []


    # --- GUI Update Slots --- (Modified)
    def update_frame(self, q_image):
        """Updates the camera feed label, drawing the path if recording in relevant modes."""
        pixmap = QPixmap.fromImage(q_image)
        # Use main window's is_recording_path flag for drawing consistency
        if self.is_recording_path and len(self.current_path_points) > 1 and \
           self.current_mode in ['collect_path', 'classify_path', 'combined']:
            painter = QPainter(pixmap); pen = QPen(QColor(0, 0, 255), 3); painter.setPen(pen)
            w, h = pixmap.width(), pixmap.height()
            path_points_scaled = [QPoint(int(p[0] * w), int(p[1] * h)) for p in self.current_path_points]
            for i in range(len(path_points_scaled) - 1): painter.drawLine(path_points_scaled[i], path_points_scaled[i+1])
            painter.end()
        scaled_pixmap = pixmap.scaled(self.camera_label.width(), self.camera_label.height(), Qt.KeepAspectRatio)
        self.camera_label.setPixmap(scaled_pixmap)

    def _on_path_recording_started(self): # Slot for combined mode drawing start
        """Sets flag to start drawing path in combined mode."""
        if self.current_mode == 'combined':
             self.is_recording_path = True
             self.current_path_points = [] # Clear points when starting a new path

    def _on_path_recording_stopped(self): # Slot for combined mode drawing stop
        """Sets flag to stop drawing path in combined mode and clears points."""
        if self.current_mode == 'combined':
             self.is_recording_path = False
             # Clear points immediately when recording stops in combined mode
             self.current_path_points = []
             self.camera_label.update() # Trigger one last paint to clear path

    def update_status(self, message): self.statusBar().showMessage(message)
    def show_error(self, title, message): QMessageBox.critical(self, title, message); self.statusBar().showMessage(f"Error: {message[:100]}")
    def show_warning(self, title, message): QMessageBox.warning(self, title, message); self.statusBar().showMessage(f"Warning: {message[:100]}")
    def show_message(self, title, message): QMessageBox.information(self, title, message)

    # --- Event Handling --- (No changes needed)
    def keyPressEvent(self, event):
        if event.isAutoRepeat(): return
        key = event.key(); key_text = event.text().upper()
        if self.current_mode == 'collect_static' and key_text in STATIC_ALLOWED_LABELS:
            if self.camera_thread: self.camera_thread.set_static_label(key_text)
        elif self.current_mode in ['collect_path', 'classify_path'] and key == PATH_RECORD_KEY:
            if not self.is_recording_path:
                if not self.selected_path_label: self.show_warning("No Path Label", "Please select a path label first."); return
                self.is_recording_path = True; self.current_recording_path_label = self.selected_path_label; self.current_path_points = []
                if self.camera_thread: self.camera_thread.set_path_recording(True)
                self.path_prediction_label.setText("Path Prediction: Recording..."); self.statusBar().showMessage(f"Recording path for '{self.current_recording_path_label}'...")
        else: super().keyPressEvent(event)
    def keyReleaseEvent(self, event):
        if event.isAutoRepeat(): return
        key = event.key()
        if self.current_mode in ['collect_path', 'classify_path'] and key == PATH_RECORD_KEY and self.is_recording_path:
            self.is_recording_path = False
            if self.camera_thread: self.camera_thread.set_path_recording(False)
            if self.current_recording_path_label:
                self.statusBar().showMessage(f"Finished recording path for '{self.current_recording_path_label}'. Processing...")
                if self.current_mode == 'collect_path': self.process_and_save_path()
                elif self.current_mode == 'classify_path': self.classify_live_path()
            else: self.statusBar().showMessage("Finished recording path (no label assigned). Discarded.")
            self.current_path_points = []; self.current_recording_path_label = None
            self.camera_label.update()
        else: super().keyReleaseEvent(event)
    def closeEvent(self, event):
        self.stop_camera()
        if (self.static_training_thread and self.static_training_thread.isRunning()) or \
           (self.path_training_thread and self.path_training_thread.isRunning()): print("Warning: Training is running. Closing anyway.")
        hands.close(); print("MediaPipe Hands closed.")
        event.accept()

# --- Main Execution ---
if __name__ == '__main__':
    from PyQt5.QtGui import QKeySequence
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
