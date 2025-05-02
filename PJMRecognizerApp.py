# -*- coding: utf-8 -*-
# gesture_recognition_app.py

import sys
import os
import csv
import time
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
    QScrollArea, QComboBox # Added QComboBox
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QColor # Added QPainter etc.
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPoint # Added QPoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout # Removed LSTM as it wasn't used
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- Constants ---
APP_NAME = "Gesture Recognition App (Static + Path)"
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 750
CAMERA_FEED_WIDTH = 640
CAMERA_FEED_HEIGHT = 480

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
# Updated Path Labels <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< UPDATED
PATH_ALLOWED_LABELS = ['J', 'left', 'right', 'down', 'tail', 'zigzagl', 'zigzagr', 'zkropka']
PATH_NUM_CLASSES = len(PATH_ALLOWED_LABELS) # Automatically updated count
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PATH_LENGTH = 100
PATH_NUM_FEATURES = PATH_LENGTH * 2 # x, y
PATH_TRACKING_LANDMARK = 8 # Index finger tip
PATH_RECORD_KEY = Qt.Key_Q # Key to press for recording paths

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# --- Helper Functions ---

# --- Static Gesture Helpers ---
def process_static_landmarks(hand_landmarks):
    """Processes landmarks for static gestures."""
    if not hand_landmarks: return None
    landmark_list = []
    base_x, base_y, base_z = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z
    for landmark in hand_landmarks.landmark:
        landmark_list.extend([landmark.x - base_x, landmark.y - base_y, landmark.z - base_z])
    if len(landmark_list) == STATIC_NUM_FEATURES: return landmark_list
    else: return None

def save_static_to_csv(filepath, label, landmarks):
    """Saves static gesture data."""
    file_exists = os.path.isfile(filepath)
    header = ['label'] + [f'{axis}{i}' for i in range(STATIC_NUM_LANDMARKS) for axis in ['x', 'y', 'z']]
    try:
        with open(filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists or os.path.getsize(filepath) == 0: writer.writerow(header)
            if landmarks and len(landmarks) == STATIC_NUM_FEATURES:
                writer.writerow([label] + landmarks)
                return True
            else: return False
    except IOError as e: print(f"Error saving static CSV: {e}"); return False

# --- Path Gesture Helpers ---
def standardize_path(points, length=PATH_LENGTH):
    """Standardizes a path to a fixed number of points using interpolation."""
    if len(points) < 2: return None # Need at least two points to interpolate

    points_np = np.array(points)
    # Calculate cumulative distance along the path
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points_np, axis=0)**2, axis=1)))
    distance = np.insert(distance, 0, 0) # Add start distance (0)

    # Handle paths with zero length (e.g., single point)
    if distance[-1] == 0:
        # Repeat the single point 'length' times
        standardized_points = np.array([points_np[0]] * length)
        return standardized_points.flatten()

    # Create evenly spaced points along the total path length
    alpha = np.linspace(0, distance[-1], length)

    # Interpolate x and y coordinates
    # Use bounds_error=False and fill_value="extrapolate" to handle edge cases if alpha goes slightly beyond distance bounds due to floating point precision
    interp_x = interp1d(distance, points_np[:, 0], kind='linear', bounds_error=False, fill_value="extrapolate")
    interp_y = interp1d(distance, points_np[:, 1], kind='linear', bounds_error=False, fill_value="extrapolate")

    standardized = np.vstack((interp_x(alpha), interp_y(alpha))).T
    return standardized.flatten() # Return flattened array [x1, y1, x2, y2, ...]

def save_path_to_csv(filepath, label, path_coords):
    """Saves standardized path data."""
    file_exists = os.path.isfile(filepath)
    header = ['label'] + [f'{axis}{i}' for i in range(PATH_LENGTH) for axis in ['x', 'y']]
    try:
        with open(filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists or os.path.getsize(filepath) == 0: writer.writerow(header)
            if path_coords is not None and len(path_coords) == PATH_NUM_FEATURES:
                writer.writerow([label] + path_coords.tolist()) # Convert numpy array to list
                return True
            else:
                print(f"Warning: Attempted to save invalid path data for label {label}. Length: {len(path_coords) if path_coords is not None else 'None'} != {PATH_NUM_FEATURES}")
                return False
    except IOError as e: print(f"Error saving path CSV: {e}"); return False


# --- PyQt Threads ---

class CameraThread(QThread):
    """Handles camera capture and processing in a separate thread."""
    frame_signal = pyqtSignal(QImage)
    static_landmarks_signal = pyqtSignal(list) # For static gesture saving
    path_point_signal = pyqtSignal(tuple)      # Emits (x,y) of tracked landmark
    status_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, mode='idle', parent=None):
        super().__init__(parent)
        self.running = False
        self.mode = mode # 'idle', 'collect_static', 'classify_static', 'collect_path', 'classify_path'
        self.current_static_label = None
        self.static_model = None
        self.static_label_encoder = None
        self.path_model = None
        self.path_label_encoder = None
        self.path_scaler = None
        self.is_recording_path = False # Controlled by main window via method

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.error_signal.emit("Error: Could not open camera.")
            self.running = False; return

        self.status_signal.emit(f"Camera started in {self.mode} mode.")

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret: time.sleep(0.1); continue

            frame = cv2.flip(frame, 1)
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = hands.process(rgb_frame)
            # Convert back to BGR for drawing with OpenCV
            # frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR) # Not needed if drawing on original frame
            frame.flags.writeable = True # Allow drawing on the frame

            static_landmarks_data = None
            path_point_data = None
            prediction_text = "" # For static prediction display

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                # Draw landmarks using MediaPipe drawing utils
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # --- Process for Static Gestures ---
                if self.mode in ['collect_static', 'classify_static']:
                    processed_static_landmarks = process_static_landmarks(hand_landmarks)
                    if processed_static_landmarks:
                        static_landmarks_data = processed_static_landmarks
                        # Classify Static Gesture
                        if self.mode == 'classify_static' and self.static_model and self.static_label_encoder:
                            try:
                                input_data = np.array([static_landmarks_data], dtype=np.float32)
                                prediction = self.static_model.predict(input_data, verbose=0)[0]
                                idx = np.argmax(prediction)
                                conf = prediction[idx]
                                label = self.static_label_encoder.inverse_transform([idx])[0]
                                prediction_text = f"Static: {label} ({conf:.2f})" # Only show static prediction here
                            except Exception as e: self.status_signal.emit(f"Static Classif. Error: {e}")

                # --- Process for Path Gestures ---
                if self.mode in ['collect_path', 'classify_path']:
                    lm = hand_landmarks.landmark[PATH_TRACKING_LANDMARK]
                    path_point_data = (lm.x, lm.y) # Coordinates relative to image (0.0-1.0)
                    # Emit point if recording (drawing is handled in main thread now)
                    if self.is_recording_path:
                        self.path_point_signal.emit(path_point_data)
                    # Draw tracking circle (optional, as path will be drawn)
                    # h, w, _ = frame.shape
                    # center_coordinates = (int(lm.x * w), int(lm.y * h))
                    # cv2.circle(frame, center_coordinates, 5, (255, 0, 0), -1) # Blue circle

            # --- Display Static Prediction Text ---
            if prediction_text: # Only display static prediction on frame
                 cv2.putText(frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # --- Emit Signals ---
            if self.mode == 'collect_static' and self.current_static_label and static_landmarks_data:
                self.static_landmarks_signal.emit([self.current_static_label] + static_landmarks_data)
                self.status_signal.emit(f"Captured static '{self.current_static_label}'. Press key or Stop.")
                self.current_static_label = None

            # Emit frame (QImage for Qt display)
            qt_image = self._convert_cv_qt(frame)
            self.frame_signal.emit(qt_image)

        cap.release()
        self.status_signal.emit("Camera stopped.")
        self.running = False

    def stop(self):
        self.running = False
        self.is_recording_path = False
        self.status_signal.emit("Stopping camera...")

    def set_static_label(self, label):
        if self.mode == 'collect_static' and label in STATIC_ALLOWED_LABELS:
            self.current_static_label = label
            self.status_signal.emit(f"Ready for static '{label}'. Show gesture.")
        elif self.mode != 'collect_static':
             self.status_signal.emit("Not in static data collection mode.")

    def set_path_recording(self, is_recording):
        """Informs the thread if path recording is active (for emitting points)."""
        # This method is primarily for the thread to know *when* to emit points
        self.is_recording_path = is_recording
        # Status updates are handled in the main thread based on key press/release

    def load_static_model(self, model, encoder):
        self.static_model = model
        self.static_label_encoder = encoder
        if model and encoder: self.status_signal.emit("Static model loaded.")
        else: self.error_signal.emit("Failed to load static model/encoder.")

    def load_path_model(self, model, encoder, scaler):
        self.path_model = model
        self.path_label_encoder = encoder
        self.path_scaler = scaler
        if model and encoder and scaler: self.status_signal.emit("Path model loaded.")
        else: self.error_signal.emit("Failed to load path model/encoder/scaler.")

    def _convert_cv_qt(self, cv_img):
        """Converts OpenCV image (BGR) to QImage (RGB888)."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return convert_to_Qt_format

# --- Training Threads --- (No changes needed in training threads)

class StaticTrainingThread(QThread):
    """Handles static gesture model training."""
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, csv_path, model_path, encoder_path, parent=None):
        super().__init__(parent)
        self.csv_path = csv_path
        self.model_path = model_path
        self.encoder_path = encoder_path

    def run(self):
        try:
            self.progress_signal.emit("Starting static model training...")
            # --- Load Data ---
            self.progress_signal.emit(f"Loading static data from {self.csv_path}...")
            if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
                raise FileNotFoundError(f"Static CSV '{self.csv_path}' not found or empty.")
            df = pd.read_csv(self.csv_path); df.dropna(inplace=True) # Drop rows with NaNs
            if df.empty: raise ValueError("Static CSV is empty after dropping NaNs.")
            self.progress_signal.emit(f"Static data loaded: {df.shape[0]} samples.")

            # --- Preprocess ---
            self.progress_signal.emit("Preprocessing static data...")
            X = df.iloc[:, 1:].values
            y_labels = df.iloc[:, 0].values
            label_encoder = LabelEncoder(); label_encoder.fit(STATIC_ALLOWED_LABELS)
            try: y_encoded = label_encoder.transform(y_labels)
            except ValueError as e: raise ValueError(f"Unknown static label(s) in CSV: {set(y_labels) - set(label_encoder.classes_)}. Error: {e}")
            y_categorical = to_categorical(y_encoded, num_classes=STATIC_NUM_CLASSES)
            with open(self.encoder_path, 'wb') as f: pickle.dump(label_encoder, f)
            self.progress_signal.emit(f"Static label encoder saved to {self.encoder_path}")

            if len(df) < 10 or df['label'].nunique() < 2: raise ValueError("Need >= 10 samples and >= 2 classes for static training.")
            try: X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical)
            except ValueError: X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42) # Fallback split
            self.progress_signal.emit(f"Static data split: Train={X_train.shape[0]}, Val={X_val.shape[0]}")

            # --- Define & Train Model ---
            self.progress_signal.emit("Building static Keras model...")
            model = Sequential([ Dense(128, activation='relu', input_shape=(STATIC_NUM_FEATURES,)), Dropout(0.3),
                                 Dense(64, activation='relu'), Dropout(0.3),
                                 Dense(STATIC_NUM_CLASSES, activation='softmax') ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.summary(print_fn=lambda x: self.progress_signal.emit(x))
            self.progress_signal.emit("Starting static model training...")
            early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
            history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping, reduce_lr], verbose=0)
            val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
            self.progress_signal.emit(f"Static training finished. Val Accuracy: {val_accuracy:.4f}")

            # --- Save Model ---
            self.progress_signal.emit(f"Saving static model to {self.model_path}...")
            model.save(self.model_path)
            self.finished_signal.emit(True, f"Static training complete. Model saved. Val Acc: {val_accuracy:.4f}")

        except (FileNotFoundError, ValueError) as e: self.finished_signal.emit(False, f"Static Training Error: {e}")
        except Exception as e: import traceback; traceback.print_exc(); self.finished_signal.emit(False, f"Unexpected static training error: {e}")

class PathTrainingThread(QThread):
    """Handles path gesture model training."""
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, csv_path, model_path, encoder_path, scaler_path, parent=None):
        super().__init__(parent)
        self.csv_path = csv_path
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.scaler_path = scaler_path

    def run(self):
        try:
            self.progress_signal.emit("Starting path model training...")
            # --- Load Data ---
            self.progress_signal.emit(f"Loading path data from {self.csv_path}...")
            if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
                raise FileNotFoundError(f"Path CSV '{self.csv_path}' not found or empty.")
            df = pd.read_csv(self.csv_path); df.dropna(inplace=True) # Drop rows with NaNs
            if df.empty: raise ValueError("Path CSV is empty after dropping NaNs.")
            self.progress_signal.emit(f"Path data loaded: {df.shape[0]} samples.")

            # --- Preprocess ---
            self.progress_signal.emit("Preprocessing path data...")
            X = df.iloc[:, 1:].values # Shape: (n_samples, PATH_NUM_FEATURES)
            y_labels = df.iloc[:, 0].values

            # Scale features (important for distance-based/gradient models)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            with open(self.scaler_path, 'wb') as f: pickle.dump(scaler, f)
            self.progress_signal.emit(f"Path feature scaler saved to {self.scaler_path}")

            # Encode labels using the global constant
            label_encoder = LabelEncoder(); label_encoder.fit(PATH_ALLOWED_LABELS)
            try: y_encoded = label_encoder.transform(y_labels)
            except ValueError as e: raise ValueError(f"Unknown path label(s) in CSV: {set(y_labels) - set(label_encoder.classes_)}. Error: {e}")
            # Use global constant for num_classes
            y_categorical = to_categorical(y_encoded, num_classes=PATH_NUM_CLASSES)
            with open(self.encoder_path, 'wb') as f: pickle.dump(label_encoder, f)
            self.progress_signal.emit(f"Path label encoder saved to {self.encoder_path}")

            if len(df) < 10 or df['label'].nunique() < 2: raise ValueError("Need >= 10 samples and >= 2 classes for path training.")
            try: X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical)
            except ValueError: X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42) # Fallback
            self.progress_signal.emit(f"Path data split: Train={X_train.shape[0]}, Val={X_val.shape[0]}")

            # --- Define & Train Model (Simple MLP for flattened path) ---
            self.progress_signal.emit("Building path Keras model (MLP)...")
            # Ensure output layer uses the correct number of classes
            model = Sequential([ Dense(256, activation='relu', input_shape=(PATH_NUM_FEATURES,)), Dropout(0.4),
                                 Dense(128, activation='relu'), Dropout(0.4),
                                 Dense(64, activation='relu'), Dropout(0.4),
                                 Dense(PATH_NUM_CLASSES, activation='softmax') ]) # Use PATH_NUM_CLASSES
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.summary(print_fn=lambda x: self.progress_signal.emit(x))
            self.progress_signal.emit("Starting path model training...")
            early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True) # Increased patience
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=0.00001)
            history = model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping, reduce_lr], verbose=0)
            val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
            self.progress_signal.emit(f"Path training finished. Val Accuracy: {val_accuracy:.4f}")

            # --- Save Model ---
            self.progress_signal.emit(f"Saving path model to {self.model_path}...")
            model.save(self.model_path)
            self.finished_signal.emit(True, f"Path training complete. Model saved. Val Acc: {val_accuracy:.4f}")

        except (FileNotFoundError, ValueError) as e: self.finished_signal.emit(False, f"Path Training Error: {e}")
        except Exception as e: import traceback; traceback.print_exc(); self.finished_signal.emit(False, f"Unexpected path training error: {e}")


# --- Main Application Window ---

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)

        # --- State Variables ---
        self.camera_thread = None
        self.static_training_thread = None
        self.path_training_thread = None
        self.current_mode = 'idle'

        # Static Gesture State
        self.static_model = None
        self.static_label_encoder = None
        self.static_data_counts = {}

        # Path Gesture State
        self.path_model = None
        self.path_label_encoder = None
        self.path_scaler = None
        self.path_data_counts = {}
        self.current_path_points = [] # Stores (x,y) tuples relative to image (0.0-1.0)
        self.is_recording_path = False
        # Initialize selected_path_label using the updated PATH_ALLOWED_LABELS
        self.selected_path_label = PATH_ALLOWED_LABELS[0] if PATH_ALLOWED_LABELS else None
        self.current_recording_path_label = None # Label assigned when 'Q' is pressed

        # --- GUI Elements ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Top Row: Camera Feed and Controls
        self.top_layout = QHBoxLayout()
        self.layout.addLayout(self.top_layout)

        # Camera Feed Label
        self.camera_label = QLabel("Camera Feed")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setFixedSize(CAMERA_FEED_WIDTH, CAMERA_FEED_HEIGHT)
        self.camera_label.setStyleSheet("border: 1px solid black; background-color: #f0f0f0;")
        self.top_layout.addWidget(self.camera_label)

        # Controls Panel (Scrollable)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.controls_widget = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_widget)
        self.scroll_area.setWidget(self.controls_widget)
        self.top_layout.addWidget(self.scroll_area)

        # --- Static Gesture Controls ---
        self.static_collect_label = QLabel("1. Static Gesture Collection")
        self.static_collect_label.setFont(QFont("Arial", 11, QFont.Bold))
        self.controls_layout.addWidget(self.static_collect_label)
        self.start_static_collect_button = QPushButton("Start Collecting Static")
        self.start_static_collect_button.clicked.connect(self.start_static_collection_mode)
        self.controls_layout.addWidget(self.start_static_collect_button)
        self.stop_static_collect_button = QPushButton("Stop Collecting Static")
        self.stop_static_collect_button.clicked.connect(self.stop_camera)
        self.controls_layout.addWidget(self.stop_static_collect_button)
        self.static_collect_instructions = QLabel(f"Press keys ({', '.join(STATIC_ALLOWED_LABELS)})")
        self.controls_layout.addWidget(self.static_collect_instructions)
        self.static_data_count_label = QLabel("Static Samples: 0")
        self.controls_layout.addWidget(self.static_data_count_label)

        self.static_train_label = QLabel("2. Static Model Training")
        self.static_train_label.setFont(QFont("Arial", 11, QFont.Bold))
        self.controls_layout.addWidget(self.static_train_label)
        self.static_train_button = QPushButton("Train Static Model")
        self.static_train_button.clicked.connect(self.start_static_training)
        self.controls_layout.addWidget(self.static_train_button)

        self.static_classify_label = QLabel("3. Static Classification")
        self.static_classify_label.setFont(QFont("Arial", 11, QFont.Bold))
        self.controls_layout.addWidget(self.static_classify_label)
        self.start_static_classify_button = QPushButton("Start Real-time Static Classif.")
        self.start_static_classify_button.clicked.connect(self.start_static_classification_mode)
        self.controls_layout.addWidget(self.start_static_classify_button)
        self.stop_static_classify_button = QPushButton("Stop Static Classif.")
        self.stop_static_classify_button.clicked.connect(self.stop_camera)
        self.controls_layout.addWidget(self.stop_static_classify_button)
        self.classify_static_image_button = QPushButton("Classify Static from Image")
        self.classify_static_image_button.clicked.connect(self.classify_static_from_image)
        self.controls_layout.addWidget(self.classify_static_image_button)

        self.controls_layout.addSpacing(20)

        # --- Path Gesture Controls ---
        self.path_collect_label = QLabel("4. Path Gesture Collection")
        self.path_collect_label.setFont(QFont("Arial", 11, QFont.Bold))
        self.controls_layout.addWidget(self.path_collect_label)
        self.start_path_collect_button = QPushButton("Start Collecting Paths")
        self.start_path_collect_button.clicked.connect(self.start_path_collection_mode)
        self.controls_layout.addWidget(self.start_path_collect_button)
        self.stop_path_collect_button = QPushButton("Stop Collecting Paths")
        self.stop_path_collect_button.clicked.connect(self.stop_camera)
        self.controls_layout.addWidget(self.stop_path_collect_button)

        # Path Label Selection ComboBox - Populated with updated labels
        self.path_label_select_layout = QHBoxLayout()
        self.path_label_select_label = QLabel("Select Path Label:")
        self.path_label_select_combo = QComboBox()
        self.path_label_select_combo.addItems(PATH_ALLOWED_LABELS) # Use updated list
        if PATH_ALLOWED_LABELS: # Set initial selected label from updated list
             self.selected_path_label = PATH_ALLOWED_LABELS[0]
             self.path_label_select_combo.setCurrentText(self.selected_path_label)
        self.path_label_select_combo.currentTextChanged.connect(self._update_selected_path_label)
        self.path_label_select_layout.addWidget(self.path_label_select_label)
        self.path_label_select_layout.addWidget(self.path_label_select_combo)
        self.controls_layout.addLayout(self.path_label_select_layout)

        # Updated Instructions
        # Use QKeySequence to display the actual key name (e.g., 'Q')
        record_key_name = QKeySequence(PATH_RECORD_KEY).toString(QKeySequence.NativeText)
        self.path_collect_instructions = QLabel(f"Select label above, then Press and Hold '{record_key_name}' to record path.")
        self.controls_layout.addWidget(self.path_collect_instructions)
        self.path_data_count_label = QLabel("Path Samples: 0")
        self.controls_layout.addWidget(self.path_data_count_label)

        self.path_train_label = QLabel("5. Path Model Training")
        self.path_train_label.setFont(QFont("Arial", 11, QFont.Bold))
        self.controls_layout.addWidget(self.path_train_label)
        self.path_train_button = QPushButton("Train Path Model")
        self.path_train_button.clicked.connect(self.start_path_training)
        self.controls_layout.addWidget(self.path_train_button)

        self.path_classify_label = QLabel("6. Path Classification")
        self.path_classify_label.setFont(QFont("Arial", 11, QFont.Bold))
        self.controls_layout.addWidget(self.path_classify_label)
        self.start_path_classify_button = QPushButton("Start Real-time Path Classif.")
        self.start_path_classify_button.clicked.connect(self.start_path_classification_mode)
        self.controls_layout.addWidget(self.start_path_classify_button)
        self.stop_path_classify_button = QPushButton("Stop Path Classif.")
        self.stop_path_classify_button.clicked.connect(self.stop_camera)
        self.controls_layout.addWidget(self.stop_path_classify_button)
        # Updated Instructions
        self.path_classify_instructions = QLabel(f"Press and Hold '{record_key_name}' to record path for classification.")
        self.controls_layout.addWidget(self.path_classify_instructions)
        self.path_prediction_label = QLabel("Path Prediction: ---")
        self.controls_layout.addWidget(self.path_prediction_label)

        self.controls_layout.addStretch()

        # Status Bar
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Ready.")

        # Load initial state
        self._load_initial_counts() # Will use updated PATH_ALLOWED_LABELS
        self._load_models_and_encoders() # Will use updated PATH_NUM_CLASSES
        self._update_button_states()
        self._update_data_count_display()

    # --- Data Count & Loading ---
    def _load_initial_counts(self):
        """Loads counts for both static and path data."""
        # Static Counts
        self.static_data_counts = {label: 0 for label in STATIC_ALLOWED_LABELS}
        if os.path.exists(STATIC_CSV_FILE) and os.path.getsize(STATIC_CSV_FILE) > 0:
            try:
                df = pd.read_csv(STATIC_CSV_FILE)
                if 'label' in df.columns:
                    counts = df['label'].value_counts().to_dict()
                    for label, count in counts.items():
                        if label in self.static_data_counts: self.static_data_counts[label] = count
            except Exception as e: self.show_error("Static CSV Load Error", f"{e}")
        # Path Counts - uses updated PATH_ALLOWED_LABELS
        self.path_data_counts = {label: 0 for label in PATH_ALLOWED_LABELS}
        if os.path.exists(PATH_CSV_FILE) and os.path.getsize(PATH_CSV_FILE) > 0:
            try:
                df = pd.read_csv(PATH_CSV_FILE)
                if 'label' in df.columns:
                    counts = df['label'].value_counts().to_dict()
                    for label, count in counts.items():
                        if label in self.path_data_counts: self.path_data_counts[label] = count
            except Exception as e: self.show_error("Path CSV Load Error", f"{e}")

    def _update_data_count_display(self):
        """Updates both static and path count labels."""
        total_static = sum(self.static_data_counts.values())
        self.static_data_count_label.setText(f"Static Samples: {total_static}")
        total_path = sum(self.path_data_counts.values())
        self.path_data_count_label.setText(f"Path Samples: {total_path}")

    def _load_models_and_encoders(self):
        """Loads both static and path models/encoders/scalers."""
        static_loaded = False
        path_loaded = False
        # Load Static
        if os.path.exists(STATIC_MODEL_FILE) and os.path.exists(STATIC_ENCODER_FILE):
            try:
                self.static_model = load_model(STATIC_MODEL_FILE)
                with open(STATIC_ENCODER_FILE, 'rb') as f: self.static_label_encoder = pickle.load(f)
                # Check against current constants
                if len(self.static_label_encoder.classes_) == STATIC_NUM_CLASSES and self.static_model.output_shape[-1] == STATIC_NUM_CLASSES:
                    static_loaded = True
                else: self.show_warning("Static Model Mismatch", f"Loaded static model/encoder class count ({len(self.static_label_encoder.classes_)}) doesn't match expected ({STATIC_NUM_CLASSES}). Please retrain."); self.static_model = self.static_label_encoder = None
            except Exception as e: self.show_error("Static Load Error", f"{e}"); self.static_model = self.static_label_encoder = None
        # Load Path
        if os.path.exists(PATH_MODEL_FILE) and os.path.exists(PATH_ENCODER_FILE) and os.path.exists(PATH_SCALER_FILE):
             try:
                self.path_model = load_model(PATH_MODEL_FILE)
                with open(PATH_ENCODER_FILE, 'rb') as f: self.path_label_encoder = pickle.load(f)
                with open(PATH_SCALER_FILE, 'rb') as f: self.path_scaler = pickle.load(f)
                # Check against current constants
                if len(self.path_label_encoder.classes_) == PATH_NUM_CLASSES and self.path_model.output_shape[-1] == PATH_NUM_CLASSES:
                     path_loaded = True
                else: self.show_warning("Path Model Mismatch", f"Loaded path model/encoder class count ({len(self.path_label_encoder.classes_)}) doesn't match expected ({PATH_NUM_CLASSES}). Please retrain."); self.path_model = self.path_label_encoder = self.path_scaler = None
             except Exception as e: self.show_error("Path Load Error", f"{e}"); self.path_model = self.path_label_encoder = self.path_scaler = None

        status_parts = []
        if static_loaded: status_parts.append("Static model loaded.")
        if path_loaded: status_parts.append("Path model loaded.")
        if not status_parts: self.statusBar().showMessage("No models loaded. Collect data and train.")
        else: self.statusBar().showMessage(" ".join(status_parts))

        # Ensure models are None if loading failed (already handled in try/except)
        self._update_button_states()
        return static_loaded, path_loaded

    # --- Mode Control & Button States ---
    def _set_mode(self, new_mode):
        """Sets the application mode and updates UI accordingly."""
        if self.camera_thread and self.camera_thread.isRunning():
             self.show_message("Camera Busy", "Stop the current camera process first.")
             return False

        self.current_mode = new_mode
        self.is_recording_path = False # Reset path recording state
        self.current_path_points = []
        self.current_recording_path_label = None # Reset recording label
        self.path_prediction_label.setText("Path Prediction: ---")

        if new_mode == 'idle':
            self.stop_camera()
        else:
            self.camera_thread = CameraThread(mode=new_mode)
            self._connect_camera_signals()
            if new_mode == 'classify_static':
                if not self.static_model or not self.static_label_encoder:
                    self.show_message("Model Error", "Static model not loaded."); self.current_mode = 'idle'; return False
                self.camera_thread.load_static_model(self.static_model, self.static_label_encoder)
            elif new_mode == 'classify_path':
                if not self.path_model or not self.path_label_encoder or not self.path_scaler:
                    self.show_message("Model Error", "Path model/scaler not loaded."); self.current_mode = 'idle'; return False
                self.camera_thread.load_path_model(self.path_model, self.path_label_encoder, self.path_scaler)

            self.camera_thread.start()
            self.statusBar().showMessage(f"Mode set to: {new_mode}")
            if new_mode in ['collect_static', 'collect_path', 'classify_path']:
                 self.setFocus()

        self._update_button_states()
        return True

    def _update_button_states(self):
        """Enables/disables buttons based on the current mode and model availability."""
        mode = self.current_mode
        is_idle = mode == 'idle'
        is_collecting_static = mode == 'collect_static'
        is_classifying_static = mode == 'classify_static'
        is_collecting_path = mode == 'collect_path'
        is_classifying_path = mode == 'classify_path'
        is_training = (self.static_training_thread and self.static_training_thread.isRunning()) or \
                      (self.path_training_thread and self.path_training_thread.isRunning())
        static_model_loaded = self.static_model is not None and self.static_label_encoder is not None
        path_model_loaded = self.path_model is not None and self.path_label_encoder is not None and self.path_scaler is not None

        # Static Buttons
        self.start_static_collect_button.setEnabled(is_idle and not is_training)
        self.stop_static_collect_button.setEnabled(is_collecting_static)
        self.static_train_button.setEnabled(is_idle and not is_training)
        self.start_static_classify_button.setEnabled(is_idle and not is_training and static_model_loaded)
        self.stop_static_classify_button.setEnabled(is_classifying_static)
        self.classify_static_image_button.setEnabled(is_idle and not is_training and static_model_loaded)

        # Path Buttons
        self.start_path_collect_button.setEnabled(is_idle and not is_training)
        self.stop_path_collect_button.setEnabled(is_collecting_path)
        self.path_train_button.setEnabled(is_idle and not is_training)
        self.start_path_classify_button.setEnabled(is_idle and not is_training and path_model_loaded)
        self.stop_path_classify_button.setEnabled(is_classifying_path)

        # Path Label Selector - Enable only when idle or in path modes
        self.path_label_select_combo.setEnabled(is_idle or is_collecting_path or is_classifying_path)

        # Disable all start buttons if any camera is running
        is_camera_running = self.camera_thread is not None and self.camera_thread.isRunning()
        if is_camera_running:
            self.start_static_collect_button.setEnabled(False)
            self.start_static_classify_button.setEnabled(False)
            self.start_path_collect_button.setEnabled(False)
            self.start_path_classify_button.setEnabled(False)
            self.classify_static_image_button.setEnabled(False)
        if is_training: # Also disable start buttons if training
            self.start_static_collect_button.setEnabled(False)
            self.start_static_classify_button.setEnabled(False)
            self.start_path_collect_button.setEnabled(False)
            self.start_path_classify_button.setEnabled(False)
            self.classify_static_image_button.setEnabled(False)
            self.static_train_button.setEnabled(False)
            self.path_train_button.setEnabled(False)


    def stop_camera(self):
        """Stops the camera thread regardless of mode."""
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
            # self.camera_thread.wait(500)
            self.camera_thread = None
        self.current_mode = 'idle'
        self.is_recording_path = False
        self.current_path_points = [] # Clear points when stopping camera
        self.camera_label.setText("Camera Stopped.")
        self.camera_label.setPixmap(QPixmap())
        self.statusBar().showMessage("Camera stopped. Mode set to Idle.")
        self._update_button_states()

    # --- Signal Connections & Handling ---
    def _connect_camera_signals(self):
        """Connects signals from the camera thread."""
        if not self.camera_thread: return
        # Disconnect previous connections first
        try: self.camera_thread.frame_signal.disconnect()
        except TypeError: pass
        try: self.camera_thread.static_landmarks_signal.disconnect()
        except TypeError: pass
        try: self.camera_thread.path_point_signal.disconnect()
        except TypeError: pass
        try: self.camera_thread.status_signal.disconnect()
        except TypeError: pass
        try: self.camera_thread.error_signal.disconnect()
        except TypeError: pass
        try: self.camera_thread.finished.disconnect()
        except TypeError: pass

        # Connect new signals
        self.camera_thread.frame_signal.connect(self.update_frame)
        self.camera_thread.static_landmarks_signal.connect(self.save_static_landmarks)
        self.camera_thread.path_point_signal.connect(self.append_path_point) # Append point live
        self.camera_thread.status_signal.connect(self.update_status)
        self.camera_thread.error_signal.connect(self.show_error)
        self.camera_thread.finished.connect(self._on_camera_thread_finished)

    def _on_camera_thread_finished(self):
        """Handles camera thread finishing."""
        if self.current_mode != 'idle':
             self.statusBar().showMessage("Camera thread finished unexpectedly.")
             self.current_mode = 'idle'
        self.camera_thread = None
        self._update_button_states()

    # --- Static Gesture Methods ---
    def start_static_collection_mode(self): self._set_mode('collect_static')
    def start_static_classification_mode(self): self._set_mode('classify_static')

    def save_static_landmarks(self, data):
        if len(data) > 1:
            label = data[0]; landmarks = data[1:]
            if save_static_to_csv(STATIC_CSV_FILE, label, landmarks):
                self.static_data_counts[label] = self.static_data_counts.get(label, 0) + 1
                self._update_data_count_display()
            else: self.show_error("Save Error", f"Failed to save static data for '{label}'.")

    def start_static_training(self):
        if self.static_training_thread and self.static_training_thread.isRunning():
            self.show_message("Training Busy", "Static training already running.")
            return
        if self.current_mode != 'idle':
            self.show_message("Action Required", "Stop camera before static training.")
            return
        if not os.path.exists(STATIC_CSV_FILE) or os.path.getsize(STATIC_CSV_FILE) == 0:
            self.show_error("Training Error", f"Static CSV '{STATIC_CSV_FILE}' not found/empty.")
            return

        self.statusBar().showMessage("Starting static training...")
        self.static_training_thread = StaticTrainingThread(STATIC_CSV_FILE, STATIC_MODEL_FILE, STATIC_ENCODER_FILE)
        self.static_training_thread.progress_signal.connect(self.update_status)
        self.static_training_thread.finished_signal.connect(self.on_static_training_finished)
        self.static_training_thread.start()
        self._update_button_states()

    def on_static_training_finished(self, success, message):
        if success:
            self.show_message("Static Training Complete", message)
            self._load_models_and_encoders()
        else: self.show_error("Static Training Failed", message)
        self.static_training_thread = None
        self._update_button_states()
        self.statusBar().showMessage("Static training finished. Ready.")

    def classify_static_from_image(self):
        """Classifies static gesture from an image file."""
        if not self.static_model or not self.static_label_encoder:
            self.show_message("Model Not Loaded", "Static model not available.")
            return

        options = QFileDialog.Options(); options |= QFileDialog.DontUseNativeDialog
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        if not filepath: return

        self.statusBar().showMessage(f"Classifying static gesture in image...")
        try:
            img = cv2.imread(filepath)
            if img is None: raise ValueError("Could not read image.")
            img_display = img.copy()
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb_image.flags.writeable = False
            results = hands.process(rgb_image)
            rgb_image.flags.writeable = True

            prediction_text = "No hand detected."
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                processed_landmarks = process_static_landmarks(hand_landmarks)
                if processed_landmarks:
                    input_data = np.array([processed_landmarks], dtype=np.float32)
                    prediction = self.static_model.predict(input_data, verbose=0)[0]
                    idx = np.argmax(prediction); conf = prediction[idx]
                    label = self.static_label_encoder.inverse_transform([idx])[0]
                    prediction_text = f"Static: {label} (Conf: {conf:.2f})"
                    mp_drawing.draw_landmarks(img_display, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing_styles.get_default_hand_landmarks_style(),
                                              mp_drawing_styles.get_default_hand_connections_style())
                else: prediction_text = "Hand detected, landmarks invalid."

            cv2.putText(img_display, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            qt_image = self._convert_cv_qt(img_display)
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(self.camera_label.width(), self.camera_label.height(), Qt.KeepAspectRatio)
            self.camera_label.setPixmap(scaled_pixmap)
            self.statusBar().showMessage(f"Image classification: {prediction_text}")

        except Exception as e: self.show_error("Static Image Classification Error", f"{e}"); self.statusBar().showMessage("Static image classification failed.")

    # --- Path Gesture Methods ---
    def start_path_collection_mode(self): self._set_mode('collect_path')
    def start_path_classification_mode(self): self._set_mode('classify_path')

    def _update_selected_path_label(self, label):
        """Slot to update the selected path label from the ComboBox."""
        self.selected_path_label = label
        # print(f"Selected path label: {self.selected_path_label}") # For debugging

    def append_path_point(self, point):
        """Appends a tracked point (0.0-1.0) to the current path list."""
        if self.is_recording_path:
            self.current_path_points.append(point)
            # Trigger a repaint of the camera label to show the updated path
            self.camera_label.update()


    def process_and_save_path(self):
        """Processes the completed path, standardizes, and saves it."""
        if not self.current_recording_path_label: # Check the label assigned during recording
             print("Warning: Path recorded without a label assigned.")
             return
        if len(self.current_path_points) < 5:
            self.statusBar().showMessage(f"Path '{self.current_recording_path_label}' too short, discarded.")
            return

        self.statusBar().showMessage(f"Processing path '{self.current_recording_path_label}' ({len(self.current_path_points)} points)...")
        standardized_flat = standardize_path(self.current_path_points)

        if standardized_flat is not None:
            # Use the label that was active when recording started
            if save_path_to_csv(PATH_CSV_FILE, self.current_recording_path_label, standardized_flat):
                self.path_data_counts[self.current_recording_path_label] = self.path_data_counts.get(self.current_recording_path_label, 0) + 1
                self._update_data_count_display()
                self.statusBar().showMessage(f"Path '{self.current_recording_path_label}' saved.")
            else:
                self.show_error("Save Error", f"Failed to save path data for '{self.current_recording_path_label}'.")
        else:
            self.show_warning("Path Processing Error", f"Could not standardize path for '{self.current_recording_path_label}'.")

    def classify_live_path(self):
        """Processes the completed path and classifies it using the path model."""
        if not self.path_model or not self.path_label_encoder or not self.path_scaler:
             self.statusBar().showMessage("Path model/encoder/scaler not loaded for classification.")
             return
        if len(self.current_path_points) < 5:
            self.statusBar().showMessage("Path too short to classify.")
            self.path_prediction_label.setText("Path Prediction: Too short")
            return

        self.statusBar().showMessage(f"Classifying path ({len(self.current_path_points)} points)...")
        standardized_flat = standardize_path(self.current_path_points)

        if standardized_flat is not None:
            try:
                input_data = self.path_scaler.transform(standardized_flat.reshape(1, -1))
                prediction = self.path_model.predict(input_data, verbose=0)[0]
                idx = np.argmax(prediction); conf = prediction[idx]
                label = self.path_label_encoder.inverse_transform([idx])[0]
                pred_text = f"Path Prediction: {label} ({conf:.2f})"
                self.path_prediction_label.setText(pred_text)
                self.statusBar().showMessage(f"Classification result: {label} ({conf:.2f})")
            except Exception as e:
                self.show_error("Path Classification Error", f"{e}")
                self.path_prediction_label.setText("Path Prediction: Error")
                self.statusBar().showMessage("Path classification error.")
        else:
            self.show_warning("Path Processing Error", "Could not standardize path for classification.")
            self.path_prediction_label.setText("Path Prediction: Proc. Error")
            self.statusBar().showMessage("Path processing error during classification.")


    def start_path_training(self):
        if self.path_training_thread and self.path_training_thread.isRunning():
            self.show_message("Training Busy", "Path training already running.")
            return
        if self.current_mode != 'idle':
            self.show_message("Action Required", "Stop camera before path training.")
            return
        if not os.path.exists(PATH_CSV_FILE) or os.path.getsize(PATH_CSV_FILE) == 0:
            self.show_error("Training Error", f"Path CSV '{PATH_CSV_FILE}' not found/empty.")
            return

        self.statusBar().showMessage("Starting path training...")
        self.path_training_thread = PathTrainingThread(PATH_CSV_FILE, PATH_MODEL_FILE, PATH_ENCODER_FILE, PATH_SCALER_FILE)
        self.path_training_thread.progress_signal.connect(self.update_status)
        self.path_training_thread.finished_signal.connect(self.on_path_training_finished)
        self.path_training_thread.start()
        self._update_button_states()

    def on_path_training_finished(self, success, message):
        if success:
            self.show_message("Path Training Complete", message)
            self._load_models_and_encoders()
        else: self.show_error("Path Training Failed", message)
        self.path_training_thread = None
        self._update_button_states()
        self.statusBar().showMessage("Path training finished. Ready.")

    # --- GUI Update Slots ---
    def update_frame(self, q_image):
        """Updates the camera feed label, drawing the path if recording."""
        pixmap = QPixmap.fromImage(q_image)

        # Draw the path overlay if recording
        if self.is_recording_path and len(self.current_path_points) > 1:
            painter = QPainter(pixmap)
            pen = QPen(QColor(0, 0, 255), 3) # Blue color, thickness 3
            painter.setPen(pen)

            # Convert normalized points (0.0-1.0) to pixmap coordinates
            w, h = pixmap.width(), pixmap.height()
            path_points_scaled = [QPoint(int(p[0] * w), int(p[1] * h)) for p in self.current_path_points]

            # Draw lines between consecutive points
            for i in range(len(path_points_scaled) - 1):
                painter.drawLine(path_points_scaled[i], path_points_scaled[i+1])

            painter.end()

        # Scale pixmap before setting to fit the label
        scaled_pixmap = pixmap.scaled(
             self.camera_label.width(), self.camera_label.height(), Qt.KeepAspectRatio
        )
        self.camera_label.setPixmap(scaled_pixmap)


    def update_status(self, message): self.statusBar().showMessage(message)
    def show_error(self, title, message): QMessageBox.critical(self, title, message); self.statusBar().showMessage(f"Error: {message[:100]}")
    def show_warning(self, title, message): QMessageBox.warning(self, title, message); self.statusBar().showMessage(f"Warning: {message[:100]}")
    def show_message(self, title, message): QMessageBox.information(self, title, message)

    # --- Event Handling ---
    def keyPressEvent(self, event):
        """Handles key presses for static gestures and path recording trigger."""
        if event.isAutoRepeat(): return

        key = event.key()
        key_text = event.text().upper()

        # --- Static Gesture Collection ---
        if self.current_mode == 'collect_static' and key_text in STATIC_ALLOWED_LABELS:
            if self.camera_thread: self.camera_thread.set_static_label(key_text)

        # --- Path Gesture Collection / Classification ---
        elif self.current_mode in ['collect_path', 'classify_path'] and key == PATH_RECORD_KEY:
            if not self.is_recording_path: # Start recording on 'Q' press
                # Check if a valid label is selected in the dropdown
                if not self.selected_path_label:
                    self.show_warning("No Path Label", "Please select a path label from the dropdown first.")
                    return

                self.is_recording_path = True
                # Assign the currently selected label from dropdown for saving/processing later
                self.current_recording_path_label = self.selected_path_label
                self.current_path_points = [] # Clear previous points
                if self.camera_thread: self.camera_thread.set_path_recording(True)
                self.path_prediction_label.setText("Path Prediction: Recording...")
                self.statusBar().showMessage(f"Recording path for '{self.current_recording_path_label}'...")

        else: super().keyPressEvent(event) # Pass other keys up

    def keyReleaseEvent(self, event):
        """Handles key releases for path recording trigger."""
        if event.isAutoRepeat(): return

        key = event.key()

        # --- Path Gesture Collection / Classification ---
        # Stop recording ONLY if 'Q' is released and we were recording
        if self.current_mode in ['collect_path', 'classify_path'] and \
           key == PATH_RECORD_KEY and self.is_recording_path:

            self.is_recording_path = False
            if self.camera_thread: self.camera_thread.set_path_recording(False)
            # Check if a label was assigned when recording started
            if self.current_recording_path_label:
                self.statusBar().showMessage(f"Finished recording path for '{self.current_recording_path_label}'. Processing...")

                # Process the completed path
                if self.current_mode == 'collect_path':
                    self.process_and_save_path()
                elif self.current_mode == 'classify_path':
                    self.classify_live_path()
            else:
                # This case shouldn't happen with the check in keyPressEvent, but good to have
                self.statusBar().showMessage("Finished recording path (no label assigned). Discarded.")


            # Clear points after processing
            self.current_path_points = []
            # Clear the label associated with the finished recording
            self.current_recording_path_label = None

            # Trigger one last repaint to clear the path from the screen
            self.camera_label.update()


        else: super().keyReleaseEvent(event)


    def closeEvent(self, event):
        self.stop_camera()
        if (self.static_training_thread and self.static_training_thread.isRunning()) or \
           (self.path_training_thread and self.path_training_thread.isRunning()):
             print("Warning: Training is running. Closing anyway.")
        hands.close()
        print("MediaPipe Hands closed.")
        event.accept()

# --- Main Execution ---
if __name__ == '__main__':
    # Import QKeySequence here if needed for PATH_RECORD_KEY display name
    from PyQt5.QtGui import QKeySequence
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
