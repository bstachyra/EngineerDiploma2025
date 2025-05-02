import sys
import os
import csv
import cv2 # OpenCV for image/video handling
import mediapipe as mp # For hand landmark detection
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QComboBox, QProgressBar, QTextEdit, QTabWidget)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- Constants ---
APP_NAME = "ASL Hand Sign Recognizer"
CSV_FILE = 'hand_landmarks.csv'
MODEL_FILE = 'asl_model.keras' # Use .keras extension for Keras 3+
NUM_LANDMARKS = 21
NUM_FEATURES = NUM_LANDMARKS * 3 # x, y, z for each landmark
LABELS = [chr(ord('A') + i) for i in range(26)] # A-Z

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=True,       # Process static images
    max_num_hands=1,              # Detect only one hand
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

hands_video = mp_hands.Hands(
    static_image_mode=False,      # Process video stream
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# --- Helper Functions ---
def extract_landmarks(image_rgb):
    """Extracts landmarks from a single RGB image."""
    results = hands.process(image_rgb)
    landmarks = []
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0] # Get first detected hand
        # Normalize landmarks relative to the wrist (landmark 0)
        # This helps make the model more robust to hand position/scale
        wrist = hand_landmarks.landmark[0]
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
        # Optional: Further normalization (e.g., scale by hand size) could be added
    return landmarks

def extract_landmarks_video(image_rgb):
    """Extracts landmarks for video/real-time feed."""
    results = hands_video.process(image_rgb)
    landmarks = []
    hand_landmarks_for_drawing = None
    if results.multi_hand_landmarks:
        hand_landmarks_for_drawing = results.multi_hand_landmarks[0]
        wrist = hand_landmarks_for_drawing.landmark[0]
        for lm in hand_landmarks_for_drawing.landmark:
            landmarks.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    return landmarks, hand_landmarks_for_drawing

def draw_landmarks(image, hand_landmarks):
    """Draws landmarks on the image."""
    if hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS)
    return image

# --- Worker Threads ---
class DataExtractionThread(QThread):
    """Worker thread for extracting landmarks from multiple images."""
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, image_files, labels):
        super().__init__()
        self.image_files = image_files
        self.labels = labels
        self.is_running = True

    def run(self):
        try:
            total_files = len(self.image_files)
            header = ['label'] + [f'{axis}_{i}' for i in range(NUM_LANDMARKS) for axis in ['x', 'y', 'z']]
            file_exists = os.path.isfile(CSV_FILE)

            with open(CSV_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists or os.path.getsize(CSV_FILE) == 0:
                    writer.writerow(header) # Write header only if file is new/empty

                for i, (file_path, label) in enumerate(zip(self.image_files, self.labels)):
                    if not self.is_running:
                        break
                    image = cv2.imread(file_path)
                    if image is None:
                        print(f"Warning: Could not read image {file_path}")
                        continue
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    landmarks = extract_landmarks(image_rgb)

                    if len(landmarks) == NUM_FEATURES:
                        writer.writerow([label] + landmarks)
                    else:
                         print(f"Warning: No hand or incomplete landmarks found in {file_path}")

                    self.progress.emit(int((i + 1) / total_files * 100))

            if self.is_running:
                self.finished.emit(f"Landmark data saved to {CSV_FILE}")
            else:
                self.finished.emit("Data extraction cancelled.")
        except Exception as e:
            self.error.emit(f"Error during data extraction: {e}")

    def stop(self):
        self.is_running = False

class TrainingThread(QThread):
    """Worker thread for training the TensorFlow model."""
    progress = pyqtSignal(str) # Send text updates
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.is_running = True

    def run(self):
        try:
            self.progress.emit("Loading data...")
            if not os.path.exists(CSV_FILE):
                self.error.emit(f"Error: {CSV_FILE} not found. Please extract landmark data first.")
                return

            data = pd.read_csv(CSV_FILE)
            if data.empty:
                 self.error.emit(f"Error: {CSV_FILE} is empty.")
                 return

            # Check if data has the expected number of columns
            expected_columns = 1 + NUM_FEATURES # label + features
            if data.shape[1] != expected_columns:
                 self.error.emit(f"Error: CSV file has {data.shape[1]} columns, expected {expected_columns}.")
                 return

            labels = data['label'].values
            features = data.drop('label', axis=1).values

            # --- Data Preprocessing ---
            self.progress.emit("Preprocessing data...")
            # 1. Encode Labels (Letter to Integer)
            label_encoder = LabelEncoder()
            integer_encoded = label_encoder.fit_transform(labels)

            # 2. One-Hot Encode Labels
            onehot_encoder = OneHotEncoder(sparse_output=False) # Use sparse_output instead of sparse
            integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
            onehot_encoded_labels = onehot_encoder.fit_transform(integer_encoded)

            # 3. Split Data
            X_train, X_test, y_train, y_test = train_test_split(
                features, onehot_encoded_labels, test_size=0.2, random_state=42, stratify=onehot_encoded_labels
            ) # Stratify ensures class distribution is similar in train/test

            # --- Model Definition ---
            self.progress.emit("Building model...")
            model = Sequential([
                Dense(128, activation='relu', input_shape=(NUM_FEATURES,)),
                Dropout(0.3), # Add dropout for regularization
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(len(LABELS), activation='softmax') # Output layer: size = number of classes
            ])

            model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            model.summary(print_fn=lambda x: self.progress.emit(x)) # Print summary to progress log

            # --- Callbacks ---
            # Stop training early if validation loss doesn't improve
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            # Save the best model found during training
            model_checkpoint = ModelCheckpoint(MODEL_FILE, monitor='val_accuracy', save_best_only=True, mode='max')

            # --- Training ---
            self.progress.emit("Starting training...")
            history = model.fit(X_train, y_train,
                                epochs=100, # Adjust epochs as needed
                                batch_size=32,
                                validation_data=(X_test, y_test),
                                callbacks=[early_stopping, model_checkpoint],
                                verbose=0) # Verbose=0 to avoid printing logs directly to console

            # --- Evaluation ---
            self.progress.emit("Evaluating model...")
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            self.progress.emit(f"Test Accuracy: {accuracy:.4f}")
            self.progress.emit(f"Test Loss: {loss:.4f}")

            # No need to save again if ModelCheckpoint saved the best one
            # model.save(MODEL_FILE)
            self.progress.emit(f"Best model saved to {MODEL_FILE}")

            # Save the label encoder classes for later use during prediction
            np.save('label_classes.npy', label_encoder.classes_)
            self.progress.emit("Label encoder classes saved.")

            self.finished.emit(f"Training complete. Model saved. Test Accuracy: {accuracy:.4f}")

        except Exception as e:
            self.error.emit(f"Error during training: {e}")

    def stop(self):
        # Note: Stopping TensorFlow training mid-epoch is non-trivial.
        # This provides a basic flag, but TF might finish the current step.
        self.is_running = False
        self.progress.emit("Training cancellation requested...")


class RecognitionThread(QThread):
    """Worker thread for real-time recognition."""
    frame_processed = pyqtSignal(QImage, str) # Send processed frame and prediction
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, source=0, model_path=MODEL_FILE, labels_path='label_classes.npy'):
        super().__init__()
        self.source = source # 0 for webcam, or video file path
        self.model_path = model_path
        self.labels_path = labels_path
        self.is_running = True
        self.model = None
        self.label_encoder_classes = None

    def run(self):
        try:
            # Load the trained model
            if not os.path.exists(self.model_path):
                self.error.emit(f"Model file not found: {self.model_path}")
                return
            self.model = load_model(self.model_path)

            # Load the label encoder classes
            if not os.path.exists(self.labels_path):
                 self.error.emit(f"Label classes file not found: {self.labels_path}")
                 return
            self.label_encoder_classes = np.load(self.labels_path, allow_pickle=True)

            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                self.error.emit(f"Error opening video source: {self.source}")
                return

            while self.is_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    if self.source != 0: # If it's a file, stop at the end
                         self.is_running = False
                    continue # Skip empty frames (e.g., end of video file)

                # Flip frame horizontally for a later selfie-view display
                # This makes movement feel more natural on webcam
                if self.source == 0:
                    frame = cv2.flip(frame, 1)

                # Convert the BGR image to RGB
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False # Improve performance

                # Process the image and get landmarks
                landmarks_data, hand_landmarks_for_drawing = extract_landmarks_video(image_rgb)

                image_rgb.flags.writeable = True
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) # Convert back for drawing

                prediction_text = "No Hand Detected"
                if landmarks_data and len(landmarks_data) == NUM_FEATURES:
                    # Prepare landmarks for prediction
                    landmarks_array = np.array(landmarks_data).reshape(1, -1)

                    # Predict
                    prediction = self.model.predict(landmarks_array, verbose=0) # verbose=0 to suppress prediction logs
                    predicted_index = np.argmax(prediction)
                    confidence = np.max(prediction)

                    # Get the letter label only if confidence is above a threshold
                    if confidence > 0.6: # Confidence threshold
                        predicted_label = self.label_encoder_classes[predicted_index]
                        prediction_text = f"Prediction: {predicted_label} ({confidence:.2f})"
                    else:
                        prediction_text = "Prediction: Uncertain"


                # Draw landmarks on the BGR image
                processed_frame = draw_landmarks(image_bgr.copy(), hand_landmarks_for_drawing)

                # Add prediction text to the frame
                cv2.putText(processed_frame, prediction_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


                # Convert frame to QImage
                h, w, ch = processed_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(processed_frame.data, w, h, bytes_per_line, QImage.Format_BGR888) # Use BGR888
                self.frame_processed.emit(qt_image, prediction_text) # Emit signal

            cap.release()
            self.finished.emit()

        except Exception as e:
            self.error.emit(f"Error during recognition: {e}")
        finally:
            if 'cap' in locals() and cap.isOpened():
                cap.release()

    def stop(self):
        self.is_running = False


# --- Main Application Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setGeometry(100, 100, 900, 700) # x, y, width, height

        # --- Threads ---
        self.data_thread = None
        self.train_thread = None
        self.recog_thread = None

        # --- Central Widget and Tabs ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        # --- Create Tabs ---
        self.create_data_tab()
        self.create_train_tab()
        self.create_recognize_tab()

        # --- Status Bar ---
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")

    # --- Tab Creation Methods ---
    def create_data_tab(self):
        self.data_tab = QWidget()
        self.tabs.addTab(self.data_tab, "1. Data Collection")
        layout = QVBoxLayout(self.data_tab)
        layout_top = QHBoxLayout()
        layout_bottom = QVBoxLayout()

        # --- Image Selection ---
        self.btn_select_images = QPushButton("Select Images for Label")
        self.btn_select_images.clicked.connect(self.select_images)
        self.lbl_image_folder = QLabel("No folder selected")
        self.lbl_image_folder.setWordWrap(True)

        # --- Label Selection ---
        self.lbl_assign_label = QLabel("Assign Label:")
        self.combo_label = QComboBox()
        self.combo_label.addItems(LABELS) # A-Z

        # --- Extraction ---
        self.btn_extract = QPushButton("Extract Landmarks and Save to CSV")
        self.btn_extract.clicked.connect(self.start_extraction)
        self.btn_cancel_extract = QPushButton("Cancel Extraction")
        self.btn_cancel_extract.clicked.connect(self.cancel_extraction)
        self.btn_cancel_extract.setEnabled(False)
        self.progress_bar_data = QProgressBar()

        # --- Log ---
        self.log_data = QTextEdit()
        self.log_data.setReadOnly(True)
        self.log_data.append(f"Landmark data will be saved to: {os.path.abspath(CSV_FILE)}")
        self.log_data.append("Instructions: Select one or more image files. Assign the correct letter label using the dropdown. Click 'Extract Landmarks'. Repeat for all letters.")


        # --- Layout Setup ---
        layout_top.addWidget(self.btn_select_images)
        layout_top.addWidget(self.lbl_assign_label)
        layout_top.addWidget(self.combo_label)
        layout_top.addStretch()

        layout.addLayout(layout_top)
        layout.addWidget(self.lbl_image_folder)
        layout.addWidget(self.btn_extract)
        layout.addWidget(self.btn_cancel_extract)
        layout.addWidget(self.progress_bar_data)
        layout.addWidget(QLabel("Log:"))
        layout.addWidget(self.log_data)
        layout.addLayout(layout_bottom)

        self.selected_image_files = [] # Store paths of selected images

    def create_train_tab(self):
        self.train_tab = QWidget()
        self.tabs.addTab(self.train_tab, "2. Model Training")
        layout = QVBoxLayout(self.train_tab)

        # --- Controls ---
        self.btn_train = QPushButton(f"Train Model using {CSV_FILE}")
        self.btn_train.clicked.connect(self.start_training)
        self.btn_cancel_train = QPushButton("Cancel Training")
        self.btn_cancel_train.clicked.connect(self.cancel_training)
        self.btn_cancel_train.setEnabled(False)

        # --- Log ---
        self.log_train = QTextEdit()
        self.log_train.setReadOnly(True)
        self.log_train.append(f"Model will be saved to: {os.path.abspath(MODEL_FILE)}")
        self.log_train.append("Ensure landmark data exists in CSV before training.")

        # --- Layout Setup ---
        layout.addWidget(self.btn_train)
        layout.addWidget(self.btn_cancel_train)
        layout.addWidget(QLabel("Training Log:"))
        layout.addWidget(self.log_train)

    def create_recognize_tab(self):
        self.recognize_tab = QWidget()
        self.tabs.addTab(self.recognize_tab, "3. Recognition")
        layout = QVBoxLayout(self.recognize_tab)
        layout_controls = QHBoxLayout()
        layout_display = QHBoxLayout()

        # --- Controls ---
        self.btn_recog_image = QPushButton("Recognize from Image")
        self.btn_recog_image.clicked.connect(self.recognize_from_image)
        self.btn_recog_video = QPushButton("Recognize from Video")
        self.btn_recog_video.clicked.connect(self.recognize_from_video)
        self.btn_recog_camera = QPushButton("Start Real-time Recognition")
        self.btn_recog_camera.clicked.connect(self.start_realtime_recognition)
        self.btn_stop_camera = QPushButton("Stop Real-time Recognition")
        self.btn_stop_camera.clicked.connect(self.stop_realtime_recognition)
        self.btn_stop_camera.setEnabled(False)

        # --- Display Area ---
        self.lbl_image_display = QLabel("Load Image/Video or Start Camera")
        self.lbl_image_display.setAlignment(Qt.AlignCenter)
        self.lbl_image_display.setMinimumSize(640, 480) # Set a minimum size
        self.lbl_image_display.setStyleSheet("border: 1px solid black; background-color: #eee;")

        # --- Prediction Label ---
        self.lbl_prediction = QLabel("Prediction: --")
        font = self.lbl_prediction.font()
        font.setPointSize(16)
        self.lbl_prediction.setFont(font)
        self.lbl_prediction.setAlignment(Qt.AlignCenter)

        # --- Layout Setup ---
        layout_controls.addWidget(self.btn_recog_image)
        layout_controls.addWidget(self.btn_recog_video)
        layout_controls.addWidget(self.btn_recog_camera)
        layout_controls.addWidget(self.btn_stop_camera)

        layout_display_v = QVBoxLayout() # Vertical layout for image and prediction
        layout_display_v.addWidget(self.lbl_image_display)
        layout_display_v.addWidget(self.lbl_prediction)

        layout_display.addLayout(layout_display_v) # Add vertical layout to horizontal

        layout.addLayout(layout_controls)
        layout.addLayout(layout_display)

        # Timer for potential static image display duration (optional)
        self.display_timer = QTimer(self)
        self.display_timer.setSingleShot(True)
        # self.display_timer.timeout.connect(lambda: self.lbl_image_display.setText("Load Image/Video or Start Camera"))

    # --- Slot Methods (Event Handlers) ---
    def select_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Hand Sign Images", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if files:
            self.selected_image_files = files
            self.lbl_image_folder.setText(f"{len(files)} images selected for label '{self.combo_label.currentText()}'")
            self.log_data.append(f"Selected {len(files)} images.")
        else:
            self.selected_image_files = []
            self.lbl_image_folder.setText("No images selected")

    def start_extraction(self):
        if not self.selected_image_files:
            self.log_data.append("Error: No images selected.")
            return
        if self.data_thread and self.data_thread.isRunning():
            self.log_data.append("Error: Extraction already in progress.")
            return

        current_label = self.combo_label.currentText()
        labels_for_files = [current_label] * len(self.selected_image_files)

        self.log_data.append(f"Starting extraction for label '{current_label}'...")
        self.status_bar.showMessage("Extracting landmarks...")
        self.progress_bar_data.setValue(0)
        self.btn_extract.setEnabled(False)
        self.btn_select_images.setEnabled(False)
        self.combo_label.setEnabled(False)
        self.btn_cancel_extract.setEnabled(True)

        # Pass copies of the lists
        self.data_thread = DataExtractionThread(list(self.selected_image_files), labels_for_files)
        self.data_thread.progress.connect(self.update_progress_bar_data)
        self.data_thread.finished.connect(self.extraction_finished)
        self.data_thread.error.connect(self.extraction_error)
        self.data_thread.start()

        # Clear selection after starting
        self.selected_image_files = []
        self.lbl_image_folder.setText("Extraction started. Select new images for next label.")


    def cancel_extraction(self):
        if self.data_thread and self.data_thread.isRunning():
            self.data_thread.stop()
            self.btn_cancel_extract.setEnabled(False)
            self.log_data.append("Requesting cancellation...")
        else:
             self.log_data.append("No extraction process to cancel.")

    def update_progress_bar_data(self, value):
        self.progress_bar_data.setValue(value)

    def extraction_finished(self, message):
        self.log_data.append(message)
        self.status_bar.showMessage("Extraction finished.")
        self.progress_bar_data.setValue(100) # Ensure it reaches 100
        self.reset_data_controls()

    def extraction_error(self, message):
        self.log_data.append(f"ERROR: {message}")
        self.status_bar.showMessage("Extraction error.")
        self.reset_data_controls()

    def reset_data_controls(self):
        self.btn_extract.setEnabled(True)
        self.btn_select_images.setEnabled(True)
        self.combo_label.setEnabled(True)
        self.btn_cancel_extract.setEnabled(False)
        self.data_thread = None # Clear the thread object

    def start_training(self):
        if self.train_thread and self.train_thread.isRunning():
            self.log_train.append("Error: Training already in progress.")
            return

        self.log_train.setText(f"Model will be saved to: {os.path.abspath(MODEL_FILE)}\n--- Starting Training ---\n") # Clear log
        self.status_bar.showMessage("Starting model training...")
        self.btn_train.setEnabled(False)
        self.btn_cancel_train.setEnabled(True)

        self.train_thread = TrainingThread()
        self.train_thread.progress.connect(self.update_train_log)
        self.train_thread.finished.connect(self.training_finished)
        self.train_thread.error.connect(self.training_error)
        self.train_thread.start()

    def cancel_training(self):
        if self.train_thread and self.train_thread.isRunning():
            self.train_thread.stop() # Request stop
            self.btn_cancel_train.setEnabled(False) # Disable button after requesting
            self.log_train.append("Requesting training cancellation...")
        else:
            self.log_train.append("No training process to cancel.")


    def update_train_log(self, message):
        self.log_train.append(message)

    def training_finished(self, message):
        self.log_train.append(f"--- Finished --- \n{message}")
        self.status_bar.showMessage("Training finished.")
        self.reset_train_controls()

    def training_error(self, message):
        self.log_train.append(f"--- ERROR --- \n{message}")
        self.status_bar.showMessage("Training error.")
        self.reset_train_controls()

    def reset_train_controls(self):
         self.btn_train.setEnabled(True)
         self.btn_cancel_train.setEnabled(False)
         self.train_thread = None

    def recognize_from_image(self):
        # Stop any running recognition first
        self.stop_realtime_recognition()

        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image for Recognition", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if not file_path:
            return

        # Load model and labels if not already loaded (e.g., first recognition)
        if not hasattr(self, 'recognition_model') or not hasattr(self, 'recognition_labels'):
             try:
                 if not os.path.exists(MODEL_FILE):
                     self.status_bar.showMessage(f"Error: Model file '{MODEL_FILE}' not found. Train the model first.")
                     self.lbl_prediction.setText("Prediction: Model not found")
                     return
                 if not os.path.exists('label_classes.npy'):
                     self.status_bar.showMessage(f"Error: Label file 'label_classes.npy' not found. Train the model first.")
                     self.lbl_prediction.setText("Prediction: Labels not found")
                     return

                 self.recognition_model = load_model(MODEL_FILE)
                 self.recognition_labels = np.load('label_classes.npy', allow_pickle=True)
                 self.status_bar.showMessage("Model and labels loaded.")
             except Exception as e:
                 self.status_bar.showMessage(f"Error loading model/labels: {e}")
                 self.lbl_prediction.setText("Prediction: Error loading model")
                 return

        try:
            image = cv2.imread(file_path)
            if image is None:
                self.status_bar.showMessage(f"Error reading image: {file_path}")
                return

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False # Performance
            landmarks_data, hand_landmarks_drawing = extract_landmarks_video(image_rgb) # Use video func for drawing info
            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) # Convert back

            prediction_text = "Prediction: No Hand Detected"
            if landmarks_data and len(landmarks_data) == NUM_FEATURES:
                landmarks_array = np.array(landmarks_data).reshape(1, -1)
                prediction = self.recognition_model.predict(landmarks_array, verbose=0)
                predicted_index = np.argmax(prediction)
                confidence = np.max(prediction)

                if confidence > 0.6: # Confidence threshold
                    predicted_label = self.recognition_labels[predicted_index]
                    prediction_text = f"Prediction: {predicted_label} ({confidence:.2f})"
                else:
                    prediction_text = "Prediction: Uncertain"

            # Draw landmarks
            processed_frame = draw_landmarks(image_bgr.copy(), hand_landmarks_drawing)

            # Display image and prediction
            self.display_image(processed_frame)
            self.lbl_prediction.setText(prediction_text)
            self.status_bar.showMessage(f"Recognized from {os.path.basename(file_path)}")

        except Exception as e:
            self.status_bar.showMessage(f"Error during image recognition: {e}")
            self.lbl_prediction.setText("Prediction: Error")


    def recognize_from_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video for Recognition", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_path:
            self.start_realtime_recognition(source=file_path)

    def start_realtime_recognition(self, source=0): # Default to webcam (0)
        if self.recog_thread and self.recog_thread.isRunning():
            self.status_bar.showMessage("Recognition already running.")
            return

        # Check if model exists before starting thread
        if not os.path.exists(MODEL_FILE) or not os.path.exists('label_classes.npy'):
             self.status_bar.showMessage("Error: Model or labels not found. Train the model first.")
             self.lbl_prediction.setText("Prediction: Model/Labels missing")
             # Optionally clear display
             self.lbl_image_display.clear()
             self.lbl_image_display.setText("Model or Labels not found")
             self.lbl_image_display.setStyleSheet("border: 1px solid black; background-color: #eee;")
             return

        self.status_bar.showMessage("Starting recognition...")
        self.btn_recog_image.setEnabled(False)
        self.btn_recog_video.setEnabled(False)
        self.btn_recog_camera.setEnabled(False)
        self.btn_stop_camera.setEnabled(True)

        self.recog_thread = RecognitionThread(source=source)
        self.recog_thread.frame_processed.connect(self.update_recognition_frame)
        self.recog_thread.finished.connect(self.recognition_stopped)
        self.recog_thread.error.connect(self.recognition_error)
        self.recog_thread.start()

    def stop_realtime_recognition(self):
        if self.recog_thread and self.recog_thread.isRunning():
            self.recog_thread.stop()
            # Don't re-enable buttons immediately, wait for finished signal
            self.status_bar.showMessage("Stopping recognition...")
        else:
             # If thread isn't running, ensure controls are reset
             self.reset_recognize_controls()


    def update_recognition_frame(self, q_image, prediction):
        """Updates the image display label with a new QImage."""
        pixmap = QPixmap.fromImage(q_image)
        # Scale pixmap to fit the label while maintaining aspect ratio
        self.lbl_image_display.setPixmap(pixmap.scaled(
            self.lbl_image_display.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation))
        self.lbl_prediction.setText(prediction)

    def recognition_stopped(self):
        self.status_bar.showMessage("Recognition stopped.")
        self.reset_recognize_controls()
        # Optionally clear the display when stopped
        self.lbl_image_display.clear()
        self.lbl_image_display.setText("Load Image/Video or Start Camera")
        self.lbl_image_display.setStyleSheet("border: 1px solid black; background-color: #eee;")
        self.lbl_prediction.setText("Prediction: --")


    def recognition_error(self, message):
        self.status_bar.showMessage(f"Recognition Error: {message}")
        self.lbl_prediction.setText("Prediction: Error")
        self.reset_recognize_controls()
        # Optionally clear display on error
        self.lbl_image_display.clear()
        self.lbl_image_display.setText(f"Error: {message}")
        self.lbl_image_display.setStyleSheet("border: 1px solid red; background-color: #fee;")


    def reset_recognize_controls(self):
        self.btn_recog_image.setEnabled(True)
        self.btn_recog_video.setEnabled(True)
        self.btn_recog_camera.setEnabled(True)
        self.btn_stop_camera.setEnabled(False)
        self.recog_thread = None # Clear thread reference

    def display_image(self, cv_image):
        """Converts an OpenCV image to QPixmap and displays it."""
        try:
            # Check if image is valid
            if cv_image is None or cv_image.size == 0:
                self.lbl_image_display.setText("Invalid image")
                self.lbl_image_display.setStyleSheet("border: 1px solid red; background-color: #fee;")
                return

            # Convert BGR (OpenCV default) to RGB
            if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
                rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            elif len(cv_image.shape) == 2: # Grayscale
                h, w = cv_image.shape
                bytes_per_line = w
                qt_image = QImage(cv_image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
            else:
                 self.lbl_image_display.setText("Unsupported image format")
                 self.lbl_image_display.setStyleSheet("border: 1px solid red; background-color: #fee;")
                 return


            pixmap = QPixmap.fromImage(qt_image)
            self.lbl_image_display.setPixmap(pixmap.scaled(
                self.lbl_image_display.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation))
            self.lbl_image_display.setStyleSheet("border: 1px solid black;") # Reset style

        except Exception as e:
             print(f"Error displaying image: {e}")
             self.lbl_image_display.setText("Error displaying image")
             self.lbl_image_display.setStyleSheet("border: 1px solid red; background-color: #fee;")


    def closeEvent(self, event):
        """Ensure threads are stopped cleanly on exit."""
        if self.data_thread and self.data_thread.isRunning():
            self.data_thread.stop()
            self.data_thread.wait() # Wait for thread to finish
        if self.train_thread and self.train_thread.isRunning():
            self.train_thread.stop()
            self.train_thread.wait()
        if self.recog_thread and self.recog_thread.isRunning():
            self.recog_thread.stop()
            self.recog_thread.wait()
        event.accept()


# --- Main Execution ---
if __name__ == '__main__':
    # Set environment variable to potentially mitigate TensorFlow/OpenCV conflicts
    # os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' # If using GPU
    # os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0' # May help with camera access on Windows

    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
