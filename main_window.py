# main_window.py

import sys
import os
import pickle
import cv2
import numpy as np
import pandas as pd 
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QStatusBar,
    QScrollArea, QComboBox, QTextEdit, QTabWidget
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QColor, QKeySequence
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QPoint
from tensorflow.keras.models import load_model 

# Import komponentów z innych plików
from constants import * 
from helpers import (process_static_landmarks, save_static_to_csv,
                     standardize_path, save_path_to_csv, convert_cv_qt)
from camera_thread import CameraThread
from training_threads import StaticTrainingThread, PathTrainingThread
from mp_setup import hands, mp_drawing, mp_drawing_styles, mp_hands 

# Główny widok aplikacji
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)

        # Zmienne stanu aplikacji
        self.camera_thread = None
        self.static_training_thread = None
        self.path_training_thread = None
        self.current_mode = 'idle'
        self.static_model = None; self.static_label_encoder = None; self.static_data_counts = {}
        self.path_model = None; self.path_label_encoder = None; self.path_scaler = None; self.path_data_counts = {}
        self.current_path_points = []; self.is_recording_path = False # Controls drawing in main window
        self.selected_path_label = PATH_ALLOWED_LABELS[0] if PATH_ALLOWED_LABELS else None; self.current_recording_path_label = None
        self.combined_result = "---"

        # Inicjalizacja GUI
        self._setup_ui()

        # Ładowanie inicjalnych stanów aplikacji
        self._load_initial_counts()
        self._load_models_and_encoders()
        self._update_button_states()
        self._update_data_count_display()

    # Tworzenie elementów GUI i ich ułożenia
    def _setup_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.top_layout = QHBoxLayout()
        self.layout.addLayout(self.top_layout)

        self.camera_label = QLabel("Rozpocznij rejestrację/klasyfikację gestów w celu włączenia kamery")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setFixedSize(CAMERA_FEED_WIDTH, CAMERA_FEED_HEIGHT)
        self.camera_label.setStyleSheet("border: 1px solid black; background-color: #f0f0f0;")
        self.top_layout.addWidget(self.camera_label)

        self.tab_widget = QTabWidget()
        self.top_layout.addWidget(self.tab_widget)

        self.static_tab = QWidget()
        self.path_tab = QWidget()
        self.combined_tab = QWidget()
        self.static_layout = QVBoxLayout(self.static_tab)
        self.path_layout = QVBoxLayout(self.path_tab)
        self.combined_layout = QVBoxLayout(self.combined_tab)

        self.tab_widget.addTab(self.static_tab, "Gesty Statyczne")
        self.tab_widget.addTab(self.path_tab, "Gesty Ścieżki")
        self.tab_widget.addTab(self.combined_tab, "Tryb Połączony")

        self._populate_static_tab()

        self._populate_path_tab()

        self._populate_combined_tab()

        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Gotowe.")

    def _populate_static_tab(self):
        layout = self.static_layout
        bold_font = QFont("Arial", 11, QFont.Bold)

        # Kolekcja gestów
        lbl = QLabel("1. Kolekcja gestów statycznych"); lbl.setFont(bold_font); layout.addWidget(lbl)
        self.start_static_collect_button = QPushButton("Zacznij kolekcje gestów statycznych"); self.start_static_collect_button.clicked.connect(self.start_static_collection_mode); layout.addWidget(self.start_static_collect_button)
        self.stop_static_collect_button = QPushButton("Zacznij kolekcje gestów statycznych"); self.stop_static_collect_button.clicked.connect(self.stop_camera); layout.addWidget(self.stop_static_collect_button)
        self.static_collect_instructions = QLabel(f"Naciśnij klawisz ({', '.join(STATIC_ALLOWED_LABELS)})"); layout.addWidget(self.static_collect_instructions)
        self.static_data_count_label = QLabel("Ilość próbek: 0"); layout.addWidget(self.static_data_count_label)
        layout.addSpacing(15)

        # Trening
        lbl = QLabel("2. Trenowanie modelu klasyfikacji gestów statycznych"); lbl.setFont(bold_font); layout.addWidget(lbl)
        self.static_train_button = QPushButton("Trenuj model"); self.static_train_button.clicked.connect(self.start_static_training); layout.addWidget(self.static_train_button)
        layout.addSpacing(15)

        # Klasyfikacja
        lbl = QLabel("3. Klasyfikacja gestów statycznych"); lbl.setFont(bold_font); layout.addWidget(lbl)
        self.start_static_classify_button = QPushButton("Zacznij klasyfikacje w czasie rzeczywistym"); self.start_static_classify_button.clicked.connect(self.start_static_classification_mode); layout.addWidget(self.start_static_classify_button)
        self.stop_static_classify_button = QPushButton("Zakończ klasyfikacje w czasie rzeczywistym"); self.stop_static_classify_button.clicked.connect(self.stop_camera); layout.addWidget(self.stop_static_classify_button)
        self.classify_static_image_button = QPushButton("Klasyfikuj z obrazu"); self.classify_static_image_button.clicked.connect(self.classify_static_from_image); layout.addWidget(self.classify_static_image_button)
        layout.addStretch()

    # Dodanie widgetów do okna gestów ścieżki
    def _populate_path_tab(self):
        layout = self.path_layout
        bold_font = QFont("Arial", 11, QFont.Bold)
        record_key_name = QKeySequence(PATH_RECORD_KEY).toString(QKeySequence.NativeText)

        # Kolekcja
        lbl = QLabel("4. Kolekcja gestów ścieżki"); lbl.setFont(bold_font); layout.addWidget(lbl)
        self.start_path_collect_button = QPushButton("Rozpocznij kolekcje"); self.start_path_collect_button.clicked.connect(self.start_path_collection_mode); layout.addWidget(self.start_path_collect_button)
        self.stop_path_collect_button = QPushButton("Zakończ kolekcje"); self.stop_path_collect_button.clicked.connect(self.stop_camera); layout.addWidget(self.stop_path_collect_button)
        # Layout
        combo_layout = QHBoxLayout(); lbl_combo = QLabel("Wybierz etykiete ścieżki:"); self.path_label_select_combo = QComboBox(); self.path_label_select_combo.addItems(PATH_ALLOWED_LABELS)
        if PATH_ALLOWED_LABELS: self.selected_path_label = PATH_ALLOWED_LABELS[0]; self.path_label_select_combo.setCurrentText(self.selected_path_label)
        self.path_label_select_combo.currentTextChanged.connect(self._update_selected_path_label); combo_layout.addWidget(lbl_combo); combo_layout.addWidget(self.path_label_select_combo); layout.addLayout(combo_layout)
        self.path_collect_instructions = QLabel(f"Wybierz etykiete, następnie przytrzymaj '{record_key_name}' w celu rejestracji."); layout.addWidget(self.path_collect_instructions)
        self.path_data_count_label = QLabel("Ilość Próbek: 0"); layout.addWidget(self.path_data_count_label)
        layout.addSpacing(15)

        # Trening
        lbl = QLabel("5. Trenowanie modelu klasyfikacji gestów ściezki"); lbl.setFont(bold_font); layout.addWidget(lbl)
        self.path_train_button = QPushButton("Trenuj model"); self.path_train_button.clicked.connect(self.start_path_training); layout.addWidget(self.path_train_button)
        layout.addSpacing(15)

        # Klasyfikacja
        lbl = QLabel("6. Klasyfikacja gestów ścieżki"); lbl.setFont(bold_font); layout.addWidget(lbl)
        self.start_path_classify_button = QPushButton("Rozpocznij klasyfikację w czasie rzeczywistym"); self.start_path_classify_button.clicked.connect(self.start_path_classification_mode); layout.addWidget(self.start_path_classify_button)
        self.stop_path_classify_button = QPushButton("Zakończ klasyfikację w czasie rzeczywistym"); self.stop_path_classify_button.clicked.connect(self.stop_camera); layout.addWidget(self.stop_path_classify_button)
        self.path_classify_instructions = QLabel(f"Przytrzymaj '{record_key_name}' żeby zarejestrować ścieżkę do klasyfikacji"); layout.addWidget(self.path_classify_instructions)
        self.path_prediction_label = QLabel("Predykcja ścieżki: ---"); layout.addWidget(self.path_prediction_label)
        layout.addStretch()

    # Widżety okna trybu połączonego
    def _populate_combined_tab(self):
        layout = self.combined_layout
        bold_font = QFont("Arial", 11, QFont.Bold)
        result_font = QFont(); result_font.setPointSize(14); result_font.setBold(True)

        lbl = QLabel("7. Tryb gestów ruchomych"); lbl.setFont(bold_font); layout.addWidget(lbl)
        self.start_combined_button = QPushButton("Rozpocznij tryb gestów ruchomych"); self.start_combined_button.clicked.connect(self.start_combined_mode); layout.addWidget(self.start_combined_button)
        self.stop_combined_button = QPushButton("Zatrzymaj tryb gestów ruchomych"); self.stop_combined_button.clicked.connect(self.stop_camera); layout.addWidget(self.stop_combined_button)
        self.combined_result_label = QLabel("Wynik: ---"); self.combined_result_label.setFont(result_font); layout.addWidget(self.combined_result_label)

        self.history_label = QLabel("Rozpoznana sekwencja:"); layout.addWidget(self.history_label)
        self.history_text_edit = QTextEdit(); self.history_text_edit.setReadOnly(True); self.history_text_edit.setFixedHeight(150); layout.addWidget(self.history_text_edit)
        layout.addStretch()

    # Liczenie danych
    def _load_initial_counts(self):
        # Ładowanie liczników
        self.static_data_counts = {label: 0 for label in STATIC_ALLOWED_LABELS}
        if os.path.exists(STATIC_CSV_FILE) and os.path.getsize(STATIC_CSV_FILE) > 0:
            try:
                df = pd.read_csv(STATIC_CSV_FILE)
                if 'label' in df.columns:
                    counts = df['label'].value_counts().to_dict()
                    # Update liczników tylko dla danych zgadzających się z formatem
                    for label, count in counts.items():
                        if label in self.static_data_counts:
                            self.static_data_counts[label] = count
            except pd.errors.EmptyDataError:
                self.statusBar().showMessage("Plik CSV jest pusty.")
            except Exception as e:
                self.show_error("Nie udało się załadować pliku CSV", f"{e}")

        self.path_data_counts = {label: 0 for label in PATH_ALLOWED_LABELS}
        if os.path.exists(PATH_CSV_FILE) and os.path.getsize(PATH_CSV_FILE) > 0:
            try:
                df = pd.read_csv(PATH_CSV_FILE)
                if 'label' in df.columns:
                    counts = df['label'].value_counts().to_dict()
                    for label, count in counts.items():
                        if label in self.path_data_counts:
                            self.path_data_counts[label] = count
            except pd.errors.EmptyDataError:
                self.statusBar().showMessage("Plik CSV jest pusty.")
            except Exception as e:
                self.show_error("Nie udało się załadować pliku CSV", f"{e}")

    def _update_data_count_display(self):
        self.static_data_count_label.setText(f"Próbki: {sum(self.static_data_counts.values())}")
        self.path_data_count_label.setText(f"Próbki: {sum(self.path_data_counts.values())}")

    # Ładowanie modeli i enkoderów
    def _load_models_and_encoders(self):
        static_loaded, path_loaded = False, False
        if os.path.exists(STATIC_MODEL_FILE) and os.path.exists(STATIC_ENCODER_FILE):
            try:
                self.static_model = load_model(STATIC_MODEL_FILE)
                with open(STATIC_ENCODER_FILE, 'rb') as f: self.static_label_encoder = pickle.load(f)
                if len(self.static_label_encoder.classes_) == STATIC_NUM_CLASSES and self.static_model.output_shape[-1] == STATIC_NUM_CLASSES:
                    static_loaded = True
                else:
                    self.show_warning("Błąd modelu", f"Nieodpowiednia ilość klas, oczekiwano: {STATIC_NUM_CLASSES}. Trenuj ponownie.")
                    self.static_model = self.static_label_encoder = None 
            except Exception as e:
                self.show_error("Błąd ładowania modelu", f"{e}")
                self.static_model = self.static_label_encoder = None
        if os.path.exists(PATH_MODEL_FILE) and os.path.exists(PATH_ENCODER_FILE) and os.path.exists(PATH_SCALER_FILE):
             try:
                self.path_model = load_model(PATH_MODEL_FILE)
                with open(PATH_ENCODER_FILE, 'rb') as f: self.path_label_encoder = pickle.load(f)
                with open(PATH_SCALER_FILE, 'rb') as f: self.path_scaler = pickle.load(f)
                if len(self.path_label_encoder.classes_) == PATH_NUM_CLASSES and self.path_model.output_shape[-1] == PATH_NUM_CLASSES:
                     path_loaded = True
                else:
                     self.show_warning("Błąd modelu", f"Nieodpowiednia ilość klas, oczekiwano: {PATH_NUM_CLASSES}. Trenuj ponownie.")
                     self.path_model = self.path_label_encoder = self.path_scaler = None 
             except Exception as e:
                self.show_error("Błąd ładowania modelu", f"{e}")
                self.path_model = self.path_label_encoder = self.path_scaler = None

        status = []
        if static_loaded: status.append("Model załadowany.")
        if path_loaded: status.append("Model załadowany.")
        self.statusBar().showMessage(" ".join(status) if status else "Żaden model nie został załadowany. Zbierz dane i wytrenuj model.")
        self._update_button_states() 
        return static_loaded, path_loaded

    # --- Mode Control & Button States ---
    def _set_mode(self, new_mode):
        """Sets the application mode and updates UI accordingly."""
        if self.camera_thread and self.camera_thread.isRunning(): self.show_message("Kamera zajęta", "Najpierw zatrzymaj aktualną instancję kamery."); return False
        # Reset UI elements specific to modes
        self.combined_result = "---"; self.combined_result_label.setText(f"Wynik połączony: {self.combined_result}")
        self.history_text_edit.clear() # Clear history on mode change
        self.path_prediction_label.setText("Klasyfikacja ścieżki: ---")
        # Set internal state
        self.current_mode = new_mode; self.is_recording_path = False; self.current_path_points = []; self.current_recording_path_label = None

        if new_mode == 'idle':
            self.stop_camera() # Ensure camera is stopped if setting to idle
        else:
            # Check model prerequisites
            models_ok = True
            if new_mode == 'classify_static' and (not self.static_model or not self.static_label_encoder): self.show_message("Model Error", "Static model not loaded."); models_ok = False
            elif new_mode == 'classify_path' and (not self.path_model or not self.path_label_encoder or not self.path_scaler): self.show_message("Model Error", "Path model/scaler not loaded."); models_ok = False
            elif new_mode == 'combined' and (not self.static_model or not self.static_label_encoder or not self.path_model or not self.path_label_encoder or not self.path_scaler): self.show_message("Model Error", "Both Static and Path models/etc must be loaded."); models_ok = False
            if not models_ok: self.current_mode = 'idle'; self._update_button_states(); return False

            # Start camera thread
            self.camera_thread = CameraThread(mode=new_mode); self._connect_camera_signals()
            if new_mode in ['classify_static', 'combined']: self.camera_thread.load_static_model(self.static_model, self.static_label_encoder)
            if new_mode in ['classify_path', 'combined']: self.camera_thread.load_path_model(self.path_model, self.path_label_encoder, self.path_scaler)
            self.camera_thread.start(); self.statusBar().showMessage(f"Mode set to: {new_mode}")
            if new_mode in ['collect_static', 'collect_path', 'classify_path']: self.setFocus() # Focus for key input

        self._update_button_states()
        return True

    def _update_button_states(self):
        """Enables/disables buttons based on the current mode and model availability."""
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
        """Stops the camera thread and resets UI."""
        if self.camera_thread and self.camera_thread.isRunning(): self.camera_thread.stop(); self.camera_thread = None
        self.current_mode = 'idle'; self.is_recording_path = False; self.current_path_points = []
        self.camera_label.setText("Kamera zatrzymana."); self.camera_label.setPixmap(QPixmap()); self.statusBar().showMessage("Kamera zatrzymana. Model w trybie Idle.")
        self.combined_result_label.setText("Wynik połączony: ---")
        self.history_text_edit.clear() # Clear history on stop
        self._update_button_states()

    # --- Signal Connections & Handling ---
    def _connect_camera_signals(self):
        """Connects signals from the camera thread to main window slots."""
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
        # Disconnect safely before reconnecting
        for signal, slot in connections:
            try: signal.disconnect(slot)
            except TypeError: pass # Signal not connected or already disconnected
            signal.connect(slot)

    def _on_camera_thread_finished(self):
        """Handles cleanup when the camera thread finishes."""
        if self.current_mode != 'idle': self.statusBar().showMessage("Wątek kamery niespodziewanie się zakończył.")
        self.camera_thread = None; self._update_button_states()

    # --- Static Gesture Methods ---
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
            # Use standalone conversion function
            qt_image = convert_cv_qt(img_display); scaled_pixmap = QPixmap.fromImage(qt_image).scaled(self.camera_label.width(), self.camera_label.height(), Qt.KeepAspectRatio)
            self.camera_label.setPixmap(scaled_pixmap); self.statusBar().showMessage(f"Image classification: {prediction_text}")
        except Exception as e: self.show_error("Static Image Classification Error", f"{e}"); self.statusBar().showMessage("Static image classification failed.")

    # --- Path Gesture Methods ---
    def start_path_collection_mode(self): self._set_mode('collect_path')
    def start_path_classification_mode(self): self._set_mode('classify_path')
    def _update_selected_path_label(self, label): self.selected_path_label = label
    def append_path_point(self, point):
        """Appends point and triggers repaint for drawing (used by path & combined modes)."""
        if self.is_recording_path: # Check main window flag for drawing
             self.current_path_points.append(point)
             self.camera_label.update() # Trigger repaint
    def process_and_save_path(self):
        """Processes and saves path data (called by key release in collect mode)."""
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
        """Classifies recorded path (called by key release in classify mode)."""
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

    # --- Event Handling ---
    def keyPressEvent(self, event):
        """Handles key presses for static gestures and path recording trigger."""
        if event.isAutoRepeat(): return
        key = event.key(); key_text = event.text().upper()

        # Only handle static collection keys if Static tab is active
        if self.tab_widget.currentWidget() == self.static_tab and \
           self.current_mode == 'collect_static' and key_text in STATIC_ALLOWED_LABELS:
            if self.camera_thread: self.camera_thread.set_static_label(key_text)

        # Only handle path recording key if Path tab is active
        elif self.tab_widget.currentWidget() == self.path_tab and \
             self.current_mode in ['collect_path', 'classify_path'] and key == PATH_RECORD_KEY:
            if not self.is_recording_path: # Start recording on 'Q' press
                if not self.selected_path_label: self.show_warning("No Path Label", "Please select a path label first."); return
                self.is_recording_path = True; self.current_recording_path_label = self.selected_path_label; self.current_path_points = []
                if self.camera_thread: self.camera_thread.set_path_recording(True) # Tell thread internal state
                self.path_prediction_label.setText("Path Prediction: Recording..."); self.statusBar().showMessage(f"Recording path for '{self.current_recording_path_label}'...")
        else: super().keyPressEvent(event) # Pass other keys up

    def keyReleaseEvent(self, event):
        """Handles key releases for path recording trigger."""
        if event.isAutoRepeat(): return
        key = event.key()

        # Only handle path recording key release if Path tab is active
        if self.tab_widget.currentWidget() == self.path_tab and \
           self.current_mode in ['collect_path', 'classify_path'] and \
           key == PATH_RECORD_KEY and self.is_recording_path:
            self.is_recording_path = False # Stop drawing flag
            if self.camera_thread: self.camera_thread.set_path_recording(False) # Tell thread internal state
            if self.current_recording_path_label:
                self.statusBar().showMessage(f"Finished recording path for '{self.current_recording_path_label}'. Processing...")
                if self.current_mode == 'collect_path': self.process_and_save_path()
                elif self.current_mode == 'classify_path': self.classify_live_path()
            else: self.statusBar().showMessage("Finished recording path (no label assigned). Discarded.")
            self.current_path_points = []; self.current_recording_path_label = None
            self.camera_label.update() # Trigger repaint to clear path
        else: super().keyReleaseEvent(event)

    def closeEvent(self, event):
        """Ensures threads are stopped and resources released on close."""
        self.stop_camera() # Stop camera thread first
        # Wait briefly for threads to potentially finish (optional)
        # if self.static_training_thread and self.static_training_thread.isRunning(): self.static_training_thread.wait(500)
        # if self.path_training_thread and self.path_training_thread.isRunning(): self.path_training_thread.wait(500)
        if (self.static_training_thread and self.static_training_thread.isRunning()) or \
           (self.path_training_thread and self.path_training_thread.isRunning()): print("Warning: Training is running. Closing anyway.")

        # Import and call close_hands from mp_setup
        from mp_setup import close_hands
        close_hands()
        event.accept()

