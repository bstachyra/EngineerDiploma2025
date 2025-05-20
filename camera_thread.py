# camera_thread.py

import time
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

# Import komponentów z innych plików
from constants import STATIC_CONFIDENCE_THRESHOLD, STATIC_HOLD_DURATION_SEC, PATH_MAX_DURATION_SEC, PATH_TRACKING_LANDMARK
from helpers import process_static_landmarks, convert_cv_qt
from mp_setup import hands, mp_drawing, mp_drawing_styles, mp_hands 

# Obsługa rejestrowania i procesowania kamery w osobnych wątkach
class CameraThread(QThread):
    # Definicja sygnałów
    frame_signal = pyqtSignal(QImage)
    static_landmarks_signal = pyqtSignal(list)
    path_point_signal = pyqtSignal(tuple)
    combined_path_ready_signal = pyqtSignal(list, str)
    path_recording_started_signal = pyqtSignal()
    path_recording_stopped_signal = pyqtSignal()
    status_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, mode='idle', parent=None):
        super().__init__(parent)
        self.running = False
        self.mode = mode
        # Odniesienia do modelu i enkodera
        self.current_static_label = None # Rejestracja statyczna
        self.static_model = None
        self.static_label_encoder = None
        self.path_model = None
        self.path_label_encoder = None
        self.path_scaler = None
        # Flagi i dane stanów wewnętrznych
        self.is_recording_path = False
        self.combined_trigger_label = None
        self.current_path_points = []
        # Zmienne czasu
        self.high_conf_start_time = None
        self.high_conf_label = None
        self.path_record_start_time = None

    # Głowna pętla wątku kamery
    def run(self):
        self.running = True
        cap = cv2.VideoCapture(0) # Użyj kamery 0 (domyślnej)
        if not cap.isOpened():
            self.error_signal.emit("Error: Could not open camera.")
            self.running = False
            return
        self.status_signal.emit(f"Camera started in {self.mode} mode.")

        while self.running and cap.isOpened():
            current_time = time.time()
            ret, frame = cap.read()
            if not ret:
                self.status_signal.emit("Waiting for frame...")
                time.sleep(0.1)
                continue

            # Procesowanie klatki
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False # Optymalizacja processingu
            results = hands.process(rgb_frame)
            frame.flags.writeable = True # Powrotne włączenie +w w celu rysowania ścieżki

            # Resetowanie danych co klatka
            static_landmarks_data = None
            path_point_data = None
            static_pred_label = None
            static_pred_conf = 0.0
            display_prediction_text = ""
            hand_detected = results.multi_hand_landmarks is not None

            if hand_detected:
                hand_landmarks = results.multi_hand_landmarks[0]
                # Rysowanie punktów charakterystycznych
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing_styles.get_default_hand_landmarks_style(),
                                          mp_drawing_styles.get_default_hand_connections_style())
                # Procesowanie punktów charakterystycznych dla gestów statycznych
                processed_static_landmarks = process_static_landmarks(hand_landmarks)

                # Klasyfikacja gestów statycznych
                if self.mode in ['classify_static', 'combined'] and processed_static_landmarks and self.static_model and self.static_label_encoder:
                    try:
                        input_data = np.array([processed_static_landmarks], dtype=np.float32)
                        prediction = self.static_model.predict(input_data, verbose=0)[0]
                        idx = np.argmax(prediction)
                        static_pred_conf = prediction[idx]
                        static_pred_label = self.static_label_encoder.inverse_transform([idx])[0]
                        if self.mode == 'classify_static':
                            display_prediction_text = f"Static: {static_pred_label} ({static_pred_conf:.2f})"
                    except Exception as e:
                        self.status_signal.emit(f"Static Classif. Error: {e}")
                        static_pred_label = None; static_pred_conf = 0.0

                # Rejestracja ścieżki
                if self.mode in ['collect_path', 'classify_path', 'combined']:
                    lm = hand_landmarks.landmark[PATH_TRACKING_LANDMARK]
                    path_point_data = (lm.x, lm.y)

                # Logika zależna od trybu
                self._handle_mode_logic(current_time, processed_static_landmarks, path_point_data, static_pred_label, static_pred_conf)

            # Stracenie widoku ręki
            if not hand_detected:
                self._handle_hand_lost(current_time)

            # Pokaż tekst i ramkę
            if display_prediction_text:
                cv2.putText(frame, display_prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            qt_image = convert_cv_qt(frame) 
            self.frame_signal.emit(qt_image)

        # Wyczyszczenie nakładek na obraz
        cap.release()
        self.status_signal.emit("Camera stopped.")
        self.running = False

    # Obsługa logiki zależnie od trybu działania
    def _handle_mode_logic(self, current_time, processed_static_landmarks, path_point_data, static_pred_label, static_pred_conf):
        # Rejestracja gestów statycznych
        if self.mode == 'collect_static' and self.current_static_label and processed_static_landmarks:
            self.static_landmarks_signal.emit([self.current_static_label] + processed_static_landmarks)
            self.status_signal.emit(f"Captured static '{self.current_static_label}'. Press key or Stop.")
            self.current_static_label = None

        # 2. Rejestracja/klasyfikacja ścieżki
        elif self.mode in ['collect_path', 'classify_path'] and self.is_recording_path and path_point_data:
            self.path_point_signal.emit(path_point_data) 

        # 3. Tryb połączony
        elif self.mode == 'combined':
            self._handle_combined_mode(current_time, path_point_data, static_pred_label, static_pred_conf)

    # Obsługa zmiany stanów i logiki na trybu połączonego
    def _handle_combined_mode(self, current_time, path_point_data, static_pred_label, static_pred_conf):
        # Sprawdzenie minimalnej ufności dla gestu statycznego 
        if static_pred_conf >= STATIC_CONFIDENCE_THRESHOLD:
            # Sledzenie czas trwania gestu statycznego na wysokim poziomie unfości
            if self.high_conf_label != static_pred_label:
                self._reset_high_conf_timer(static_pred_label, current_time)
                # Zatrzymanie rejestrowanie ścieżki jeśli etykieta znaku się zmieniła
                if self.is_recording_path: self._stop_and_process_combined_path("Static label changed mid-hold.")

            # Sprawdzenie czy minimalny czas trwania gestu minął i czy rejestracja ścieżki już trwa
            elif not self.is_recording_path and self.high_conf_start_time is not None and \
                 (current_time - self.high_conf_start_time >= STATIC_HOLD_DURATION_SEC):
                self._start_combined_path_recording(static_pred_label, current_time)

            # Jeśli trwa rejestracja ścieżki
            if self.is_recording_path:
                # Sprawdzenie limitu czasowego rejestracji ścieżki
                if self.path_record_start_time is not None and \
                   (current_time - self.path_record_start_time > PATH_MAX_DURATION_SEC):
                    self._stop_and_process_combined_path(f"Path time limit ({PATH_MAX_DURATION_SEC:.1f}s) exceeded.")
                # Sprawdzenie czy etykieta gestu statycznego zmieniła się w trakcie rejestracji ścieżki
                elif self.combined_trigger_label != static_pred_label:
                    self._stop_and_process_combined_path("Static label changed during path.")
                # Dodanie punktu ścieżki jeśli wszystkie powyższe warunki spełnione
                elif path_point_data:
                    self.current_path_points.append(path_point_data)
                    self.path_point_signal.emit(path_point_data)

        # Jeśli pewność spadła poniżej limitu
        else:
            self._reset_high_conf_timer(None, None) # Reset zegara
            if self.is_recording_path:
                self._stop_and_process_combined_path("Static conf low.")

    # Obsługa logiki w przypadku "zgubienia" ręki
    def _handle_hand_lost(self, current_time):
        self._reset_high_conf_timer(None, None)
        if self.is_recording_path and self.mode == 'combined':
            self._stop_and_process_combined_path("Hand lost.")

    # Reset zegara stanu wysokiej pewności
    def _reset_high_conf_timer(self, label, start_time):
        self.high_conf_label = label
        self.high_conf_start_time = start_time

    # Zaczęcie rejestracji ścieżki w trybie połączonym
    def _start_combined_path_recording(self, trigger_label, start_time):
        self.is_recording_path = True
        self.combined_trigger_label = trigger_label
        self.current_path_points = []
        self.path_record_start_time = start_time
        self.status_signal.emit(f"Held '{self.combined_trigger_label}' >= {STATIC_HOLD_DURATION_SEC:.1f}s. Recording path...")
        self.path_recording_started_signal.emit()

    # Zatrzymanie rejestracji ścieżki, emitowanie sygnału statusu jeśli ścieżka jest poprawna
    def _stop_and_process_combined_path(self, reason):
        self.status_signal.emit(f"{reason} Stopping path for '{self.combined_trigger_label}'.")
        if len(self.current_path_points) > 1: # Procesowanie jedynie jeżeli ścieżka nie jest punktem (brak ruchu)
            self.combined_path_ready_signal.emit(list(self.current_path_points), self.combined_trigger_label)
        # Reset wszystkich stanów trybu połączonego
        self.is_recording_path = False
        self.current_path_points = []
        self.combined_trigger_label = None
        self.path_record_start_time = None
        self.high_conf_start_time = None 
        self.high_conf_label = None
        self.path_recording_stopped_signal.emit() # Sygnał stop rysowania

    # Zatrzymanie wątku kamery
    def stop(self):
        self.running = False
        # Procesowanie ścieżki przerwanej w trakcie w trybie połączonym
        if self.is_recording_path and self.mode == 'combined' and self.combined_trigger_label:
             self.status_signal.emit(f"Camera stopping. Processing path for '{self.combined_trigger_label}'.")
             if len(self.current_path_points) > 1:
                 self.combined_path_ready_signal.emit(list(self.current_path_points), self.combined_trigger_label)
             self.path_recording_stopped_signal.emit()
        # Reset stanów po stopie
        self.is_recording_path = False
        self.current_path_points = []
        self.combined_trigger_label = None
        self.high_conf_start_time = None
        self.high_conf_label = None
        self.path_record_start_time = None
        self.status_signal.emit("Stopping camera...")

    # Ustawienie etykiet dla rejestracji gestów statycznych
    def set_static_label(self, label):
        if self.mode == 'collect_static' and label in STATIC_ALLOWED_LABELS:
            self.current_static_label = label
            self.status_signal.emit(f"Ready for static '{label}'. Show gesture.")
        elif self.mode != 'collect_static':
             self.status_signal.emit("Not in static data collection mode.")

    # Ustawienie stanu nagrywania dla rejestracji/klasyfikacji ścieżki
    def set_path_recording(self, is_recording):
        if self.mode in ['collect_path', 'classify_path']:
            self.is_recording_path = is_recording

    # Ładowanie statycznego modelu i enkodera
    def load_static_model(self, model, encoder):
        self.static_model = model
        self.static_label_encoder = encoder
        msg = "Static model loaded." if model and encoder else "Failed to load static model/encoder."
        if model and encoder: self.status_signal.emit(msg)
        else: self.error_signal.emit(msg)

    # Ładowanie modelu, enkodera i skalowania ścieżki
    def load_path_model(self, model, encoder, scaler):
        self.path_model = model
        self.path_label_encoder = encoder
        self.path_scaler = scaler
        msg = "Path model loaded." if model and encoder and scaler else "Failed to load path model/encoder/scaler."
        if model and encoder and scaler: self.status_signal.emit(msg)
        else: self.error_signal.emit(msg)

