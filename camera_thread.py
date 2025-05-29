# camera_thread.py

import time
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

from constants import STATIC_CONFIDENCE_THRESHOLD, STATIC_HOLD_DURATION_SEC, PATH_MAX_DURATION_SEC, PATH_TRACKING_LANDMARK
from helpers import process_static_landmarks, convert_cv_qt
from mp_setup import hands, mp_drawing, mp_drawing_styles, mp_hands 


class CameraThread(QThread):
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
        self.current_static_label = None
        self.static_model = None
        self.static_label_encoder = None
        self.path_model = None
        self.path_label_encoder = None
        self.path_scaler = None
        self.is_recording_path = False
        self.combined_trigger_label = None
        self.current_path_points = []
        self.high_conf_start_time = None
        self.high_conf_label = None
        self.path_record_start_time = None


    def run(self):
        self.running = True
        cap = cv2.VideoCapture(0)
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

                if self.mode in ['collect_path', 'classify_path', 'combined']:
                    lm = hand_landmarks.landmark[PATH_TRACKING_LANDMARK]
                    path_point_data = (lm.x, lm.y)

                self._handle_mode_logic(current_time, processed_static_landmarks, path_point_data, static_pred_label, static_pred_conf)

            if not hand_detected:
                self._handle_hand_lost(current_time)

            if display_prediction_text:
                cv2.putText(frame, display_prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            qt_image = convert_cv_qt(frame) 
            self.frame_signal.emit(qt_image)

        cap.release()
        self.status_signal.emit("Camera stopped.")
        self.running = False


    def _handle_mode_logic(self, current_time, processed_static_landmarks, path_point_data, static_pred_label, static_pred_conf):

        if self.mode == 'collect_static' and self.current_static_label and processed_static_landmarks:
            self.static_landmarks_signal.emit([self.current_static_label] + processed_static_landmarks)
            self.status_signal.emit(f"Captured static '{self.current_static_label}'. Press key or Stop.")
            self.current_static_label = None

        elif self.mode in ['collect_path', 'classify_path'] and self.is_recording_path and path_point_data:
            self.path_point_signal.emit(path_point_data) 

        elif self.mode == 'combined':
            self._handle_combined_mode(current_time, path_point_data, static_pred_label, static_pred_conf)


    def _handle_combined_mode(self, current_time, path_point_data, static_pred_label, static_pred_conf): 
        if static_pred_conf >= STATIC_CONFIDENCE_THRESHOLD:
            if self.high_conf_label != static_pred_label:
                self._reset_high_conf_timer(static_pred_label, current_time)
                if self.is_recording_path: self._stop_and_process_combined_path("Static label changed mid-hold.")

            elif not self.is_recording_path and self.high_conf_start_time is not None and \
                 (current_time - self.high_conf_start_time >= STATIC_HOLD_DURATION_SEC):
                self._start_combined_path_recording(static_pred_label, current_time)

            if self.is_recording_path:
                if self.path_record_start_time is not None and \
                   (current_time - self.path_record_start_time > PATH_MAX_DURATION_SEC):
                    self._stop_and_process_combined_path(f"Path time limit ({PATH_MAX_DURATION_SEC:.1f}s) exceeded.")
                elif self.combined_trigger_label != static_pred_label:
                    self._stop_and_process_combined_path("Static label changed during path.")
                elif path_point_data:
                    self.current_path_points.append(path_point_data)
                    self.path_point_signal.emit(path_point_data)

        else:
            self._reset_high_conf_timer(None, None)
            if self.is_recording_path:
                self._stop_and_process_combined_path("Static conf low.")


    def _handle_hand_lost(self, current_time):
        self._reset_high_conf_timer(None, None)
        if self.is_recording_path and self.mode == 'combined':
            self._stop_and_process_combined_path("Hand lost.")


    def _reset_high_conf_timer(self, label, start_time):
        self.high_conf_label = label
        self.high_conf_start_time = start_time


    def _start_combined_path_recording(self, trigger_label, start_time):
        self.is_recording_path = True
        self.combined_trigger_label = trigger_label
        self.current_path_points = []
        self.path_record_start_time = start_time
        self.status_signal.emit(f"Held '{self.combined_trigger_label}' >= {STATIC_HOLD_DURATION_SEC:.1f}s. Recording path...")
        self.path_recording_started_signal.emit()


    def _stop_and_process_combined_path(self, reason):
        self.status_signal.emit(f"{reason} Stopping path for '{self.combined_trigger_label}'.")
        if len(self.current_path_points) > 1:
            self.combined_path_ready_signal.emit(list(self.current_path_points), self.combined_trigger_label)
        self.is_recording_path = False
        self.current_path_points = []
        self.combined_trigger_label = None
        self.path_record_start_time = None
        self.high_conf_start_time = None 
        self.high_conf_label = None
        self.path_recording_stopped_signal.emit()


    def stop(self):
        self.running = False
        if self.is_recording_path and self.mode == 'combined' and self.combined_trigger_label:
             self.status_signal.emit(f"Camera stopping. Processing path for '{self.combined_trigger_label}'.")
             if len(self.current_path_points) > 1:
                 self.combined_path_ready_signal.emit(list(self.current_path_points), self.combined_trigger_label)
             self.path_recording_stopped_signal.emit()
        self.is_recording_path = False
        self.current_path_points = []
        self.combined_trigger_label = None
        self.high_conf_start_time = None
        self.high_conf_label = None
        self.path_record_start_time = None
        self.status_signal.emit("Stopping camera...")


    def set_static_label(self, label):
        if self.mode == 'collect_static' and label in STATIC_ALLOWED_LABELS:
            self.current_static_label = label
            self.status_signal.emit(f"Ready for static '{label}'. Show gesture.")
        elif self.mode != 'collect_static':
             self.status_signal.emit("Not in static data collection mode.")


    def set_path_recording(self, is_recording):
        if self.mode in ['collect_path', 'classify_path']:
            self.is_recording_path = is_recording


    def load_static_model(self, model, encoder):
        self.static_model = model
        self.static_label_encoder = encoder
        msg = "Static model loaded." if model and encoder else "Failed to load static model/encoder."
        if model and encoder: self.status_signal.emit(msg)
        else: self.error_signal.emit(msg)


    def load_path_model(self, model, encoder, scaler):
        self.path_model = model
        self.path_label_encoder = encoder
        self.path_scaler = scaler
        msg = "Path model loaded." if model and encoder and scaler else "Failed to load path model/encoder/scaler."
        if model and encoder and scaler: self.status_signal.emit(msg)
        else: self.error_signal.emit(msg)

