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
    QScrollArea, QComboBox, QTextEdit, QTabWidget, QFrame,
    QSizePolicy
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QColor, QKeySequence, QPalette
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QPoint
from tensorflow.keras.models import load_model

from constants import *
from helpers import (process_static_landmarks, save_static_to_csv,
                     standardize_path, save_path_to_csv, convert_cv_qt)
from camera_thread import CameraThread
from training_threads import StaticTrainingThread, PathTrainingThread
from testing_threads import StaticModelTestThread, PathModelTestThread, PathFileClassifierThread, DPI
from mp_setup import hands, mp_drawing, mp_drawing_styles, mp_hands

import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
import io

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setGeometry(100, 100, WINDOW_WIDTH + 100, WINDOW_HEIGHT + 50)

        self.camera_thread = None
        self.static_training_thread = None
        self.path_training_thread = None
        self.static_model_test_thread = None
        self.path_model_test_thread = None
        self.path_file_classifier_thread = None

        self.current_mode = 'idle'
        self.static_model = None; self.static_label_encoder = None; self.static_data_counts = {}
        self.path_model = None; self.path_label_encoder = None; self.path_scaler = None; self.path_data_counts = {}
        self.current_path_points = []; self.is_recording_path = False
        self.selected_path_label = PATH_ALLOWED_LABELS[0] if PATH_ALLOWED_LABELS else None; self.current_recording_path_label = None
        self.combined_result = "---"

        self.static_image_classification_result_label = None
        self.test_static_model_button = None
        self.stop_test_static_model_button = None
        self.static_test_results_area = None
        self.static_confusion_matrix_label = None

        self.test_path_model_button = None
        self.stop_test_path_model_button = None
        self.path_test_results_area = None
        self.path_confusion_matrix_label = None

        self.classify_path_from_file_button = None
        self.stop_classify_path_from_file_button = None

        self._setup_ui()
        self._apply_styles()
        self._load_initial_counts()
        self._load_models_and_encoders()
        self._update_button_states()
        self._update_data_count_display()

    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QTabWidget::pane { 
                border-top: 2px solid #C2C7CB;
                background-color: #ffffff; 
            }
            QTabBar::tab {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                            stop: 0 #E1E1E1, stop: 0.4 #DDDDDD,
                                            stop: 0.5 #D8D8D8, stop: 1.0 #D3D3D3);
                border: 1px solid #C4C4C3;
                border-bottom-color: #C2C7CB; 
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 150px;
                padding: 8px 10px;
                margin-right: 2px; 
                font-size: 10pt; 
                text-align: center;
            }
            QTabBar::tab:selected {
                background: #ffffff; 
                border-color: #9B9B9B;
                border-bottom-color: #ffffff; 
            }
            QTabBar::tab:!selected:hover {
                background: #E8E8E8; 
            }
            QPushButton {
                background-color: #4CAF50; 
                color: white;
                border-radius: 5px; 
                padding: 8px 15px; 
                font-size: 10pt; 
                border: 1px solid #388E3C; 
                min-height: 20px; 
            }
            QPushButton:hover {
                background-color: #45a049; 
            }
            QPushButton:pressed {
                background-color: #3e8e41; 
            }
            QPushButton:disabled {
                background-color: #d3d3d3; 
                color: #a0a0a0;
                border-color: #c0c0c0;
            }
            QLabel {
                font-size: 10pt; 
                color: #333333; 
                padding: 2px;
            }
            QLabel#CameraLabel { 
                border: 2px solid #4CAF50; 
                background-color: #e8f5e9; 
                border-radius: 5px;
            }
            QLabel#CombinedResultLabel, QLabel#PathPredictionLabel {
                font-size: 14pt;
                font-weight: bold;
                color: #2c3e50; 
                padding: 5px;
                border: 1px solid #bdc3c7;
                background-color: #ecf0f1;
                border-radius: 4px;
                min-height: 25px;
                text-align: center;
            }
            QLabel#StaticImageResultLabel {
                font-size: 10pt;
                color: #333333;
                padding: 5px;
                border: 1px solid #cccccc;
                background-color: #f9f9f9;
                border-radius: 4px;
                min-height: 25px; 
                text-align: center;
            }
            QLabel#StaticConfusionMatrixStatusLabel, QLabel#PathConfusionMatrixStatusLabel {
                font-size: 9pt; /* Smaller font for more text */
                color: #333333;
                padding: 5px;
                border: 1px solid #cccccc;
                background-color: #f0f0f0;
                border-radius: 4px;
                min-height: 100px; /* Increased height for multiple lines */
                text-align: center;
                word-wrap: break-word; 
                alignment: 'AlignCenter';
            }
            QComboBox {
                border: 1px solid #bdc3c7; 
                border-radius: 4px;
                padding: 5px;
                background-color: white;
                font-size: 10pt;
                min-height: 20px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left-width: 1px;
                border-left-color: #bdc3c7;
                border-left-style: solid;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }
            QComboBox::down-arrow {
                image: url(placeholder_down_arrow.png); /* Placeholder */
            }
            QTextEdit {
                border: 1px solid #bdc3c7; 
                border-radius: 4px;
                padding: 5px;
                background-color: #fdfdfd; 
                font-size: 10pt;
                color: #333333;
            }
            QStatusBar {
                background-color: #34495e; 
                color: white; 
                font-size: 9pt;
            }
            QStatusBar::item {
                border: none; 
            }
        """)

    def _setup_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.layout.setSpacing(10)

        self.top_layout = QHBoxLayout()
        self.top_layout.setSpacing(15)
        self.layout.addLayout(self.top_layout)

        self.camera_label = QLabel("Rozpocznij rejestrację/klasyfikację gestów w celu włączenia kamery")
        self.camera_label.setObjectName("CameraLabel")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setFixedSize(CAMERA_FEED_WIDTH, CAMERA_FEED_HEIGHT)
        self.top_layout.addWidget(self.camera_label, 0) 

        self.tab_widget = QTabWidget()
        tab_widget_size_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.tab_widget.setSizePolicy(tab_widget_size_policy)
        self.top_layout.addWidget(self.tab_widget, 1)
    
        self.static_tab = QWidget()
        self.path_tab = QWidget()
        self.combined_tab = QWidget()

        self.static_layout = QVBoxLayout(self.static_tab)
        self.static_layout.setContentsMargins(10, 10, 10, 10)
        self.static_layout.setSpacing(10)

        self.path_layout = QVBoxLayout(self.path_tab)
        self.path_layout.setContentsMargins(10, 10, 10, 10)
        self.path_layout.setSpacing(10)

        self.combined_layout = QVBoxLayout(self.combined_tab)
        self.combined_layout.setContentsMargins(10, 10, 10, 10)
        self.combined_layout.setSpacing(10)

        self.tab_widget.addTab(self.static_tab, "Gesty Statyczne")
        self.tab_widget.addTab(self.path_tab, "Gesty Ścieżki")
        self.tab_widget.addTab(self.combined_tab, "Tryb Połączony")

        self._populate_static_tab()
        self._populate_path_tab()
        self._populate_combined_tab()

        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Gotowe.")

    def _create_section_label(self, text):
        lbl = QLabel(text)
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        lbl.setFont(font)
        return lbl

    def _populate_static_tab(self):
        layout = self.static_layout
        expanding_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout.addWidget(self._create_section_label("1. Kolekcja gestów statycznych"))
        self.start_static_collect_button = QPushButton("Rozpocznij kolekcję gestów statycznych")
        self.start_static_collect_button.setSizePolicy(expanding_policy)
        self.start_static_collect_button.clicked.connect(self.start_static_collection_mode)
        layout.addWidget(self.start_static_collect_button)

        self.stop_static_collect_button = QPushButton("Zakończ kolekcję gestów statycznych")
        self.stop_static_collect_button.setSizePolicy(expanding_policy)
        self.stop_static_collect_button.clicked.connect(self.stop_camera)
        layout.addWidget(self.stop_static_collect_button)

        self.static_collect_instructions = QLabel(f"Naciśnij klawisz ({', '.join(STATIC_ALLOWED_LABELS)}) aby zapisać gest.")
        layout.addWidget(self.static_collect_instructions)
        self.static_data_count_label = QLabel("Ilość próbek: 0")
        layout.addWidget(self.static_data_count_label)
        layout.addSpacing(10)

        layout.addWidget(self._create_section_label("2. Trenowanie modelu klasyfikacji gestów statycznych"))
        self.static_train_button = QPushButton("Trenuj model statyczny")
        self.static_train_button.setSizePolicy(expanding_policy)
        self.static_train_button.clicked.connect(self.start_static_training)
        layout.addWidget(self.static_train_button)
        layout.addSpacing(10)

        layout.addWidget(self._create_section_label("3. Klasyfikacja gestów statycznych"))
        self.start_static_classify_button = QPushButton("Rozpocznij klasyfikację (na żywo)")
        self.start_static_classify_button.setSizePolicy(expanding_policy)
        self.start_static_classify_button.clicked.connect(self.start_static_classification_mode)
        layout.addWidget(self.start_static_classify_button)

        self.stop_static_classify_button = QPushButton("Zakończ klasyfikację (na żywo)")
        self.stop_static_classify_button.setSizePolicy(expanding_policy)
        self.stop_static_classify_button.clicked.connect(self.stop_camera)
        layout.addWidget(self.stop_static_classify_button)

        self.classify_static_image_button = QPushButton("Klasyfikuj z obrazu")
        self.classify_static_image_button.setSizePolicy(expanding_policy)
        self.classify_static_image_button.clicked.connect(self.classify_static_from_image)
        layout.addWidget(self.classify_static_image_button)

        self.static_image_classification_result_label = QLabel("Wynik klasyfikacji z obrazu: ---")
        self.static_image_classification_result_label.setObjectName("StaticImageResultLabel")
        self.static_image_classification_result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.static_image_classification_result_label)
        layout.addSpacing(10)

        layout.addWidget(self._create_section_label("4. Testowanie Modelu Statycznego"))
        self.test_static_model_button = QPushButton("Testuj Model z Folderu")
        self.test_static_model_button.setSizePolicy(expanding_policy)
        self.test_static_model_button.clicked.connect(self.start_static_model_test)
        layout.addWidget(self.test_static_model_button)
        
        self.stop_test_static_model_button = QPushButton("Zatrzymaj Testowanie")
        self.stop_test_static_model_button.setSizePolicy(expanding_policy)
        self.stop_test_static_model_button.clicked.connect(self.stop_static_model_test)
        self.stop_test_static_model_button.setEnabled(False) 
        layout.addWidget(self.stop_test_static_model_button)

        layout.addWidget(QLabel("Wyniki testu (model statyczny):"))
        self.static_test_results_area = QTextEdit()
        self.static_test_results_area.setReadOnly(True)
        self.static_test_results_area.setFixedHeight(100) 
        layout.addWidget(self.static_test_results_area)

        layout.addWidget(QLabel("Status zapisanych metryk (model statyczny):"))
        self.static_confusion_matrix_label = QLabel("Metryki (macierz pomyłek, wykresy) zostaną zapisane do plików po teście.")
        self.static_confusion_matrix_label.setObjectName("StaticConfusionMatrixStatusLabel")
        self.static_confusion_matrix_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.static_confusion_matrix_label)
        
        layout.addStretch()

    def _populate_path_tab(self):
        layout = self.path_layout
        expanding_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        record_key_name = QKeySequence(PATH_RECORD_KEY).toString(QKeySequence.NativeText)

        layout.addWidget(self._create_section_label("1. Kolekcja gestów ścieżki"))
        self.start_path_collect_button = QPushButton("Rozpocznij kolekcję ścieżek")
        self.start_path_collect_button.setSizePolicy(expanding_policy) 
        self.start_path_collect_button.clicked.connect(self.start_path_collection_mode)
        layout.addWidget(self.start_path_collect_button)
        
        self.stop_path_collect_button = QPushButton("Zakończ kolekcję ścieżek")
        self.stop_path_collect_button.setSizePolicy(expanding_policy)
        self.stop_path_collect_button.clicked.connect(self.stop_camera)
        layout.addWidget(self.stop_path_collect_button)

        combo_layout = QHBoxLayout() 
        lbl_combo = QLabel("Wybierz etykietę ścieżki:")
        self.path_label_select_combo = QComboBox()
        self.path_label_select_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed) 
        self.path_label_select_combo.addItems(PATH_ALLOWED_LABELS)
        if PATH_ALLOWED_LABELS:
            self.selected_path_label = PATH_ALLOWED_LABELS[0]
            self.path_label_select_combo.setCurrentText(self.selected_path_label)
        self.path_label_select_combo.currentTextChanged.connect(self._update_selected_path_label)
        combo_layout.addWidget(lbl_combo) 
        combo_layout.addWidget(self.path_label_select_combo, 1) 
        layout.addLayout(combo_layout)

        self.path_collect_instructions = QLabel(f"Wybierz etykietę, następnie przytrzymaj '{record_key_name}' aby zarejestrować ścieżkę.")
        layout.addWidget(self.path_collect_instructions)
        self.path_data_count_label = QLabel("Ilość próbek: 0")
        layout.addWidget(self.path_data_count_label)
        layout.addSpacing(10) 

        layout.addWidget(self._create_section_label("2. Trenowanie modelu klasyfikacji gestów ścieżki"))
        self.path_train_button = QPushButton("Trenuj model ścieżek")
        self.path_train_button.setSizePolicy(expanding_policy)
        self.path_train_button.clicked.connect(self.start_path_training)
        layout.addWidget(self.path_train_button)
        layout.addSpacing(10) 

        layout.addWidget(self._create_section_label("3. Klasyfikacja gestów ścieżki (na żywo)"))
        self.start_path_classify_button = QPushButton("Rozpocznij klasyfikację ścieżek (na żywo)")
        self.start_path_classify_button.setSizePolicy(expanding_policy)
        self.start_path_classify_button.clicked.connect(self.start_path_classification_mode)
        layout.addWidget(self.start_path_classify_button)

        self.stop_path_classify_button = QPushButton("Zakończ klasyfikację ścieżek (na żywo)")
        self.stop_path_classify_button.setSizePolicy(expanding_policy)
        self.stop_path_classify_button.clicked.connect(self.stop_camera)
        layout.addWidget(self.stop_path_classify_button)
        
        self.path_classify_instructions_live = QLabel(f"Dla klasyfikacji na żywo: Przytrzymaj '{record_key_name}' aby zarejestrować ścieżkę.")
        layout.addWidget(self.path_classify_instructions_live)
        self.path_prediction_label = QLabel("Predykcja ścieżki (na żywo): ---")
        self.path_prediction_label.setObjectName("PathPredictionLabel") 
        self.path_prediction_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.path_prediction_label)
        layout.addSpacing(10)

        layout.addWidget(self._create_section_label("4. Klasyfikacja ścieżki z pliku/sekwencji"))
        self.classify_path_from_file_button = QPushButton("Klasyfikuj ścieżkę z pliku/sekwencji")
        self.classify_path_from_file_button.setSizePolicy(expanding_policy)
        self.classify_path_from_file_button.clicked.connect(self.start_classify_path_from_file)
        layout.addWidget(self.classify_path_from_file_button)

        self.stop_classify_path_from_file_button = QPushButton("Zatrzymaj klasyfikację z pliku")
        self.stop_classify_path_from_file_button.setSizePolicy(expanding_policy)
        self.stop_classify_path_from_file_button.clicked.connect(self.stop_path_file_classification)
        self.stop_classify_path_from_file_button.setEnabled(False)
        layout.addWidget(self.stop_classify_path_from_file_button)
        layout.addSpacing(10)

        layout.addWidget(self._create_section_label("5. Testowanie Modelu Ścieżek"))
        self.test_path_model_button = QPushButton("Testuj Model Ścieżek z Folderu")
        self.test_path_model_button.setSizePolicy(expanding_policy)
        self.test_path_model_button.clicked.connect(self.start_path_model_test) 
        layout.addWidget(self.test_path_model_button)
        
        self.stop_test_path_model_button = QPushButton("Zatrzymaj Testowanie Ścieżek")
        self.stop_test_path_model_button.setSizePolicy(expanding_policy)
        self.stop_test_path_model_button.clicked.connect(self.stop_path_model_test) 
        self.stop_test_path_model_button.setEnabled(False) 
        layout.addWidget(self.stop_test_path_model_button)

        layout.addWidget(QLabel("Wyniki testu (model ścieżek):"))
        self.path_test_results_area = QTextEdit() 
        self.path_test_results_area.setReadOnly(True)
        self.path_test_results_area.setFixedHeight(100) 
        layout.addWidget(self.path_test_results_area)

        layout.addWidget(QLabel("Status zapisanych metryk (model ścieżek):"))
        self.path_confusion_matrix_label = QLabel("Metryki (macierz pomyłek, wykresy) zostaną zapisane do plików po teście.") 
        self.path_confusion_matrix_label.setObjectName("PathConfusionMatrixStatusLabel")
        self.path_confusion_matrix_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.path_confusion_matrix_label)
        
        layout.addStretch()

    def _populate_combined_tab(self):
        layout = self.combined_layout
        expanding_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout.addWidget(self._create_section_label("Tryb gestów ruchomych (połączony)"))
        self.start_combined_button = QPushButton("Rozpocznij tryb połączony")
        self.start_combined_button.setSizePolicy(expanding_policy)
        self.start_combined_button.clicked.connect(self.start_combined_mode)
        layout.addWidget(self.start_combined_button)

        self.stop_combined_button = QPushButton("Zatrzymaj tryb połączony")
        self.stop_combined_button.setSizePolicy(expanding_policy)
        self.stop_combined_button.clicked.connect(self.stop_camera)
        layout.addWidget(self.stop_combined_button)

        self.combined_result_label = QLabel("Wynik: ---")
        self.combined_result_label.setObjectName("CombinedResultLabel")
        self.combined_result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.combined_result_label)
        layout.addSpacing(10)

        layout.addWidget(self._create_section_label("Rozpoznana sekwencja:"))
        self.history_text_edit = QTextEdit() 
        self.history_text_edit.setReadOnly(True)
        self.history_text_edit.setFixedHeight(150)
        layout.addWidget(self.history_text_edit)
        layout.addStretch()
    
    def _load_initial_counts(self):
        self.static_data_counts = {label: 0 for label in STATIC_ALLOWED_LABELS}
        if os.path.exists(STATIC_CSV_FILE) and os.path.getsize(STATIC_CSV_FILE) > 0:
            try:
                df = pd.read_csv(STATIC_CSV_FILE)
                if 'label' in df.columns:
                    counts = df['label'].value_counts().to_dict()
                    for label, count in counts.items():
                        if label in self.static_data_counts:
                            self.static_data_counts[label] = count
            except pd.errors.EmptyDataError:
                self.statusBar().showMessage("Plik CSV gestów statycznych jest pusty.")
            except Exception as e:
                self.show_error("Błąd ładowania CSV (statyczne)", f"Nie udało się załadować pliku {STATIC_CSV_FILE}: {e}")

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
                self.statusBar().showMessage("Plik CSV gestów ścieżki jest pusty.")
            except Exception as e:
                self.show_error("Błąd ładowania CSV (ścieżki)", f"Nie udało się załadować pliku {PATH_CSV_FILE}: {e}")

    def _update_data_count_display(self):
        total_static = sum(self.static_data_counts.values())
        self.static_data_count_label.setText(f"Zebrane próbki statyczne: {total_static}")
        total_path = sum(self.path_data_counts.values())
        self.path_data_count_label.setText(f"Zebrane próbki ścieżek: {total_path}")

    def _load_models_and_encoders(self):
        static_loaded, path_loaded = False, False
        if os.path.exists(STATIC_MODEL_FILE) and os.path.exists(STATIC_ENCODER_FILE):
            try:
                self.static_model = load_model(STATIC_MODEL_FILE)
                with open(STATIC_ENCODER_FILE, 'rb') as f: self.static_label_encoder = pickle.load(f)
                if len(self.static_label_encoder.classes_) == STATIC_NUM_CLASSES and \
                   hasattr(self.static_model, 'output_shape') and self.static_model.output_shape[-1] == STATIC_NUM_CLASSES:
                    static_loaded = True
                else:
                    self.show_warning("Błąd modelu statycznego", f"Niezgodność klas w modelu statycznym ({self.static_model.output_shape[-1]} vs {STATIC_NUM_CLASSES} oczekiwane) lub enkoderze ({len(self.static_label_encoder.classes_)} vs {STATIC_NUM_CLASSES} oczekiwane). Proszę wytrenować model ponownie.")
                    self.static_model = self.static_label_encoder = None
            except Exception as e:
                self.show_error("Błąd ładowania modelu statycznego", f"Nie udało się załadować modelu/enkodera: {e}")
                self.static_model = self.static_label_encoder = None
        
        if os.path.exists(PATH_MODEL_FILE) and os.path.exists(PATH_ENCODER_FILE) and os.path.exists(PATH_SCALER_FILE):
             try:
                self.path_model = load_model(PATH_MODEL_FILE)
                with open(PATH_ENCODER_FILE, 'rb') as f: self.path_label_encoder = pickle.load(f)
                with open(PATH_SCALER_FILE, 'rb') as f: self.path_scaler = pickle.load(f)
                if len(self.path_label_encoder.classes_) == PATH_NUM_CLASSES and \
                   hasattr(self.path_model, 'output_shape') and self.path_model.output_shape[-1] == PATH_NUM_CLASSES:
                     path_loaded = True
                else:
                     self.show_warning("Błąd modelu ścieżek", f"Niezgodność klas w modelu ścieżek ({self.path_model.output_shape[-1]} vs {PATH_NUM_CLASSES} oczekiwane), enkoderze ({len(self.path_label_encoder.classes_)} vs {PATH_NUM_CLASSES} oczekiwane) lub skalerze. Proszę wytrenować model ponownie.")
                     self.path_model = self.path_label_encoder = self.path_scaler = None
             except Exception as e:
                self.show_error("Błąd ładowania modelu ścieżek", f"Nie udało się załadować modelu/enkodera/skalera: {e}")
                self.path_model = self.path_label_encoder = self.path_scaler = None

        status_messages = []
        if static_loaded: status_messages.append("Model statyczny załadowany.")
        else: status_messages.append("Model statyczny NIEzaładowany.")
        if path_loaded: status_messages.append("Model ścieżek załadowany.")
        else: status_messages.append("Model ścieżek NIEzaładowany.")
        
        if not static_loaded or not path_loaded:
            status_messages.append("Zbierz dane i wytrenuj brakujące modele.")
            
        self.statusBar().showMessage(" ".join(status_messages))
        self._update_button_states()
        return static_loaded, path_loaded
        
    def _set_mode(self, new_mode):
        # Stop any ongoing long operations before changing mode or starting camera
        if self.camera_thread and self.camera_thread.isRunning():
            self.show_message("Kamera zajęta", "Najpierw zatrzymaj aktualną operację kamery.")
            return False
        if self.path_file_classifier_thread and self.path_file_classifier_thread.isRunning():
            self.show_message("Przetwarzanie pliku w toku", "Zakończ lub zatrzymaj bieżące przetwarzanie pliku ścieżki.")
            return False
        if self.static_model_test_thread and self.static_model_test_thread.isRunning():
            self.show_message("Testowanie statyczne w toku", "Zakończ lub zatrzymaj bieżące testowanie modelu statycznego.")
            return False
        if self.path_model_test_thread and self.path_model_test_thread.isRunning():
            self.show_message("Testowanie ścieżek w toku", "Zakończ lub zatrzymaj bieżące testowanie modelu ścieżek.")
            return False
        if self.static_training_thread and self.static_training_thread.isRunning():
            self.show_message("Trening w toku", "Zakończ lub poczekaj na zakończenie treningu statycznego.")
            return False
        if self.path_training_thread and self.path_training_thread.isRunning():
            self.show_message("Trening w toku", "Zakończ lub poczekaj na zakończenie treningu ścieżek.")
            return False


        # Reset elementów UI
        self.combined_result = "---"
        self.combined_result_label.setText(f"Wynik: {self.combined_result}")
        self.history_text_edit.clear()
        self.path_prediction_label.setText("Predykcja ścieżki (na żywo): ---")
        if self.static_image_classification_result_label: 
            self.static_image_classification_result_label.setText("Wynik klasyfikacji z obrazu: ---")
     
        if self.static_test_results_area: 
            self.static_test_results_area.clear()
            self.static_confusion_matrix_label.setText("Metryki (macierz pomyłek, wykresy) zostaną zapisane do plików po teście.")
            self.static_confusion_matrix_label.setPixmap(QPixmap())
        if self.path_test_results_area:
            self.path_test_results_area.clear()
            self.path_confusion_matrix_label.setText("Metryki (macierz pomyłek, wykresy) zostaną zapisane do plików po teście.")
            self.path_confusion_matrix_label.setPixmap(QPixmap())

        self.current_mode = new_mode
        self.is_recording_path = False
        self.current_path_points = []
        self.current_recording_path_label = None

        if new_mode == 'idle':
            self.stop_camera()
        else:
            models_ok = True
            error_msg = ""
            if new_mode == 'classify_static' and (not self.static_model or not self.static_label_encoder):
                error_msg = "Model statyczny nie jest załadowany."
                models_ok = False
            elif new_mode == 'classify_path' and (not self.path_model or not self.path_label_encoder or not self.path_scaler):
                error_msg = "Model ścieżek lub skaler nie są załadowane."
                models_ok = False
            elif new_mode == 'combined' and (not self.static_model or not self.static_label_encoder or 
                                            not self.path_model or not self.path_label_encoder or not self.path_scaler):
                error_msg = "Oba modele (statyczny i ścieżek) oraz skaler muszą być załadowane dla trybu kombinowanego."
                models_ok = False
            
            if not models_ok:
                self.show_error("Błąd modelu", error_msg + " Proszę załadować lub wytrenować odpowiednie modele.")
                self.current_mode = 'idle'
                self._update_button_states()
                return False

            self.camera_thread = CameraThread(mode=new_mode)
            self._connect_camera_signals()
            if new_mode in ['classify_static', 'combined']:
                self.camera_thread.load_static_model(self.static_model, self.static_label_encoder)
            if new_mode in ['classify_path', 'combined']:
                self.camera_thread.load_path_model(self.path_model, self.path_label_encoder, self.path_scaler)
            
            self.camera_thread.start()
            self.statusBar().showMessage(f"Tryb zmieniony na: {new_mode}. Kamera uruchomiona.")
            if new_mode in ['collect_static', 'collect_path', 'classify_path']:
                self.setFocus()

        self._update_button_states()
        return True

    def _update_button_states(self):
        mode = self.current_mode
        is_idle = mode == 'idle'
        
        is_static_training = self.static_training_thread is not None and self.static_training_thread.isRunning()
        is_path_training = self.path_training_thread is not None and self.path_training_thread.isRunning()
        is_testing_static = self.static_model_test_thread is not None and self.static_model_test_thread.isRunning()
        is_testing_path = self.path_model_test_thread is not None and self.path_model_test_thread.isRunning()
        is_cam_running = self.camera_thread is not None and self.camera_thread.isRunning()
        is_path_file_classifying = self.path_file_classifier_thread is not None and self.path_file_classifier_thread.isRunning()

        is_any_long_operation = (is_static_training or is_path_training or 
                                 is_testing_static or is_testing_path or
                                 is_cam_running or is_path_file_classifying)
        can_start_generic_action = is_idle and not is_any_long_operation

        if hasattr(self, 'start_static_collect_button'):
            self.start_static_collect_button.setEnabled(can_start_generic_action)
        if hasattr(self, 'stop_static_collect_button'):
            self.stop_static_collect_button.setEnabled(mode == 'collect_static' and is_cam_running)
        if hasattr(self, 'static_train_button'):
            self.static_train_button.setEnabled(can_start_generic_action and os.path.exists(STATIC_CSV_FILE) and os.path.getsize(STATIC_CSV_FILE) > 0)
        if hasattr(self, 'start_static_classify_button'):
            self.start_static_classify_button.setEnabled(can_start_generic_action and self.static_model is not None)
        if hasattr(self, 'stop_static_classify_button'):
            self.stop_static_classify_button.setEnabled(mode == 'classify_static' and is_cam_running)
        if hasattr(self, 'classify_static_image_button'):
            self.classify_static_image_button.setEnabled(can_start_generic_action and self.static_model is not None)
        
        if hasattr(self, 'test_static_model_button'):
            self.test_static_model_button.setEnabled(can_start_generic_action and self.static_model is not None)
        if hasattr(self, 'stop_test_static_model_button'):
            self.stop_test_static_model_button.setEnabled(is_testing_static)

        if hasattr(self, 'start_path_collect_button'):
            self.start_path_collect_button.setEnabled(can_start_generic_action)
        if hasattr(self, 'stop_path_collect_button'):
            self.stop_path_collect_button.setEnabled(mode == 'collect_path' and is_cam_running)
        if hasattr(self, 'path_train_button'):
            self.path_train_button.setEnabled(can_start_generic_action and os.path.exists(PATH_CSV_FILE) and os.path.getsize(PATH_CSV_FILE) > 0)
        if hasattr(self, 'start_path_classify_button'):
            self.start_path_classify_button.setEnabled(can_start_generic_action and self.path_model is not None)
        if hasattr(self, 'stop_path_classify_button'):
            self.stop_path_classify_button.setEnabled(mode == 'classify_path' and is_cam_running)
        
        if hasattr(self, 'classify_path_from_file_button'):
            self.classify_path_from_file_button.setEnabled(
                can_start_generic_action and
                self.path_model is not None and
                self.path_label_encoder is not None and
                self.path_scaler is not None
            )
        if hasattr(self, 'stop_classify_path_from_file_button'):
            self.stop_classify_path_from_file_button.setEnabled(is_path_file_classifying)

        if hasattr(self, 'test_path_model_button'):
            self.test_path_model_button.setEnabled(
                can_start_generic_action and 
                self.path_model is not None and
                self.path_label_encoder is not None and
                self.path_scaler is not None
            )
        if hasattr(self, 'stop_test_path_model_button'):
            self.stop_test_path_model_button.setEnabled(is_testing_path)
        
        if hasattr(self, 'path_label_select_combo'):
             self.path_label_select_combo.setEnabled(
                 (is_idle or mode == 'collect_path' or mode == 'classify_path') and 
                 not is_any_long_operation 
             )

        if hasattr(self, 'start_combined_button'):
            self.start_combined_button.setEnabled(can_start_generic_action and self.static_model is not None and self.path_model is not None)
        if hasattr(self, 'stop_combined_button'):
            self.stop_combined_button.setEnabled(mode == 'combined' and is_cam_running)

    def stop_camera(self): 
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
        
        if self.current_mode != 'idle':
            self.current_mode = 'idle'
        
        self.is_recording_path = False
        self.current_path_points = []
        
        self.camera_label.setText("Kamera zatrzymana. Wybierz tryb, aby rozpocząć.")
        self.camera_label.setPixmap(QPixmap()) 
        self.statusBar().showMessage("Kamera zatrzymana. System w trybie jałowym.")
        
        # Reset result labels
        self.combined_result_label.setText("Wynik: ---")
        self.path_prediction_label.setText("Predykcja ścieżki (na żywo): ---")
        self.history_text_edit.clear()
        if self.static_image_classification_result_label: 
            self.static_image_classification_result_label.setText("Wynik klasyfikacji z obrazu: ---")
        
        if hasattr(self, 'static_test_results_area') and self.static_test_results_area:
            self.static_test_results_area.clear()
        if hasattr(self, 'static_confusion_matrix_label') and self.static_confusion_matrix_label:
            self.static_confusion_matrix_label.setText("Metryki (macierz pomyłek, wykresy) zostaną zapisane do plików po teście.")
            self.static_confusion_matrix_label.setPixmap(QPixmap())
        
        if hasattr(self, 'path_test_results_area') and self.path_test_results_area:
            self.path_test_results_area.clear()
        if hasattr(self, 'path_confusion_matrix_label') and self.path_confusion_matrix_label:
            self.path_confusion_matrix_label.setText("Metryki (macierz pomyłek, wykresy) zostaną zapisane do plików po teście.")
            self.path_confusion_matrix_label.setPixmap(QPixmap())
            
        self._update_button_states()

    def _connect_camera_signals(self): 
        if not self.camera_thread: return
        try: self.camera_thread.frame_signal.disconnect(self.update_frame)
        except TypeError: pass
        try: self.camera_thread.static_landmarks_signal.disconnect(self.save_static_landmarks)
        except TypeError: pass
        try: self.camera_thread.path_point_signal.disconnect(self.append_path_point)
        except TypeError: pass
        try: self.camera_thread.combined_path_ready_signal.disconnect(self.handle_combined_path)
        except TypeError: pass
        try: self.camera_thread.path_recording_started_signal.disconnect(self._on_path_recording_started)
        except TypeError: pass
        try: self.camera_thread.path_recording_stopped_signal.disconnect(self._on_path_recording_stopped)
        except TypeError: pass
        try: self.camera_thread.status_signal.disconnect(self.update_status)
        except TypeError: pass
        try: self.camera_thread.error_signal.disconnect(self.show_error_message_box)
        except TypeError: pass
        try: self.camera_thread.finished.disconnect(self._on_camera_thread_finished)
        except TypeError: pass

        self.camera_thread.frame_signal.connect(self.update_frame)
        self.camera_thread.static_landmarks_signal.connect(self.save_static_landmarks)
        self.camera_thread.path_point_signal.connect(self.append_path_point)
        self.camera_thread.combined_path_ready_signal.connect(self.handle_combined_path)
        self.camera_thread.path_recording_started_signal.connect(self._on_path_recording_started)
        self.camera_thread.path_recording_stopped_signal.connect(self._on_path_recording_stopped)
        self.camera_thread.status_signal.connect(self.update_status)
        self.camera_thread.error_signal.connect(self.show_error_message_box)
        self.camera_thread.finished.connect(self._on_camera_thread_finished)


    def _on_camera_thread_finished(self): 
        is_intentional_stop = (self.current_mode == 'idle')
        
        self.camera_thread = None 

        if not is_intentional_stop:
            self.statusBar().showMessage("Wątek kamery nieoczekiwanie zakończył działanie. Przełączanie na tryb jałowy.")
            self.current_mode = 'idle' 
            self.camera_label.setText("Problem z kamerą lub wątek zakończony. Wybierz tryb ponownie.")
            self.camera_label.setPixmap(QPixmap())
        self._update_button_states()


    def start_static_collection_mode(self): self._set_mode('collect_static') 
    def start_static_classification_mode(self): self._set_mode('classify_static') 

    def save_static_landmarks(self, data): 
        if len(data) > 1:
            label, landmarks = data[0], data[1:]
            if save_static_to_csv(STATIC_CSV_FILE, label, landmarks):
                self.static_data_counts[label] = self.static_data_counts.get(label, 0) + 1
                self._update_data_count_display()
                self.update_status(f"Zapisano gest statyczny: '{label}'. Naciśnij klawisz dla kolejnego lub Zakończ.")
            else:
                self.show_error_message_box("Błąd zapisu", f"Nie udało się zapisać danych statycznych dla gestu '{label}'.")
        else:
            self.show_warning_message_box("Błąd danych", "Otrzymano niekompletne dane dla gestu statycznego.")

    def start_static_training(self): 
        if self.static_training_thread and self.static_training_thread.isRunning():
            self.show_message("Trening w toku", "Trening modelu statycznego jest już uruchomiony.")
            return
        if not self._set_mode('idle'): return
        
        if not os.path.exists(STATIC_CSV_FILE) or os.path.getsize(STATIC_CSV_FILE) == 0:
            self.show_error_message_box("Błąd treningu", f"Plik CSV '{STATIC_CSV_FILE}' z danymi statycznymi nie istnieje lub jest pusty.")
            return

        self.statusBar().showMessage("Rozpoczynanie treningu modelu statycznego...")
        self.static_training_thread = StaticTrainingThread(STATIC_CSV_FILE, STATIC_MODEL_FILE, STATIC_ENCODER_FILE)
        self.static_training_thread.progress_signal.connect(self.update_status)
        self.static_training_thread.finished_signal.connect(self.on_static_training_finished)
        self.static_training_thread.start()
        self._update_button_states()

    def on_static_training_finished(self, success, message): 
        if success:
            self.show_message("Trening statyczny zakończony", message) 
            self._load_models_and_encoders() 
        else:
            self.show_error_message_box("Błąd treningu statycznego", message)
        self.static_training_thread = None
        self._update_button_states()
        self.statusBar().showMessage(message if success else f"Błąd treningu statycznego: {message}")


    def classify_static_from_image(self):
        if not self._set_mode('idle'): return

        if not self.static_model or not self.static_label_encoder:
            self.show_error_message_box("Model niezaładowany", "Model statyczny nie jest dostępny. Proszę go najpierw wytrenować lub załadować.")
            if self.static_image_classification_result_label:
                self.static_image_classification_result_label.setText("Model statyczny niezaładowany.")
            return
        
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(self, "Otwórz plik obrazu", "", "Pliki obrazów (*.png *.jpg *.jpeg *.bmp);;Wszystkie pliki (*)", options=options)
        
        if not filepath:
            if self.static_image_classification_result_label:
                self.static_image_classification_result_label.setText("Anulowano wybór obrazu.")
            self.statusBar().showMessage("Anulowano wybór obrazu.")
            return

        self.statusBar().showMessage(f"Klasyfikowanie gestu statycznego z obrazu: {os.path.basename(filepath)}...")
        img_to_display_on_camera_label = None
        prediction_text_for_label = "---" 

        try:
            img = cv2.imread(filepath)
            if img is None:
                raise ValueError("Nie można wczytać obrazu. Sprawdź ścieżkę lub format pliku.")
            
            img_to_display_on_camera_label = img.copy()
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb_image.flags.writeable = False
            results = hands.process(rgb_image)
            rgb_image.flags.writeable = True

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(img_to_display_on_camera_label, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing_styles.get_default_hand_landmarks_style(),
                                          mp_drawing_styles.get_default_hand_connections_style())
                
                processed_landmarks = process_static_landmarks(hand_landmarks)
                if processed_landmarks:
                    input_data = np.array([processed_landmarks], dtype=np.float32)
                    prediction = self.static_model.predict(input_data, verbose=0)[0]
                    idx = np.argmax(prediction)
                    conf = prediction[idx]
                    label = self.static_label_encoder.inverse_transform([idx])[0]
                    prediction_text_for_label = f"Gest: {label} (Pewność: {conf:.2f})"
                else:
                    prediction_text_for_label = "Wykryto dłoń, ale punkty kluczowe są nieprawidłowe."
            else:
                prediction_text_for_label = "Nie wykryto dłoni na obrazie."
            
            if self.static_image_classification_result_label:
                self.static_image_classification_result_label.setText(prediction_text_for_label)

            if img_to_display_on_camera_label is not None:
                qt_image = convert_cv_qt(img_to_display_on_camera_label)
                scaled_pixmap = QPixmap.fromImage(qt_image).scaled(self.camera_label.width(), self.camera_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.camera_label.setPixmap(scaled_pixmap)
            else: 
                self.camera_label.setText("Nie można wyświetlić obrazu.")
                self.camera_label.setPixmap(QPixmap())

            self.statusBar().showMessage(f"Klasyfikacja obrazu zakończona: {prediction_text_for_label}")
        except Exception as e:
            error_message = f"Błąd klasyfikacji: {e}"
            self.show_error_message_box("Błąd klasyfikacji obrazu (statyczna)", error_message)
            self.statusBar().showMessage("Błąd podczas klasyfikacji gestu statycznego z obrazu.")
            if self.static_image_classification_result_label:
                 self.static_image_classification_result_label.setText(error_message[:100])
            self.camera_label.setText("Błąd ładowania/przetwarzania obrazu.")
            self.camera_label.setPixmap(QPixmap())
        finally:
             self._update_button_states()

    def start_static_model_test(self):
        if not self._set_mode('idle'): return 

        if not self.static_model or not self.static_label_encoder:
            self.show_error_message_box("Model niezaładowany", "Model statyczny lub enkoder nie są załadowane. Wytrenuj lub załaduj model.")
            return

        folder_path = QFileDialog.getExistingDirectory(self, "Wybierz folder z obrazami testowymi (z podfolderami A, B, C...)", ".")
        if not folder_path:
            self.statusBar().showMessage("Anulowano wybór folderu.")
            return

        self.statusBar().showMessage("Rozpoczynanie testowania modelu statycznego...")
        self.static_test_results_area.clear()
        self.static_confusion_matrix_label.setText("Przetwarzanie (statyczne)... Metryki i wykresy zostaną zapisane do plików.")
        self.static_confusion_matrix_label.setPixmap(QPixmap())

        self.static_model_test_thread = StaticModelTestThread(folder_path, self.static_model, self.static_label_encoder)
        self.static_model_test_thread.progress_signal.connect(self.update_status) 
        self.static_model_test_thread.finished_signal.connect(self.on_static_model_test_finished)
        
        self.static_model_test_thread.start()
        self._update_button_states()

    def stop_static_model_test(self):
        if self.static_model_test_thread and self.static_model_test_thread.isRunning():
            self.static_model_test_thread.stop() 
            self.statusBar().showMessage("Zatrzymywanie testu modelu statycznego...")
        else:
            self.statusBar().showMessage("Brak aktywnego testu statycznego do zatrzymania.")
            self._update_button_states()

    def on_static_model_test_finished(self, success, accuracy, report_str, raw_conf_matrix, class_labels,
                                      class_metrics_plot_fn, norm_cm_fn, roc_auc_fn, pr_curve_fn):
        thread_was_running = self.static_model_test_thread is not None
        self.static_model_test_thread = None 
        
        saved_plots_info = []
        status_label_text_parts = []

        if success:
            results_text = f"Testowanie modelu statycznego zakończone.\n"
            results_text += f"Całkowita dokładność (Accuracy): {accuracy:.4f}\n\n"
            results_text += "Raport Klasyfikacji (Statyczny):\n"
            results_text += report_str
            self.static_test_results_area.setText(results_text)
            self.statusBar().showMessage(f"Test statyczny zakończony. Dokładność: {accuracy:.4f}.")

            raw_cm_filename = None
            if raw_conf_matrix.size > 0 and class_labels:
                raw_cm_filename = self.plot_confusion_matrix(raw_conf_matrix, class_labels,
                                                             target_label_for_status_update=self.static_confusion_matrix_label,
                                                             default_filename_prefix="static_raw_confusion_matrix")
            if raw_cm_filename and os.path.exists(raw_cm_filename):
                saved_plots_info.append(f"Surowa macierz pomyłek: {os.path.basename(raw_cm_filename)}")
                status_label_text_parts.append(f"Surowa macierz pomyłek: {os.path.basename(raw_cm_filename)}")
            elif raw_conf_matrix.size > 0 :
                status_label_text_parts.append(self.static_confusion_matrix_label.text())
            else:
                 status_label_text_parts.append("Surowa macierz pomyłek: Brak danych do wygenerowania.")


            if class_metrics_plot_fn and os.path.exists(class_metrics_plot_fn):
                saved_plots_info.append(f"Wykres metryk: {os.path.basename(class_metrics_plot_fn)}")
            if norm_cm_fn and os.path.exists(norm_cm_fn):
                saved_plots_info.append(f"Znormalizowana macierz pomyłek: {os.path.basename(norm_cm_fn)}")
            if roc_auc_fn and os.path.exists(roc_auc_fn):
                saved_plots_info.append(f"Krzywa ROC (makro): {os.path.basename(roc_auc_fn)}")
            if pr_curve_fn and os.path.exists(pr_curve_fn):
                saved_plots_info.append(f"Krzywa P-R (makro): {os.path.basename(pr_curve_fn)}")
            

            if saved_plots_info:
                self.static_confusion_matrix_label.setText("Zapisano następujące wykresy:\n" + "\n".join(saved_plots_info))
            elif not status_label_text_parts: # No raw CM, no other plots
                 self.static_confusion_matrix_label.setText("Brak wykresów do zapisania.")


        else:
            self.static_test_results_area.setText(f"Testowanie modelu statycznego nie powiodło się lub zostało przerwane.\nSzczegóły: {report_str}")
            error_msg_label = "Błąd podczas generowania/zapisu metryk."
            if class_metrics_plot_fn and os.path.exists(class_metrics_plot_fn): saved_plots_info.append(f"Wykres metryk: {os.path.basename(class_metrics_plot_fn)}")
            if norm_cm_fn and os.path.exists(norm_cm_fn): saved_plots_info.append(f"Znormalizowana macierz pomyłek: {os.path.basename(norm_cm_fn)}")
            if roc_auc_fn and os.path.exists(roc_auc_fn): saved_plots_info.append(f"Krzywa ROC (makro): {os.path.basename(roc_auc_fn)}")
            if pr_curve_fn and os.path.exists(pr_curve_fn): saved_plots_info.append(f"Krzywa P-R (makro): {os.path.basename(pr_curve_fn)}")
            if saved_plots_info: error_msg_label += "\nCzęściowo zapisane wykresy:\n" + "\n".join(saved_plots_info)
            self.static_confusion_matrix_label.setText(error_msg_label)
            
            error_prefix = "Testowanie statyczne nie powiodło się" if thread_was_running and not "przerwane" in report_str.lower() else "Testowanie statyczne zakończone z błędem"
            self.statusBar().showMessage(f"{error_prefix}: {report_str[:100]}")
        
        self._update_button_states() 

    def plot_confusion_matrix(self, cm_data, class_names, target_label_for_status_update, default_filename_prefix):
        if not class_names: 
            target_label_for_status_update.setText("Brak nazw klas, nie można zapisać surowej macierzy pomyłek.")
            return None

        plot_filename = f"{default_filename_prefix}_hd.png" 
        
        try:
            num_classes = len(class_names)
            dpi = DPI 
            base_fig_width_inches = max(FIGSIZE_HD[0] * 0.7, num_classes * 0.6 + 2)
            fig_height_inches = FIGSIZE_HD[1] * 0.7 

            if base_fig_width_inches > 25:
                 fig_height_inches = max(fig_height_inches, num_classes * 0.3 + 2)


            annot_font_size = max(5, 10 - int(num_classes * 0.1))
            tick_font_size = max(6, 9 - int(num_classes * 0.1))

            fig, ax = plt.subplots(figsize=(base_fig_width_inches, fig_height_inches), dpi=dpi)
            cmap_name = 'Oranges' if "static" in default_filename_prefix else 'Purples' 
            
            sns.heatmap(cm_data, annot=True, fmt='d', cmap=cmap_name, ax=ax, 
                        xticklabels=class_names, yticklabels=class_names, annot_kws={"size": annot_font_size}) 
            
            title_prefix = "Statyczna (Surowa)" if "static" in default_filename_prefix else "Ścieżki (Surowa)"
            ax.set_title(f'Macierz pomyłek ({title_prefix})', fontsize=tick_font_size + 2)
            ax.set_ylabel('Prawdziwa Etykieta', fontsize=tick_font_size)
            ax.set_xlabel('Przewidziana Etykieta', fontsize=tick_font_size)
            plt.xticks(rotation=45, ha='right', fontsize=tick_font_size)
            plt.yticks(rotation=0, fontsize=tick_font_size)
            plt.tight_layout(pad=2.5)

            fig.savefig(plot_filename) 
            plt.close(fig) 
            
            status_msg = f"Surowa macierz pomyłek zapisana do:\n{os.path.basename(plot_filename)}"
            target_label_for_status_update.setText(status_msg)
            return plot_filename
        except Exception as e:
            status_msg = f"Błąd zapisu surowej macierzy: {e}"
            target_label_for_status_update.setText(status_msg)
            self.show_error_message_box("Błąd Zapisu Wykresu", f"Nie udało się zapisać surowej macierzy pomyłek: {e}")
            import traceback
            traceback.print_exc()
            return None

    def start_path_collection_mode(self): self._set_mode('collect_path') 
    def start_path_classification_mode(self): self._set_mode('classify_path') 
    def _update_selected_path_label(self, label): self.selected_path_label = label 

    def append_path_point(self, point):
        if self.is_recording_path or \
           (self.camera_thread and self.camera_thread.mode == 'combined' and self.camera_thread.is_recording_path) :
             self.current_path_points.append(point) 
             self.camera_label.update()

    def process_and_save_path(self):
        if not self.current_recording_path_label: 
            self.statusBar().showMessage("Ścieżka bez etykiety, odrzucono.") 
            return

        if len(self.current_path_points) < 2:
            self.statusBar().showMessage(f"Ścieżka '{self.current_recording_path_label}' zbyt krótka ({len(self.current_path_points)} pkt.), odrzucono.") 
            return

        self.statusBar().showMessage(f"Przetwarzanie ścieżki '{self.current_recording_path_label}' ({len(self.current_path_points)} punktów)...") 
        standardized_flat = standardize_path(self.current_path_points) 
        
        if standardized_flat is not None: 
            if save_path_to_csv(PATH_CSV_FILE, self.current_recording_path_label, standardized_flat): 
                self.path_data_counts[self.current_recording_path_label] = self.path_data_counts.get(self.current_recording_path_label, 0) + 1 
                self._update_data_count_display() 
                self.statusBar().showMessage(f"Ścieżka '{self.current_recording_path_label}' zapisana pomyślnie.") 
            else:
                self.show_error_message_box("Błąd zapisu", f"Nie udało się zapisać danych ścieżki dla '{self.current_recording_path_label}'.") 
        else:
            self.show_warning_message_box("Błąd przetwarzania ścieżki", f"Nie można było zestandaryzować ścieżki dla '{self.current_recording_path_label}'.") 

    def classify_live_path(self):
        if not self.path_model or not self.path_label_encoder or not self.path_scaler: 
            self.statusBar().showMessage("Model ścieżek, enkoder lub skaler nie są załadowane.") 
            self.path_prediction_label.setText("Predykcja ścieżki (na żywo): Błąd modelu") 
            return

        if len(self.current_path_points) < 2:
            self.statusBar().showMessage("Ścieżka zbyt krótka do klasyfikacji.") 
            self.path_prediction_label.setText("Predykcja ścieżki (na żywo): Za krótka") 
            return

        self.statusBar().showMessage(f"Klasyfikowanie ścieżki na żywo ({len(self.current_path_points)} punktów)...") 
        standardized_flat = standardize_path(self.current_path_points) 
        
        if standardized_flat is not None: 
            try:
                input_data = self.path_scaler.transform(standardized_flat.reshape(1, -1)) 
                prediction = self.path_model.predict(input_data, verbose=0)[0] 
                idx = np.argmax(prediction) 
                conf = prediction[idx] 
                label = self.path_label_encoder.inverse_transform([idx])[0] 
                pred_text = f"Ścieżka: {label} ({conf:.2f})" 
                self.path_prediction_label.setText(pred_text) 
                self.statusBar().showMessage(f"Wynik klasyfikacji ścieżki (na żywo): {label} (Pewność: {conf:.2f})") 
            except Exception as e:
                self.show_error_message_box("Błąd klasyfikacji ścieżki (na żywo)", f"Wystąpił błąd: {e}") 
                self.path_prediction_label.setText("Predykcja ścieżki (na żywo): Błąd") 
                self.statusBar().showMessage("Błąd podczas klasyfikacji ścieżki (na żywo).") 
        else:
            self.show_warning_message_box("Błąd przetwarzania ścieżki (na żywo)", "Nie można było zestandaryzować ścieżki do klasyfikacji.") 
            self.path_prediction_label.setText("Predykcja ścieżki (na żywo): Błąd przetwarzania") 
            self.statusBar().showMessage("Błąd przetwarzania podczas klasyfikacji ścieżki (na żywo).") 

    def start_path_training(self):
        if self.path_training_thread and self.path_training_thread.isRunning(): 
            self.show_message("Trening w toku", "Trening modelu ścieżek jest już uruchomiony.") 
            return
        if not self._set_mode('idle'): return

        if not os.path.exists(PATH_CSV_FILE) or os.path.getsize(PATH_CSV_FILE) == 0: 
            self.show_error_message_box("Błąd treningu", f"Plik CSV '{PATH_CSV_FILE}' z danymi ścieżek nie istnieje lub jest pusty.") 
            return
        
        self.statusBar().showMessage("Rozpoczynanie treningu modelu ścieżek...") 
        self.path_training_thread = PathTrainingThread(PATH_CSV_FILE, PATH_MODEL_FILE, PATH_ENCODER_FILE, PATH_SCALER_FILE) 
        self.path_training_thread.progress_signal.connect(self.update_status) 
        self.path_training_thread.finished_signal.connect(self.on_path_training_finished) 
        self.path_training_thread.start() 
        self._update_button_states() 

    def on_path_training_finished(self, success, message):
        if success: 
            self.show_message("Trening ścieżek zakończony", message) 
            self._load_models_and_encoders() 
        else:
            self.show_error_message_box("Błąd treningu ścieżek", message) 
        self.path_training_thread = None 
        self._update_button_states() 
        self.statusBar().showMessage(message if success else f"Błąd treningu ścieżek: {message}")


    def start_combined_mode(self): self._set_mode('combined') 

    def handle_combined_path(self, path_points, static_label):
        self.statusBar().showMessage(f"Otrzymano ścieżkę dla gestu statycznego '{static_label}'. Klasyfikowanie ścieżki...") 
        final_result = static_label

        if not self.path_model or not self.path_label_encoder or not self.path_scaler: 
             self.statusBar().showMessage("Model ścieżek/enkoder/skaler niezaładowany dla trybu kombinowanego.") 
        elif len(path_points) < 2:
            self.statusBar().showMessage("Otrzymana ścieżka w trybie połączonym jest zbyt krótka.") 
        else:
            standardized_flat = standardize_path(path_points) 
            if standardized_flat is not None: 
                try:
                    input_data = self.path_scaler.transform(standardized_flat.reshape(1, -1)) 
                    prediction = self.path_model.predict(input_data, verbose=0)[0] 
                    idx = np.argmax(prediction) 
                    conf = prediction[idx] 
                    path_label_pred = self.path_label_encoder.inverse_transform([idx])[0] 
                    
                    self.statusBar().showMessage(f"Gest statyczny: {static_label}, Ścieżka: {path_label_pred} (Pewność: {conf:.2f}). Stosowanie reguł...") 
                    combination_key = (static_label, path_label_pred) 
                    final_result = COMBINATION_RULES.get(combination_key, static_label)
                except Exception as e:
                    self.show_error_message_box("Błąd klasyfikacji kombinowanej", f"Błąd podczas klasyfikacji ścieżki w trybie połączonym: {e}") 
                    self.statusBar().showMessage("Błąd klasyfikacji ścieżki w trybie połączonym.") 
            else:
                self.show_warning_message_box("Błąd przetwarzania kombinowanego", "Nie można było zestandaryzować ścieżki w trybie połączonym.") 
                self.statusBar().showMessage("Błąd przetwarzania ścieżki w trybie połączonym.") 

        self.combined_result = final_result 
        self.combined_result_label.setText(f"Wynik: {self.combined_result}") 
        self.history_text_edit.append(self.combined_result) 

    def start_classify_path_from_file(self):
        if not self._set_mode('idle'): return

        if not self.path_model or not self.path_label_encoder or not self.path_scaler:
            self.show_error_message_box("Model niezaładowany", "Model ścieżek, enkoder lub skaler nie są dostępne.")
            self.statusBar().showMessage("Błąd modelu dla klasyfikacji z pliku.")
            return

        options = QFileDialog.Options()
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Wybierz plik wideo lub sekwencję obrazów", "",
                                                       "Pliki wideo (*.mp4 *.avi *.mov *.mkv);;Pliki obrazów (*.png *.jpg *.jpeg *.bmp);;Wszystkie pliki (*)",
                                                       options=options)
        if not file_paths:
            self.statusBar().showMessage("Anulowano wybór pliku/plików.")
            return

        self.statusBar().showMessage("Rozpoczynanie przetwarzania plików ścieżki...")

        self.path_file_classifier_thread = PathFileClassifierThread(
            file_paths, self.path_model, self.path_label_encoder, self.path_scaler
        )
        self.path_file_classifier_thread.progress_signal.connect(self._on_path_file_classification_progress)
        self.path_file_classifier_thread.finished_signal.connect(self._on_path_file_classification_finished)
        self.path_file_classifier_thread.error_signal.connect(self._on_path_file_classification_error)
        self.path_file_classifier_thread.start()
        self._update_button_states()

    def stop_path_file_classification(self):
        if self.path_file_classifier_thread and self.path_file_classifier_thread.isRunning():
            self.path_file_classifier_thread.stop()
            self.statusBar().showMessage("Wysyłanie sygnału zatrzymania do przetwarzania plików...")
        else:
            self.statusBar().showMessage("Brak aktywnego przetwarzania plików do zatrzymania.")
            self._update_button_states()

    def _on_path_file_classification_progress(self, message):
        self.statusBar().showMessage(message)

    def _on_path_file_classification_finished(self, success, result_text, status_message):
        if success:
            self.show_message("Klasyfikacja z pliku zakończona", f"Wynik: {result_text}\nStatus: {status_message}")
        else:
            self.show_warning_message_box("Klasyfikacja z pliku", f"Problem: {result_text}\nStatus: {status_message}")
        self.statusBar().showMessage(status_message)
        
        if self.path_file_classifier_thread: 
             self.path_file_classifier_thread = None
        self._update_button_states()

    def _on_path_file_classification_error(self, error_message):
        self.show_error_message_box("Błąd przetwarzania plików ścieżki", error_message)
        self.statusBar().showMessage(f"Błąd krytyczny (plik): {error_message[:100]}")
        if self.path_file_classifier_thread:
            self.path_file_classifier_thread = None
        self._update_button_states()

    def start_path_model_test(self):
        if not self._set_mode('idle'): return

        if not self.path_model or not self.path_label_encoder or not self.path_scaler:
            self.show_error_message_box("Model niezaładowany", "Model ścieżek, enkoder lub skaler nie są załadowane. Wytrenuj lub załaduj.")
            return

        folder_path = QFileDialog.getExistingDirectory(self, "Wybierz folder z filmami/sekwencjami testowymi dla ścieżek (podfoldery: J, left...)", ".")
        if not folder_path:
            self.statusBar().showMessage("Anulowano wybór folderu dla testu ścieżek.")
            return

        self.statusBar().showMessage("Rozpoczynanie testowania modelu ścieżek...")
        self.path_test_results_area.clear()
        self.path_confusion_matrix_label.setText("Przetwarzanie (ścieżki)... Metryki i wykresy zostaną zapisane do plików.")
        self.path_confusion_matrix_label.setPixmap(QPixmap())

        self.path_model_test_thread = PathModelTestThread(
            folder_path, self.path_model, self.path_label_encoder, self.path_scaler
        )
        self.path_model_test_thread.progress_signal.connect(self.update_status) 
        self.path_model_test_thread.finished_signal.connect(self.on_path_model_test_finished)
        
        self.path_model_test_thread.start()
        self._update_button_states()

    def stop_path_model_test(self):
        if self.path_model_test_thread and self.path_model_test_thread.isRunning():
            self.path_model_test_thread.stop() 
            self.statusBar().showMessage("Zatrzymywanie testu modelu ścieżek...")
        else:
            self.statusBar().showMessage("Brak aktywnego testu ścieżek do zatrzymania.")
            self._update_button_states()

    def on_path_model_test_finished(self, success, accuracy, report_str, raw_conf_matrix, class_labels,
                                    class_metrics_plot_fn, norm_cm_fn, roc_auc_fn, pr_curve_fn):
        thread_was_running = self.path_model_test_thread is not None
        self.path_model_test_thread = None 
        
        saved_plots_info = []
        status_label_text_parts = []

        if success:
            results_text = f"Testowanie modelu ścieżek zakończone.\n"
            results_text += f"Całkowita dokładność (Accuracy): {accuracy:.4f}\n\n"
            results_text += "Raport Klasyfikacji (Ścieżki):\n"
            results_text += report_str
            self.path_test_results_area.setText(results_text)
            self.statusBar().showMessage(f"Test ścieżek zakończony. Dokładność: {accuracy:.4f}.")

            raw_cm_filename = None
            if raw_conf_matrix.size > 0 and class_labels:
                raw_cm_filename = self.plot_confusion_matrix(raw_conf_matrix, class_labels,
                                                             target_label_for_status_update=self.path_confusion_matrix_label,
                                                             default_filename_prefix="path_raw_confusion_matrix")
            if raw_cm_filename and os.path.exists(raw_cm_filename):
                saved_plots_info.append(f"Surowa macierz pomyłek: {os.path.basename(raw_cm_filename)}")
                status_label_text_parts.append(f"Surowa macierz pomyłek: {os.path.basename(raw_cm_filename)}")

            elif raw_conf_matrix.size > 0 :
                status_label_text_parts.append(self.path_confusion_matrix_label.text())
            else:
                 status_label_text_parts.append("Surowa macierz pomyłek: Brak danych.")


            if class_metrics_plot_fn and os.path.exists(class_metrics_plot_fn): saved_plots_info.append(f"Wykres metryk: {os.path.basename(class_metrics_plot_fn)}")
            if norm_cm_fn and os.path.exists(norm_cm_fn): saved_plots_info.append(f"Znormalizowana macierz pomyłek: {os.path.basename(norm_cm_fn)}")
            if roc_auc_fn and os.path.exists(roc_auc_fn): saved_plots_info.append(f"Krzywa ROC (makro): {os.path.basename(roc_auc_fn)}")
            if pr_curve_fn and os.path.exists(pr_curve_fn): saved_plots_info.append(f"Krzywa P-R (makro): {os.path.basename(pr_curve_fn)}")
            
            if saved_plots_info:
                self.path_confusion_matrix_label.setText("Zapisano następujące wykresy:\n" + "\n".join(saved_plots_info))
            elif not status_label_text_parts:
                 self.path_confusion_matrix_label.setText("Brak wykresów do zapisania.")
            
        else:
            self.path_test_results_area.setText(f"Testowanie modelu ścieżek nie powiodło się lub zostało przerwane.\nSzczegóły: {report_str}")
            error_msg_label = "Błąd podczas generowania/zapisu metryk."
            if class_metrics_plot_fn and os.path.exists(class_metrics_plot_fn): saved_plots_info.append(f"Wykres metryk: {os.path.basename(class_metrics_plot_fn)}")
            if norm_cm_fn and os.path.exists(norm_cm_fn): saved_plots_info.append(f"Znormalizowana macierz pomyłek: {os.path.basename(norm_cm_fn)}")
            if roc_auc_fn and os.path.exists(roc_auc_fn): saved_plots_info.append(f"Krzywa ROC (makro): {os.path.basename(roc_auc_fn)}")
            if pr_curve_fn and os.path.exists(pr_curve_fn): saved_plots_info.append(f"Krzywa P-R (makro): {os.path.basename(pr_curve_fn)}")
            if saved_plots_info: error_msg_label += "\nCzęściowo zapisane wykresy:\n" + "\n".join(saved_plots_info)
            self.path_confusion_matrix_label.setText(error_msg_label)
            
            error_prefix = "Testowanie ścieżek nie powiodło się" if thread_was_running and not "przerwane" in report_str.lower() else "Testowanie ścieżek zakończone z błędem"
            self.statusBar().showMessage(f"{error_prefix}: {report_str[:100]}")
            
        self._update_button_states() 

    def update_frame(self, q_image): 
        pixmap = QPixmap.fromImage(q_image)
        
        should_draw_path = False
        if self.current_mode in ['collect_path', 'classify_path'] and self.is_recording_path:
            should_draw_path = True
        elif self.current_mode == 'combined' and self.camera_thread and self.camera_thread.is_recording_path:
            should_draw_path = True
            
        if should_draw_path and len(self.current_path_points) > 1:
            painter = QPainter(pixmap)
            pen = QPen(QColor(0, 120, 255), 3) 
            painter.setPen(pen)
            w, h = pixmap.width(), pixmap.height()
            path_points_scaled = [QPoint(int(p[0] * w), int(p[1] * h)) for p in self.current_path_points]
            for i in range(len(path_points_scaled) - 1):
                painter.drawLine(path_points_scaled[i], path_points_scaled[i+1])
            painter.end()
        
        scaled_pixmap = pixmap.scaled(self.camera_label.width(), self.camera_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.camera_label.setPixmap(scaled_pixmap)

    def _on_path_recording_started(self):
        if self.current_mode == 'combined':
             self.current_path_points = []
             self.camera_label.update()

    def _on_path_recording_stopped(self):
        if self.current_mode == 'combined':
             self.current_path_points = []
             self.camera_label.update() 

    def update_status(self, message): 
        self.statusBar().showMessage(message)

    def show_error_message_box(self, title, message): 
        QMessageBox.critical(self, title, message)
        self.statusBar().showMessage(f"Błąd: {message[:100]}")

    def show_warning_message_box(self, title, message): 
        QMessageBox.warning(self, title, message)
        self.statusBar().showMessage(f"Ostrzeżenie: {message[:100]}")

    def show_message(self, title, message): 
        QMessageBox.information(self, title, message)

    def keyPressEvent(self, event): 
        if event.isAutoRepeat():
            return
        
        long_op_running = (
            (self.static_training_thread and self.static_training_thread.isRunning()) or
            (self.path_training_thread and self.path_training_thread.isRunning()) or
            (self.static_model_test_thread and self.static_model_test_thread.isRunning()) or
            (self.path_model_test_thread and self.path_model_test_thread.isRunning()) or 
            (self.path_file_classifier_thread and self.path_file_classifier_thread.isRunning())
        )
        if long_op_running:
            super().keyPressEvent(event)
            return

        key = event.key()
        key_text = event.text().upper()

        if self.tab_widget.currentWidget() == self.static_tab and \
           self.current_mode == 'collect_static' and key_text in STATIC_ALLOWED_LABELS:
            if self.camera_thread and self.camera_thread.isRunning():
                self.camera_thread.set_static_label(key_text)
            else:
                self.statusBar().showMessage("Kamera nie jest aktywna w trybie zbierania gestów statycznych.")

        elif self.tab_widget.currentWidget() == self.path_tab and \
             self.current_mode in ['collect_path', 'classify_path'] and key == PATH_RECORD_KEY:
            if not self.is_recording_path:
                if self.current_mode == 'collect_path' and not self.selected_path_label:
                    self.show_warning_message_box("Brak etykiety", "Proszę najpierw wybrać etykietę dla zbieranej ścieżki.")
                    return
                
                self.is_recording_path = True
                self.current_recording_path_label = self.selected_path_label if self.current_mode == 'collect_path' else "KlasyfikacjaNaŻywo"
                self.current_path_points = []
                
                if self.camera_thread and self.camera_thread.isRunning():
                    self.camera_thread.set_path_recording(True)
                
                if self.current_mode == 'classify_path':
                    self.path_prediction_label.setText("Predykcja ścieżki (na żywo): Nagrywanie...")
                self.statusBar().showMessage(f"Rozpoczęto nagrywanie ścieżki dla '{self.current_recording_path_label}'...")
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event): 
        if event.isAutoRepeat():
            return
        
        long_op_running = (
            (self.static_training_thread and self.static_training_thread.isRunning()) or
            (self.path_training_thread and self.path_training_thread.isRunning()) or
            (self.static_model_test_thread and self.static_model_test_thread.isRunning()) or
            (self.path_model_test_thread and self.path_model_test_thread.isRunning()) or
            (self.path_file_classifier_thread and self.path_file_classifier_thread.isRunning())
        )
        if long_op_running:
            super().keyReleaseEvent(event)
            return

        key = event.key()

        if self.tab_widget.currentWidget() == self.path_tab and \
           self.current_mode in ['collect_path', 'classify_path'] and \
           key == PATH_RECORD_KEY and self.is_recording_path:
            
            self.is_recording_path = False
            if self.camera_thread and self.camera_thread.isRunning(): 
                self.camera_thread.set_path_recording(False)
            
            if self.current_recording_path_label and len(self.current_path_points) > 0 :
                self.statusBar().showMessage(f"Zakończono nagrywanie ścieżki dla '{self.current_recording_path_label}'. Przetwarzanie...")
                if self.current_mode == 'collect_path':
                    self.process_and_save_path()
                elif self.current_mode == 'classify_path':
                    self.classify_live_path() 
            elif len(self.current_path_points) == 0 and self.current_recording_path_label:
                 self.statusBar().showMessage(f"Nagrywanie ścieżki dla '{self.current_recording_path_label}' przerwane, brak punktów.")
                 if self.current_mode == 'classify_path': self.path_prediction_label.setText("Predykcja ścieżki (na żywo): Anulowano")
            else:
                 self.statusBar().showMessage("Zakończono nagrywanie ścieżki (bez etykiety lub punktów). Odrzucono.")
                 if self.current_mode == 'classify_path': self.path_prediction_label.setText("Predykcja ścieżki (na żywo): Anulowano")

            self.current_path_points = []
            self.current_recording_path_label = None
            self.camera_label.update()
        else:
            super().keyReleaseEvent(event)

    def closeEvent(self, event): 
        self.stop_camera()
        
        if self.static_training_thread and self.static_training_thread.isRunning():
            print("Zatrzymywanie wątku treningu statycznego...")
            self.static_training_thread.quit()
            self.static_training_thread.wait(1000)
        if self.path_training_thread and self.path_training_thread.isRunning():
            print("Zatrzymywanie wątku treningu ścieżek...")
            self.path_training_thread.quit()
            self.path_training_thread.wait(1000)
            
        if self.static_model_test_thread and self.static_model_test_thread.isRunning():
            print("Zatrzymywanie wątku testu modelu statycznego...")
            self.static_model_test_thread.stop() 
            self.static_model_test_thread.wait(1000) 
        if self.path_model_test_thread and self.path_model_test_thread.isRunning():
            print("Zatrzymywanie wątku testu modelu ścieżek...")
            self.path_model_test_thread.stop()
            self.path_model_test_thread.wait(1000)
        if self.path_file_classifier_thread and self.path_file_classifier_thread.isRunning():
            print("Zatrzymywanie wątku klasyfikacji ścieżki z pliku...")
            self.path_file_classifier_thread.stop()
            self.path_file_classifier_thread.wait(2000) 

        from mp_setup import close_hands
        close_hands()
        event.accept()

