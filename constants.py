# constants.py

from PyQt5.QtCore import Qt

# Podstawowe stałe aplikacji
APP_NAME = "Gesture Recognition App (Tabbed)"
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 850
CAMERA_FEED_WIDTH = 640
CAMERA_FEED_HEIGHT = 480

# Stałe statycznych gestów
STATIC_CSV_FILE = 'gesture_data.csv'
STATIC_MODEL_FILE = 'gesture_model.h5'
STATIC_ENCODER_FILE = 'label_encoder.pkl'
STATIC_NUM_LANDMARKS = 21
STATIC_NUM_FEATURES = STATIC_NUM_LANDMARKS * 3 # x, y, z
STATIC_ALLOWED_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'W', 'Y', 'Z']
STATIC_NUM_CLASSES = len(STATIC_ALLOWED_LABELS)

# Stałe tras gestów
PATH_CSV_FILE = 'path_data.csv'
PATH_MODEL_FILE = 'path_model.h5'
PATH_ENCODER_FILE = 'path_label_encoder.pkl'
PATH_SCALER_FILE = 'path_scaler.pkl'
PATH_ALLOWED_LABELS = ['J', 'left', 'right', 'down', 'tail', 'zigzagl', 'zigzagr', 'zkropka']
PATH_NUM_CLASSES = len(PATH_ALLOWED_LABELS)
PATH_LENGTH = 100
PATH_NUM_FEATURES = PATH_LENGTH * 2 # x, y
PATH_TRACKING_LANDMARK = 8 # Koniuszek palca wskazującego
PATH_RECORD_KEY = Qt.Key_Q # Przycisk do startu nagrywania/klasyfikacji ścieżki

# Stałe trybu połączonego (gest statyczny + ścieżka)
STATIC_CONFIDENCE_THRESHOLD = 0.90 # Minimalna pewność modelu do rozpoczęcie klasyfikacji
STATIC_HOLD_DURATION_SEC = 1.0 # Minimalny czas utrzymania gestu do rozpoczęcia klasyfikacji
PATH_MAX_DURATION_SEC = 3.0 # Maksymalny czas śledzenia ścieżki do klasyfikacji
# Kombinacje statycznego gestu i ścieżki w celu otrzymania danego gestu
COMBINATION_RULES = {
    ('A', 'tail'): 'Ą', ('C', 'down'): 'Ć', ('I', 'J'): 'J', ('N', 'down'): 'Ń',
    ('O', 'down'): 'Ó', ('R', 'zigzagl'): 'RZ', ('S', 'down'): 'Ś', ('L', 'right'): 'Ł',
    ('B', 'left'): 'SZ', ('Z', 'zigzagl'): 'Ź', ('Z', 'zkropka'): 'Ż',
}

