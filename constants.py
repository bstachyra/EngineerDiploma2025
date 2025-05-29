# constants.py

from PyQt5.QtCore import Qt

APP_NAME = "Aplikacja Rozpoznawania Gestów Alfabetu PJM"
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 850
CAMERA_FEED_WIDTH = 640
CAMERA_FEED_HEIGHT = 480

DPI = 100
FIGSIZE_HD = (1920 / DPI, 1080 / DPI)

STATIC_CSV_FILE = 'gesture_data.csv'
STATIC_MODEL_FILE = 'gesture_model.h5'
STATIC_ENCODER_FILE = 'label_encoder.pkl'
STATIC_NUM_LANDMARKS = 21
STATIC_NUM_FEATURES = STATIC_NUM_LANDMARKS * 3
STATIC_ALLOWED_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'W', 'Y']
STATIC_NUM_CLASSES = len(STATIC_ALLOWED_LABELS)

PATH_CSV_FILE = 'path_data.csv'
PATH_MODEL_FILE = 'path_model.h5'
PATH_ENCODER_FILE = 'path_label_encoder.pkl'
PATH_SCALER_FILE = 'path_scaler.pkl'
PATH_ALLOWED_LABELS = ['J', 'left', 'right', 'down', 'tail', 'zigzagtail', 'zigzagr', 'zkropka']
PATH_NUM_CLASSES = len(PATH_ALLOWED_LABELS)
PATH_LENGTH = 100
PATH_NUM_FEATURES = PATH_LENGTH * 2
PATH_TRACKING_LANDMARK = 8
PATH_RECORD_KEY = Qt.Key_Q

STATIC_CONFIDENCE_THRESHOLD = 0.90
STATIC_HOLD_DURATION_SEC = 1.0
PATH_MAX_DURATION_SEC = 3.0

COMBINATION_RULES = {
    ('A', 'tail'): 'Ą', 
    ('C', 'down'): 'Ć', 
    ('I', 'J'): 'J', 
    ('N', 'down'): 'Ń',
    ('O', 'down'): 'Ó',
    ('R', 'zigzagr'): 'RZ',
    ('S', 'down'): 'Ś', 
    ('L', 'right'): 'Ł',
    ('B', 'left'): 'SZ',
    ('D', 'zigzagr'): 'Ź',
    ('D', 'zigzagtail'): 'Ź',
    ('D', 'zkropka'): 'Ż'
}