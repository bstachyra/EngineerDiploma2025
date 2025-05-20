# mp_setup.py

import mediapipe as mp

# Inicjalizacja komponentów mediapipe w celu łatwego importu
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Inicjalizacja obiektu Hands
hands = mp_hands.Hands(
    static_image_mode=False, # Ustawiony na False dla wideo
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# Zamknięcie zasobów Mediapipe Hands
def close_hands():
    hands.close()
    print("MediaPipe Hands resources closed.")

