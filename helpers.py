# helpers.py

import os
import csv
import numpy as np
import pandas as pd
import cv2
from scipy.interpolate import interp1d
from PyQt5.QtGui import QImage

# Import stałych
from constants import (STATIC_NUM_LANDMARKS, STATIC_NUM_FEATURES,
                       PATH_LENGTH, PATH_NUM_FEATURES)

# Funkcje pomocnicze rejestrowania/rozpoznawania gestów statycznych

# Procesowanie punktów charakterystycznych dla gestów statycznych
def process_static_landmarks(hand_landmarks):
    if not hand_landmarks: return None
    lm_list = []
    # Użycie nadgarstka jako referencji
    base_x, base_y, base_z = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z
    for lm in hand_landmarks.landmark:
        lm_list.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
    # Zwraca współrzędne punktów charakterystycznych tylko jeśli zgadza się ich liczba
    return lm_list if len(lm_list) == STATIC_NUM_FEATURES else None

# Zapisywanie danych gestów statycznych do CSV
def save_static_to_csv(filepath, label, landmarks):
    file_exists = os.path.isfile(filepath)
    # Zdefiniowanie nagłówków kolumn na podstawie stałych
    header = ['label'] + [f'{ax}{i}' for i in range(STATIC_NUM_LANDMARKS) for ax in 'xyz']
    try:
        with open(filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Dopisz nagłówki kolumn jeśli plik jest nowo utworzony albo pusty
            if not file_exists or os.path.getsize(filepath) == 0:
                writer.writerow(header)
            # Wpisz wiersz danych jeśli zgadza się ilość danych i nagłówków
            if landmarks and len(landmarks) == STATIC_NUM_FEATURES:
                writer.writerow([label] + landmarks)
                return True
            else:
                print(f"Warning: Attempted to save invalid static landmarks for label {label}.")
                return False
    except IOError as e:
        print(f"Error saving static CSV: {e}")
        return False

# Funkcje pomocnicze rejestrowania/klasifikacji ścieżki

# Standaryzacja ilości współrzędnych ścieżki za pomocą interpolacji
def standardize_path(points, length=PATH_LENGTH):
    if len(points) < 2: return None # Wymagane min. 2 punkty

    points_np = np.array(points)
    # Obliczanie kumulatywnego dystansu
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points_np, axis=0)**2, axis=1)))
    distance = np.insert(distance, 0, 0) # Dodawanie startowego dystansu

    # Obsługa przypadu ścieżce o długości 0
    if distance[-1] == 0:
        return np.array([points_np[0]] * length).flatten()

    # Stworzenie równo odległych puntków wzdłuż całej długości ścieżki
    alpha = np.linspace(0, distance[-1], length)

    # Interpolacja współrzędnych x i y
    interp_x = interp1d(distance, points_np[:, 0], kind='linear', bounds_error=False, fill_value="extrapolate")
    interp_y = interp1d(distance, points_np[:, 1], kind='linear', bounds_error=False, fill_value="extrapolate")

    # Połączenie i spłaszczenie interpolowanych współrzędnych
    standardized = np.vstack((interp_x(alpha), interp_y(alpha))).T
    return standardized.flatten() # Kształt: PATH_NUM_FEATURES

# Zapis zestandaryzowanej ścieżki do CSV
def save_path_to_csv(filepath, label, path_coords):
    file_exists = os.path.isfile(filepath)
    # Zdefiniowanie nagłówków na podstawie stałych
    header = ['label'] + [f'{ax}{i}' for i in range(PATH_LENGTH) for ax in 'xy']
    try:
        with open(filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Dopisz nagłówki kolumn jeśli plik jest nowo utworzony albo pusty
            if not file_exists or os.path.getsize(filepath) == 0:
                writer.writerow(header)
            # Wpisz wiersz danych jeśli zgadza się ilość danych i nagłówków
            if path_coords is not None and len(path_coords) == PATH_NUM_FEATURES:
                writer.writerow([label] + path_coords.tolist()) # Konwersja tablicy numpy
                return True
            else:
                print(f"Warning: Attempted to save invalid path data for label {label}. Length: {len(path_coords) if path_coords is not None else 'None'} != {PATH_NUM_FEATURES}")
                return False
    except IOError as e:
        print(f"Error saving path CSV: {e}")
        return False

# Funkcje pomocnicze konwersji obrazu

# Konwersja obrazu OpenCV (BGR) na QImage (RGB)
def convert_cv_qt(cv_img):
    # Konwersja BGR na RGB
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    # Stwórz obraz RGB z tablicy numpy
    convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return convert_to_Qt_format
