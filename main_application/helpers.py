# helpers.py

import os
import csv
import numpy as np
import pandas as pd
import cv2
from scipy.interpolate import interp1d
from PyQt5.QtGui import QImage

from constants import (STATIC_NUM_LANDMARKS, STATIC_NUM_FEATURES,
                       PATH_LENGTH, PATH_NUM_FEATURES)


def process_static_landmarks(hand_landmarks):
    if not hand_landmarks: return None
    lm_list = []
    base_x, base_y, base_z = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z
    for lm in hand_landmarks.landmark:
        lm_list.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
    return lm_list if len(lm_list) == STATIC_NUM_FEATURES else None


def save_static_to_csv(filepath, label, landmarks):
    output_dir = os.path.dirname(filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    file_exists = os.path.isfile(filepath)
    header = ['label'] + [f'{ax}{i}' for i in range(STATIC_NUM_LANDMARKS) for ax in 'xyz']
    try:
        with open(filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists or os.path.getsize(filepath) == 0:
                writer.writerow(header)
            if landmarks and len(landmarks) == STATIC_NUM_FEATURES:
                writer.writerow([label] + landmarks)
                return True
            else:
                print(f"Warning: Attempted to save invalid static landmarks for label {label}.")
                return False
    except IOError as e:
        print(f"Error saving static CSV: {e}")
        return False


def standardize_path(points, length=PATH_LENGTH):
    if len(points) < 2: return None
    points_np = np.array(points)

    distance = np.cumsum(np.sqrt(np.sum(np.diff(points_np, axis=0)**2, axis=1)))
    distance = np.insert(distance, 0, 0)

    if distance[-1] == 0:
        return np.array([points_np[0]] * length).flatten()

    alpha = np.linspace(0, distance[-1], length)
    interp_x = interp1d(distance, points_np[:, 0], kind='linear', bounds_error=False, fill_value="extrapolate")
    interp_y = interp1d(distance, points_np[:, 1], kind='linear', bounds_error=False, fill_value="extrapolate")
    standardized = np.vstack((interp_x(alpha), interp_y(alpha))).T
    return standardized.flatten()


def save_path_to_csv(filepath, label, path_coords):
    output_dir = os.path.dirname(filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    file_exists = os.path.isfile(filepath)
    header = ['label'] + [f'{ax}{i}' for i in range(PATH_LENGTH) for ax in 'xy']
    try:
        with open(filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists or os.path.getsize(filepath) == 0:
                writer.writerow(header)
            if path_coords is not None and len(path_coords) == PATH_NUM_FEATURES:
                writer.writerow([label] + path_coords.tolist())
                return True
            else:
                print(f"Warning: Attempted to save invalid path data for label {label}. Length: {len(path_coords) if path_coords is not None else 'None'} != {PATH_NUM_FEATURES}")
                return False
    except IOError as e:
        print(f"Error saving path CSV: {e}")
        return False


def convert_cv_qt(cv_img):
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return convert_to_Qt_format
