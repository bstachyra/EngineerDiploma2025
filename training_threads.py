# training_threads.py

import os
import pickle
import numpy as np
import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Import stałych
from constants import (STATIC_ALLOWED_LABELS, STATIC_NUM_CLASSES, STATIC_NUM_FEATURES,
                       PATH_ALLOWED_LABELS, PATH_NUM_CLASSES, PATH_NUM_FEATURES)

# Obsługa trenowania modelu gestów statycznych
class StaticTrainingThread(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str) # success, message

    def __init__(self, csv_path, model_path, encoder_path, parent=None):
        super().__init__(parent)
        self.csv_path = csv_path
        self.model_path = model_path
        self.encoder_path = encoder_path

    # Ładowanie danych, trenowanie modelu, zapis wyników
    def run(self):
        try:
            self.progress_signal.emit("Starting static model training...")
            # Ładowanie danych
            self.progress_signal.emit(f"Loading static data from {self.csv_path}...")
            if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
                raise FileNotFoundError(f"Static CSV '{self.csv_path}' not found or empty.")
            df = pd.read_csv(self.csv_path)
            df.dropna(inplace=True) # Usuń wiersze z brakującymi danymi punktów charakterystycznych
            if df.empty:
                raise ValueError("Static CSV is empty after removing rows with missing data.")
            self.progress_signal.emit(f"Static data loaded: {df.shape[0]} samples.")

            # Preprocessing
            self.progress_signal.emit("Preprocessing static data...")
            X = df.iloc[:, 1:].values 
            y_labels = df.iloc[:, 0].values 

            # Enkodowanie etykiet
            label_encoder = LabelEncoder()
            label_encoder.fit(STATIC_ALLOWED_LABELS)
            try:
                y_encoded = label_encoder.transform(y_labels)
            except ValueError as e:
                unknown = set(y_labels) - set(label_encoder.classes_)
                raise ValueError(f"Unknown static label(s) in CSV: {unknown}. Error: {e}")

            y_categorical = to_categorical(y_encoded, num_classes=STATIC_NUM_CLASSES)

            # Zapis enkodera
            with open(self.encoder_path, 'wb') as f:
                pickle.dump(label_encoder, f)
            self.progress_signal.emit(f"Static label encoder saved to {self.encoder_path}")

            # Sprawdzenie czy ilość danych wystarczająca
            if len(df) < 10 or df['label'].nunique() < 2:
                 raise ValueError("Need at least 10 samples and 2 different classes for static training.")

            # Podziel dane
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
                )
            except ValueError: # Fallback if stratification fails (e.g., only 1 sample per class)
                 self.progress_signal.emit("Warning: Stratification failed for static data split.")
                 X_train, X_val, y_train, y_val = train_test_split(
                    X, y_categorical, test_size=0.2, random_state=42
                )
            self.progress_signal.emit(f"Static data split: Train={X_train.shape[0]}, Val={X_val.shape[0]}")

            # Zdefiniuj i trenuj model
            self.progress_signal.emit("Building static Keras model...")
            model = Sequential([
                Dense(128, activation='relu', input_shape=(STATIC_NUM_FEATURES,)), Dropout(0.3),
                Dense(64, activation='relu'), Dropout(0.3),
                Dense(STATIC_NUM_CLASSES, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.summary(print_fn=lambda x: self.progress_signal.emit(x)) # Log summary

            self.progress_signal.emit("Starting static model training...")
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
            ]
            history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                                validation_data=(X_val, y_val), callbacks=callbacks, verbose=0)

            val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
            self.progress_signal.emit(f"Static training finished. Val Accuracy: {val_accuracy:.4f}")

            # Zapisz model
            self.progress_signal.emit(f"Saving static model to {self.model_path}...")
            model.save(self.model_path)
            self.finished_signal.emit(True, f"Static training complete. Model saved. Val Acc: {val_accuracy:.4f}")

        except (FileNotFoundError, ValueError, AssertionError) as e:
            self.finished_signal.emit(False, f"Static Training Error: {e}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished_signal.emit(False, f"Unexpected static training error: {e}")

# Obsługa trenowania modelu klasyfikacji ścieżki
class PathTrainingThread(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, csv_path, model_path, encoder_path, scaler_path, parent=None):
        super().__init__(parent)
        self.csv_path = csv_path
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.scaler_path = scaler_path

    # Ładowanie danych, trenowanie modelu, zapisanie wyników
    def run(self):
        try:
            self.progress_signal.emit("Starting path model training...")
            # Ładowanie danych
            self.progress_signal.emit(f"Loading path data from {self.csv_path}...")
            if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
                raise FileNotFoundError(f"Path CSV '{self.csv_path}' not found or empty.")
            df = pd.read_csv(self.csv_path)
            df.dropna(inplace=True) # Usunięcie wierszy z brakującymi danymi współrzędnych
            if df.empty:
                raise ValueError("Path CSV is empty after removing rows with missing data.")
            self.progress_signal.emit(f"Path data loaded: {df.shape[0]} samples.")

            # Preprocessing
            self.progress_signal.emit("Preprocessing path data...")
            X = df.iloc[:, 1:].values 
            y_labels = df.iloc[:, 0].values 

            # Skalowanie koordynatów
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            self.progress_signal.emit(f"Path feature scaler saved to {self.scaler_path}")

            # Enkodowanie etykiet
            label_encoder = LabelEncoder()
            label_encoder.fit(PATH_ALLOWED_LABELS)
            try:
                y_encoded = label_encoder.transform(y_labels)
            except ValueError as e:
                unknown = set(y_labels) - set(label_encoder.classes_)
                raise ValueError(f"Unknown path label(s) in CSV: {unknown}. Error: {e}")

            y_categorical = to_categorical(y_encoded, num_classes=PATH_NUM_CLASSES)
            with open(self.encoder_path, 'wb') as f:
                pickle.dump(label_encoder, f)
            self.progress_signal.emit(f"Path label encoder saved to {self.encoder_path}")

            # Sprawdzenie czy wystarczająco danych
            if len(df) < 10 or df['label'].nunique() < 2:
                 raise ValueError("Need at least 10 samples and 2 different classes for path training.")

            # Podzielenie danych
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_scaled, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
                )
            except ValueError:
                 self.progress_signal.emit("Warning: Stratification failed for path data split.")
                 X_train, X_val, y_train, y_val = train_test_split(
                    X_scaled, y_categorical, test_size=0.2, random_state=42
                )
            self.progress_signal.emit(f"Path data split: Train={X_train.shape[0]}, Val={X_val.shape[0]}")

            # Zdefiniowanie i trenowanie modelu
            self.progress_signal.emit("Building path Keras model (MLP)...")
            model = Sequential([
                Dense(256, activation='relu', input_shape=(PATH_NUM_FEATURES,)), Dropout(0.4),
                Dense(128, activation='relu'), Dropout(0.4),
                Dense(64, activation='relu'), Dropout(0.4),
                Dense(PATH_NUM_CLASSES, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.summary(print_fn=lambda x: self.progress_signal.emit(x))

            self.progress_signal.emit("Starting path model training...")
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=0.00001)
            ]
            history = model.fit(X_train, y_train, epochs=150, batch_size=32,
                                validation_data=(X_val, y_val), callbacks=callbacks, verbose=0)

            val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
            self.progress_signal.emit(f"Path training finished. Val Accuracy: {val_accuracy:.4f}")

            # Zapisanie modelu
            self.progress_signal.emit(f"Saving path model to {self.model_path}...")
            model.save(self.model_path)
            self.finished_signal.emit(True, f"Path training complete. Model saved. Val Acc: {val_accuracy:.4f}")

        except (FileNotFoundError, ValueError, AssertionError) as e:
            self.finished_signal.emit(False, f"Path Training Error: {e}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished_signal.emit(False, f"Unexpected path training error: {e}")

