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

import matplotlib.pyplot as plt

from constants import (STATIC_ALLOWED_LABELS, STATIC_NUM_CLASSES, STATIC_NUM_FEATURES,
                       PATH_ALLOWED_LABELS, PATH_NUM_CLASSES, PATH_NUM_FEATURES, DPI, FIGSIZE_HD)


def _save_learning_curves(history, base_filename_prefix, progress_signal):
    try:
        if not history or not history.history:
            progress_signal.emit(f"Ostrzeżenie: Brak danych historii do wykreślenia krzywych uczenia dla {base_filename_prefix}.")
            return None, None

        dpi = DPI
        figsize = FIGSIZE_HD

        acc_plot_filename = f"{base_filename_prefix}_learning_accuracy.png"
        plt.figure(figsize=figsize, dpi=dpi)
        if 'accuracy' in history.history and 'val_accuracy' in history.history:
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title(f'{base_filename_prefix.capitalize()} Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(acc_plot_filename)
            plt.close()
            progress_signal.emit(f"Krzywa uczenia dokładności zapisana do {acc_plot_filename}")
        else:
            acc_plot_filename = None
            progress_signal.emit(f"Klucze 'accuracy' nie znalezione w historii dla {base_filename_prefix}.")

        loss_plot_filename = f"{base_filename_prefix}_learning_loss.png"
        plt.figure(figsize=figsize, dpi=dpi)
        if 'loss' in history.history and 'val_loss' in history.history:
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title(f'{base_filename_prefix.capitalize()} Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(loss_plot_filename)
            plt.close()
            progress_signal.emit(f"Krzywa uczenia funkcji straty zapisana do {loss_plot_filename}")
        else:
            loss_plot_filename = None
            progress_signal.emit(f"Klucze 'loss' nie znalezione w historii dla {base_filename_prefix}.")
        
        return acc_plot_filename, loss_plot_filename

    except Exception as e:
        progress_signal.emit(f"Błąd podczas zapisywania krzywych uczenia dla {base_filename_prefix}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


class StaticTrainingThread(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)


    def __init__(self, csv_path, model_path, encoder_path, parent=None):
        super().__init__(parent)
        self.csv_path = csv_path
        self.model_path = model_path
        self.encoder_path = encoder_path


    def run(self):
        try:
            self.progress_signal.emit("Rozpoczynanie trenowania modelu statycznego...")
            self.progress_signal.emit(f"Ładowanie danych statycznych z {self.csv_path}...")
            if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
                raise FileNotFoundError(f"Plik CSV statyczny '{self.csv_path}' nie znaleziony lub pusty.")
            df = pd.read_csv(self.csv_path)
            df.dropna(inplace=True) 
            if df.empty:
                raise ValueError("Plik CSV statyczny jest pusty po usunięciu wierszy z brakującymi danymi.")
            self.progress_signal.emit(f"Dane statyczne załadowane: {df.shape[0]} próbek.")

            self.progress_signal.emit("Przetwarzanie wstępne danych statycznych...")
            X = df.iloc[:, 1:].values
            y_labels = df.iloc[:, 0].values

            label_encoder = LabelEncoder()
            label_encoder.fit(STATIC_ALLOWED_LABELS)
            try:
                y_encoded = label_encoder.transform(y_labels)
            except ValueError as e:
                unknown = set(y_labels) - set(label_encoder.classes_)
                raise ValueError(f"Nieznana etykieta(y) statyczna(e) w CSV: {unknown}. Błąd: {e}")

            y_categorical = to_categorical(y_encoded, num_classes=STATIC_NUM_CLASSES)

            encoder_dir = os.path.dirname(self.encoder_path)
            if encoder_dir:
                os.makedirs(encoder_dir, exist_ok=True)
            with open(self.encoder_path, 'wb') as f:
                pickle.dump(label_encoder, f)
            self.progress_signal.emit(f"Enkoder etykiet statycznych zapisany do {self.encoder_path}")

            if len(df) < 10 or df['label'].nunique() < 2:
                 raise ValueError("Wymagane co najmniej 10 próbek i 2 różne klasy do trenowania modelu statycznego.")

            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
                )
            except ValueError: 
                 self.progress_signal.emit("Ostrzeżenie: Stratyfikacja nie powiodła się dla podziału danych statycznych.")
                 X_train, X_val, y_train, y_val = train_test_split(
                    X, y_categorical, test_size=0.2, random_state=42
                )
            self.progress_signal.emit(f"Podział danych statycznych: Treningowe={X_train.shape[0]}, Walidacyjne={X_val.shape[0]}")

            self.progress_signal.emit("Budowanie modelu Keras statycznego...")
            model = Sequential([
                Dense(128, activation='relu', input_shape=(STATIC_NUM_FEATURES,)), Dropout(0.3),
                Dense(64, activation='relu'), Dropout(0.3),
                Dense(STATIC_NUM_CLASSES, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.summary(print_fn=lambda x: self.progress_signal.emit(x))

            self.progress_signal.emit("Rozpoczynanie trenowania modelu statycznego...")
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
            ]
            history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                                validation_data=(X_val, y_val), callbacks=callbacks, verbose=0)

            val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
            self.progress_signal.emit(f"Trenowanie modelu statycznego zakończone. Dokładność walidacyjna: {val_accuracy:.4f}")

            acc_plot_file, loss_plot_file = _save_learning_curves(history, "static", self.progress_signal)
            
            learning_curves_msg = []
            if acc_plot_file: learning_curves_msg.append(f"Wykres dokładności: {os.path.basename(acc_plot_file)}")
            if loss_plot_file: learning_curves_msg.append(f"Wykres funkcji straty: {os.path.basename(loss_plot_file)}")
            
            final_message = f"Trenowanie modelu statycznego zakończone. Model zapisany. Dokładność walidacyjna: {val_accuracy:.4f}."
            if learning_curves_msg:
                final_message += " Krzywe uczenia zapisane: " + ", ".join(learning_curves_msg)
            
            # Ensure model directory exists
            model_dir = os.path.dirname(self.model_path)
            if model_dir:
                os.makedirs(model_dir, exist_ok=True)
            self.progress_signal.emit(f"Zapisywanie modelu statycznego do {self.model_path}...")
            model.save(self.model_path)
            self.finished_signal.emit(True, final_message)

        except (FileNotFoundError, ValueError, AssertionError) as e:
            self.finished_signal.emit(False, f"Błąd trenowania modelu statycznego: {e}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished_signal.emit(False, f"Nieoczekiwany błąd trenowania modelu statycznego: {e}")


class PathTrainingThread(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)


    def __init__(self, csv_path, model_path, encoder_path, scaler_path, parent=None):
        super().__init__(parent)
        self.csv_path = csv_path
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.scaler_path = scaler_path


    def run(self):
        try:
            self.progress_signal.emit("Rozpoczynanie trenowania modelu ścieżek...")
            self.progress_signal.emit(f"Ładowanie danych ścieżek z {self.csv_path}...")
            if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
                raise FileNotFoundError(f"Plik CSV ścieżek '{self.csv_path}' nie znaleziony lub pusty.")
            df = pd.read_csv(self.csv_path)
            df.dropna(inplace=True) 
            if df.empty:
                raise ValueError("Plik CSV ścieżek jest pusty po usunięciu wierszy z brakującymi danymi.")
            self.progress_signal.emit(f"Dane ścieżek załadowane: {df.shape[0]} próbek.")

            self.progress_signal.emit("Przetwarzanie wstępne danych ścieżek...")
            X = df.iloc[:, 1:].values
            y_labels = df.iloc[:, 0].values

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            # Ensure scaler directory exists
            scaler_dir = os.path.dirname(self.scaler_path)
            if scaler_dir:
                os.makedirs(scaler_dir, exist_ok=True)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            self.progress_signal.emit(f"Skaler cech ścieżek zapisany do {self.scaler_path}")

            label_encoder = LabelEncoder()
            label_encoder.fit(PATH_ALLOWED_LABELS)
            try:
                y_encoded = label_encoder.transform(y_labels)
            except ValueError as e:
                unknown = set(y_labels) - set(label_encoder.classes_)
                raise ValueError(f"Nieznana etykieta(y) ścieżek w CSV: {unknown}. Błąd: {e}")

            y_categorical = to_categorical(y_encoded, num_classes=PATH_NUM_CLASSES)

            encoder_dir = os.path.dirname(self.encoder_path)
            if encoder_dir:
                os.makedirs(encoder_dir, exist_ok=True)
            with open(self.encoder_path, 'wb') as f:
                pickle.dump(label_encoder, f)
            self.progress_signal.emit(f"Enkoder etykiet ścieżek zapisany do {self.encoder_path}")

            if len(df) < 10 or df['label'].nunique() < 2:
                 raise ValueError("Wymagane co najmniej 10 próbek i 2 różne klasy do trenowania modelu ścieżek.")

            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_scaled, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
                )
            except ValueError:
                 self.progress_signal.emit("Ostrzeżenie: Stratyfikacja nie powiodła się dla podziału danych ścieżek.")
                 X_train, X_val, y_train, y_val = train_test_split(
                    X_scaled, y_categorical, test_size=0.2, random_state=42
                )
            self.progress_signal.emit(f"Podział danych ścieżek: Treningowe={X_train.shape[0]}, Walidacyjne={X_val.shape[0]}")

            self.progress_signal.emit("Budowanie modelu Keras ścieżek (MLP)...")
            model = Sequential([
                Dense(256, activation='relu', input_shape=(PATH_NUM_FEATURES,)), Dropout(0.4),
                Dense(128, activation='relu'), Dropout(0.4),
                Dense(64, activation='relu'), Dropout(0.4),
                Dense(PATH_NUM_CLASSES, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.summary(print_fn=lambda x: self.progress_signal.emit(x))

            self.progress_signal.emit("Rozpoczynanie trenowania modelu ścieżek...")
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=0.00001)
            ]
            history = model.fit(X_train, y_train, epochs=150, batch_size=32,
                                validation_data=(X_val, y_val), callbacks=callbacks, verbose=0)

            val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
            self.progress_signal.emit(f"Trenowanie modelu ścieżek zakończone. Dokładność walidacyjna: {val_accuracy:.4f}")

            acc_plot_file, loss_plot_file = _save_learning_curves(history, "path", self.progress_signal)

            learning_curves_msg = []
            if acc_plot_file: learning_curves_msg.append(f"Wykres dokładności: {os.path.basename(acc_plot_file)}")
            if loss_plot_file: learning_curves_msg.append(f"Wykres funkcji straty: {os.path.basename(loss_plot_file)}")

            final_message = f"Trenowanie modelu ścieżek zakończone. Model zapisany. Dokładność walidacyjna: {val_accuracy:.4f}."
            if learning_curves_msg:
                final_message += " Krzywe uczenia zapisane: " + ", ".join(learning_curves_msg)
            
            model_dir = os.path.dirname(self.model_path)
            if model_dir:
                os.makedirs(model_dir, exist_ok=True)
            self.progress_signal.emit(f"Zapisywanie modelu ścieżek do {self.model_path}...")
            model.save(self.model_path)
            self.finished_signal.emit(True, final_message)

        except (FileNotFoundError, ValueError, AssertionError) as e:
            self.finished_signal.emit(False, f"Błąd trenowania modelu ścieżek: {e}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished_signal.emit(False, f"Nieoczekiwany błąd trenowania modelu ścieżek: {e}")