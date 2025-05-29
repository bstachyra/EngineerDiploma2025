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
                       PATH_ALLOWED_LABELS, PATH_NUM_CLASSES, PATH_NUM_FEATURES)


def _save_learning_curves(history, base_filename_prefix, progress_signal):
    try:
        if not history or not history.history:
            progress_signal.emit(f"Warning: No history data to plot learning curves for {base_filename_prefix}.")
            return None, None

        dpi = 100
        figsize = (1920 / dpi, 1080 / dpi)

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
            progress_signal.emit(f"Learning accuracy curve saved to {acc_plot_filename}")
        else:
            acc_plot_filename = None
            progress_signal.emit(f"Accuracy keys not found in history for {base_filename_prefix}.")

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
            progress_signal.emit(f"Learning loss curve saved to {loss_plot_filename}")
        else:
            loss_plot_filename = None
            progress_signal.emit(f"Loss keys not found in history for {base_filename_prefix}.")
        
        return acc_plot_filename, loss_plot_filename

    except Exception as e:
        progress_signal.emit(f"Error saving learning curves for {base_filename_prefix}: {e}")
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
            self.progress_signal.emit("Starting static model training...")
            # ... (rest of the loading and preprocessing code remains the same) ...
            self.progress_signal.emit(f"Loading static data from {self.csv_path}...")
            if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
                raise FileNotFoundError(f"Static CSV '{self.csv_path}' not found or empty.")
            df = pd.read_csv(self.csv_path)
            df.dropna(inplace=True) 
            if df.empty:
                raise ValueError("Static CSV is empty after removing rows with missing data.")
            self.progress_signal.emit(f"Static data loaded: {df.shape[0]} samples.")

            self.progress_signal.emit("Preprocessing static data...")
            X = df.iloc[:, 1:].values
            y_labels = df.iloc[:, 0].values

            label_encoder = LabelEncoder()
            label_encoder.fit(STATIC_ALLOWED_LABELS)
            try:
                y_encoded = label_encoder.transform(y_labels)
            except ValueError as e:
                unknown = set(y_labels) - set(label_encoder.classes_)
                raise ValueError(f"Unknown static label(s) in CSV: {unknown}. Error: {e}")

            y_categorical = to_categorical(y_encoded, num_classes=STATIC_NUM_CLASSES)

            with open(self.encoder_path, 'wb') as f:
                pickle.dump(label_encoder, f)
            self.progress_signal.emit(f"Static label encoder saved to {self.encoder_path}")

            if len(df) < 10 or df['label'].nunique() < 2:
                 raise ValueError("Need at least 10 samples and 2 different classes for static training.")

            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
                )
            except ValueError: 
                 self.progress_signal.emit("Warning: Stratification failed for static data split.")
                 X_train, X_val, y_train, y_val = train_test_split(
                    X, y_categorical, test_size=0.2, random_state=42
                )
            self.progress_signal.emit(f"Static data split: Train={X_train.shape[0]}, Val={X_val.shape[0]}")

            self.progress_signal.emit("Building static Keras model...")
            model = Sequential([
                Dense(128, activation='relu', input_shape=(STATIC_NUM_FEATURES,)), Dropout(0.3),
                Dense(64, activation='relu'), Dropout(0.3),
                Dense(STATIC_NUM_CLASSES, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.summary(print_fn=lambda x: self.progress_signal.emit(x))

            self.progress_signal.emit("Starting static model training...")
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
            ]
            history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                                validation_data=(X_val, y_val), callbacks=callbacks, verbose=0)

            val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
            self.progress_signal.emit(f"Static training finished. Val Accuracy: {val_accuracy:.4f}")

            acc_plot_file, loss_plot_file = _save_learning_curves(history, "static", self.progress_signal)
            
            learning_curves_msg = []
            if acc_plot_file: learning_curves_msg.append(f"Accuracy plot: {os.path.basename(acc_plot_file)}")
            if loss_plot_file: learning_curves_msg.append(f"Loss plot: {os.path.basename(loss_plot_file)}")
            
            final_message = f"Static training complete. Model saved. Val Acc: {val_accuracy:.4f}."
            if learning_curves_msg:
                final_message += " Learning curves saved: " + ", ".join(learning_curves_msg)
            
            self.progress_signal.emit(f"Saving static model to {self.model_path}...")
            model.save(self.model_path)
            self.finished_signal.emit(True, final_message)

        except (FileNotFoundError, ValueError, AssertionError) as e:
            self.finished_signal.emit(False, f"Static Training Error: {e}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished_signal.emit(False, f"Unexpected static training error: {e}")


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
            self.progress_signal.emit("Starting path model training...")
            # ... (rest of the loading and preprocessing code remains the same) ...
            self.progress_signal.emit(f"Loading path data from {self.csv_path}...")
            if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
                raise FileNotFoundError(f"Path CSV '{self.csv_path}' not found or empty.")
            df = pd.read_csv(self.csv_path)
            df.dropna(inplace=True) 
            if df.empty:
                raise ValueError("Path CSV is empty after removing rows with missing data.")
            self.progress_signal.emit(f"Path data loaded: {df.shape[0]} samples.")

            self.progress_signal.emit("Preprocessing path data...")
            X = df.iloc[:, 1:].values
            y_labels = df.iloc[:, 0].values

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            self.progress_signal.emit(f"Path feature scaler saved to {self.scaler_path}")

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

            if len(df) < 10 or df['label'].nunique() < 2:
                 raise ValueError("Need at least 10 samples and 2 different classes for path training.")

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

            acc_plot_file, loss_plot_file = _save_learning_curves(history, "path", self.progress_signal)

            learning_curves_msg = []
            if acc_plot_file: learning_curves_msg.append(f"Accuracy plot: {os.path.basename(acc_plot_file)}")
            if loss_plot_file: learning_curves_msg.append(f"Loss plot: {os.path.basename(loss_plot_file)}")

            final_message = f"Path training complete. Model saved. Val Acc: {val_accuracy:.4f}."
            if learning_curves_msg:
                final_message += " Learning curves saved: " + ", ".join(learning_curves_msg)

            self.progress_signal.emit(f"Saving path model to {self.model_path}...")
            model.save(self.model_path)
            self.finished_signal.emit(True, final_message)

        except (FileNotFoundError, ValueError, AssertionError) as e:
            self.finished_signal.emit(False, f"Path Training Error: {e}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished_signal.emit(False, f"Unexpected path training error: {e}")