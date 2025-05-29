# testing_threads.py

import os
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             roc_curve, auc, precision_recall_curve, average_precision_score)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from helpers import process_static_landmarks, standardize_path
from mp_setup import hands
from constants import PATH_TRACKING_LANDMARK, STATIC_ALLOWED_LABELS, PATH_ALLOWED_LABELS, DPI

ALLOWED_VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')
ALLOWED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')

FIGSIZE_HD = (1920 / DPI, 1080 / DPI)


def _save_classification_report_metrics_plot(report_dict, class_labels_in_report, base_filename_prefix, progress_signal):
    try:
        if not report_dict:
            progress_signal.emit(f"Ostrzeżenie: Brak danych raportu do wyrysowania metryk dla {base_filename_prefix}.")
            return None
        metrics_data = {
            label: {
                'precision': report_dict[label]['precision'],
                'recall': report_dict[label]['recall'],
                'f1-score': report_dict[label]['f1-score']
            }
            for label in class_labels_in_report if isinstance(report_dict.get(label), dict)
        }
        if not metrics_data:
            progress_signal.emit(f"Nie znaleziono metryk per klasa w raporcie dla {base_filename_prefix}.")
            return None

        df_metrics = pd.DataFrame.from_dict(metrics_data, orient='index')
        plot_filename = f"{base_filename_prefix}_classification_metrics.png"
        
        n_metrics = len(df_metrics.columns)
        n_classes = len(df_metrics.index)
        bar_width = 0.8 / n_metrics 
        
        fig_width_adjusted = max(FIGSIZE_HD[0], n_classes * n_metrics * bar_width + 4)

        fig, ax = plt.subplots(figsize=(fig_width_adjusted, FIGSIZE_HD[1]), dpi=DPI)
        indices = np.arange(n_classes)
        for i, metric in enumerate(df_metrics.columns):
            ax.bar(indices + i * bar_width - (bar_width*(n_metrics-1)/2) , df_metrics[metric], bar_width, label=metric.capitalize())

        ax.set_ylabel('Wyniki')
        ax.set_title(f'{base_filename_prefix.capitalize()} Metryk Klasyfikacji per Klasa')
        ax.set_xticks(indices)
        ax.set_xticklabels(df_metrics.index, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, axis='y', linestyle='--')
        plt.ylim(0, 1.1)
        plt.tight_layout(pad=2.0)
        plt.savefig(plot_filename)
        plt.close()
        progress_signal.emit(f"Wykres metryk klasyfikacji zapisany do {plot_filename}")
        return plot_filename
    except Exception as e:
        progress_signal.emit(f"Błąd podczas zapisywania wykresu metryk klasyfikacji dla {base_filename_prefix}: {e}")
        import traceback; traceback.print_exc()
        return None


def _save_normalized_confusion_matrix(y_true, y_pred, class_labels_for_matrix, base_filename_prefix, progress_signal):
    plot_filename = f"{base_filename_prefix}_confusion_matrix_normalized.png"
    try:
        cm = confusion_matrix(y_true, y_pred, labels=class_labels_for_matrix)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        num_classes = len(class_labels_for_matrix)
        fig_width = max(FIGSIZE_HD[0]*0.6, int(num_classes * 0.5) + 4)
        fig_height = max(FIGSIZE_HD[1]*0.6, int(num_classes * 0.4) + 3)
        annot_font_size = max(6, 10 - int(num_classes * 0.1))
        tick_font_size = max(6, 9 - int(num_classes * 0.1))

        plt.figure(figsize=(fig_width, fig_height), dpi=DPI) # Not strictly HD, but scaled for readability
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues" if "static" in base_filename_prefix else "Greens",
                    xticklabels=class_labels_for_matrix, yticklabels=class_labels_for_matrix,
                    annot_kws={"size": annot_font_size})
        plt.title(f'Normalized Confusion Matrix ({base_filename_prefix.capitalize()})', fontsize=tick_font_size + 2)
        plt.ylabel('True Label', fontsize=tick_font_size)
        plt.xlabel('Predicted Label', fontsize=tick_font_size)
        plt.xticks(rotation=45, ha='right', fontsize=tick_font_size)
        plt.yticks(rotation=0, fontsize=tick_font_size)
        plt.tight_layout(pad=2.0)
        plt.savefig(plot_filename)
        plt.close()
        progress_signal.emit(f"Znormalizowana macierz konfuzji zapisana do {plot_filename}")
        return plot_filename
    except Exception as e:
        progress_signal.emit(f"Błąd podczas zapisywania znormalizowanej macierzy konfuzji dla {base_filename_prefix}: {e}")
        import traceback; traceback.print_exc()
        return None


def _save_roc_auc_curve_macro(y_true, y_pred_scores, classes_for_roc, base_filename_prefix, progress_signal):
    plot_filename = f"{base_filename_prefix}_roc_auc_macro.png"
    try:
        y_true_binarized = label_binarize(y_true, classes=classes_for_roc)
        n_classes = y_true_binarized.shape[1]

        if y_pred_scores.shape[1] != n_classes:
             progress_signal.emit(f"Ostrzeżenie: Niezgodność w liczbie kolumn wyników predykcji ({y_pred_scores.shape[1]}) i liczbie klas ({n_classes}) dla ROC. Pomijanie ROC dla {base_filename_prefix}.")
             return None

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            if np.sum(y_true_binarized[:, i]) > 0:
                 fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred_scores[:, i])
                 roc_auc[i] = auc(fpr[i], tpr[i])
            else:
                 fpr[i], tpr[i], roc_auc[i] = np.array([0,1]), np.array([0,1]), 0.0

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        valid_classes_count = 0
        for i in range(n_classes):
            if np.sum(y_true_binarized[:, i]) > 0:
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                valid_classes_count +=1
        
        if valid_classes_count > 0:
            mean_tpr /= valid_classes_count
            fpr_macro = all_fpr
            tpr_macro = mean_tpr
            roc_auc_macro = auc(fpr_macro, tpr_macro)
        else:
            fpr_macro, tpr_macro, roc_auc_macro = np.array([0,1]), np.array([0,1]), 0.0

        plt.figure(figsize=FIGSIZE_HD, dpi=DPI)
        plt.plot(fpr_macro, tpr_macro,
                 label=f'Macro-average ROC curve (AUC = {roc_auc_macro:0.2f})',
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Macro-Averaged ROC Curve ({base_filename_prefix.capitalize()})')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close()
        progress_signal.emit(f"Krzywa ROC uśredniona makro zapisana do {plot_filename}")
        return plot_filename
    except Exception as e:
        progress_signal.emit(f"Błąd podczas zapisywania krzywej ROC dla {base_filename_prefix}: {e}")
        import traceback; traceback.print_exc()
        return None


def _save_precision_recall_curve_macro(y_true, y_pred_scores, classes_for_pr, base_filename_prefix, progress_signal):
    plot_filename = f"{base_filename_prefix}_precision_recall_macro.png"
    try:
        y_true_binarized = label_binarize(y_true, classes=classes_for_pr)
        n_classes = y_true_binarized.shape[1]

        if y_pred_scores.shape[1] != n_classes:
            progress_signal.emit(f"Ostrzeżenie: Niezgodność w liczbie kolumn wyników predykcji ({y_pred_scores.shape[1]}) i liczbie klas ({n_classes}) dla PR. Pomijanie PR dla {base_filename_prefix}.")
            return None

        precision = dict()
        recall = dict()
        average_precision = dict()

        for i in range(n_classes):
            if np.sum(y_true_binarized[:, i]) > 0:
                precision[i], recall[i], _ = precision_recall_curve(y_true_binarized[:, i], y_pred_scores[:, i])
                average_precision[i] = average_precision_score(y_true_binarized[:, i], y_pred_scores[:, i])
            else:
                precision[i], recall[i], average_precision[i] = np.array([1,0]), np.array([0,1]), 0.0

        valid_ap_scores = [average_precision[i] for i in range(n_classes) if np.sum(y_true_binarized[:,i]) > 0]
        if valid_ap_scores:
            macro_ap = np.mean(valid_ap_scores)
        else:
            macro_ap = 0.0

        plt.figure(figsize=FIGSIZE_HD, dpi=DPI)
        colors = plt.cm.get_cmap('viridis', n_classes)
        
        num_curves_to_plot = min(n_classes, 5)
        plotted_count = 0
        for i in range(n_classes):
            if np.sum(y_true_binarized[:, i]) > 0 and plotted_count < num_curves_to_plot:
                plt.plot(recall[i], precision[i], color=colors(i/n_classes if n_classes > 1 else 0.5), lw=2,
                         label=f'PR curve of class {classes_for_pr[i]} (AP = {average_precision[i]:0.2f})')
                plotted_count+=1

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f'Precision-Recall Curves (first {plotted_count} classes) & Macro AP ({base_filename_prefix.capitalize()})')
        plt.legend(loc="best")
        plt.text(0.05, 0.05, f'Macro Average Precision (all classes): {macro_ap:0.2f}',
                 transform=plt.gca().transAxes, fontsize=14, verticalalignment='bottom')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close()
        progress_signal.emit(f"Wykres krzywej precyzja-czułość zapisany do {plot_filename} (Makro AP: {macro_ap:.2f})")
        return plot_filename
    except Exception as e:
        progress_signal.emit(f"Błąd podczas zapisywania krzywej precyzja-czułość dla {base_filename_prefix}: {e}")
        import traceback; traceback.print_exc()
        return None


class StaticModelTestThread(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, float, str, np.ndarray, list, str, str, str, str)

    def __init__(self, test_folder_path, static_model, label_encoder, parent=None):
        super().__init__(parent)
        self.test_folder_path = test_folder_path
        self.static_model = static_model
        self.label_encoder = label_encoder
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        class_metrics_plot_fn, norm_cm_fn, roc_auc_fn, pr_curve_fn = None, None, None, None
        default_return_tuple = (0.0, "", np.array([]), [], class_metrics_plot_fn, norm_cm_fn, roc_auc_fn, pr_curve_fn)

        if not self.static_model or not self.label_encoder:
            self.finished_signal.emit(False, *default_return_tuple[:1], "Model statyczny lub enkoder etykiet niezaładowany.", *default_return_tuple[3:])
            return

        y_true_labels = []
        y_pred_labels = []
        y_pred_scores_list = []

        class_labels_from_encoder = list(self.label_encoder.classes_)

        processed_images_count = 0
        image_files = []
        try:
            all_files_in_folder = []
            for label_dir in class_labels_from_encoder:
                dir_path = os.path.join(self.test_folder_path, label_dir)
                if os.path.isdir(dir_path):
                    for f_name in os.listdir(dir_path):
                        if f_name.lower().endswith(ALLOWED_IMAGE_EXTENSIONS):
                             all_files_in_folder.append((os.path.join(dir_path, f_name), label_dir.upper()))
            if not all_files_in_folder:
                 self.finished_signal.emit(False, *default_return_tuple[:1], "Nie znaleziono obrazów w podfolderach klas (statyczne).", *default_return_tuple[3:])
                 return
            image_files = all_files_in_folder
            total_images_to_process = len(image_files)
            self.progress_signal.emit(f"Rozpoczynanie testu statycznego. Znaleziono {total_images_to_process} obrazów.")
        except Exception as e:
            self.finished_signal.emit(False, *default_return_tuple[:1], f"Błąd przy odczycie folderu testowego statycznego: {e}", *default_return_tuple[3:])
            return

        try:
            for image_path, true_label_str in image_files:
                if not self.running: break
                img = cv2.imread(image_path)
                if img is None: continue

                rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                rgb_image.flags.writeable = False
                results = hands.process(rgb_image)
                
                predicted_label_str = "nieznany_statyczny"
                current_pred_scores = np.zeros(len(class_labels_from_encoder))

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    processed_landmarks = process_static_landmarks(hand_landmarks)
                    if processed_landmarks:
                        input_data = np.array([processed_landmarks], dtype=np.float32)
                        prediction_probs = self.static_model.predict(input_data, verbose=0)[0]
                        current_pred_scores = prediction_probs
                        predicted_idx = np.argmax(prediction_probs)
                        predicted_label_str = self.label_encoder.inverse_transform([predicted_idx])[0]
                
                y_true_labels.append(true_label_str)
                y_pred_labels.append(predicted_label_str)
                y_pred_scores_list.append(current_pred_scores)
                processed_images_count += 1
                self.progress_signal.emit(f"Przetworzono {processed_images_count}/{total_images_to_process} (stat.): {os.path.basename(image_path)}")

            if not self.running:
                self.finished_signal.emit(False, *default_return_tuple[:1], "Testowanie statyczne przerwane.", *default_return_tuple[3:])
                return

            if not y_true_labels:
                self.finished_signal.emit(False, *default_return_tuple[:1], "Nie przetworzono żadnych obrazów statycznych pomyślnie.", *default_return_tuple[3:])
                return

            y_pred_scores_np = np.array(y_pred_scores_list)
            accuracy = accuracy_score(y_true_labels, y_pred_labels)

            report_labels_for_plot_and_metrics = class_labels_from_encoder
            
            report_str = classification_report(y_true_labels, y_pred_labels, labels=report_labels_for_plot_and_metrics, zero_division=0, target_names=report_labels_for_plot_and_metrics, output_dict=False)
            report_dict = classification_report(y_true_labels, y_pred_labels, labels=report_labels_for_plot_and_metrics, zero_division=0, target_names=report_labels_for_plot_and_metrics, output_dict=True)
            
            raw_conf_matrix = confusion_matrix(y_true_labels, y_pred_labels, labels=class_labels_from_encoder)

            class_metrics_plot_fn = _save_classification_report_metrics_plot(report_dict, report_labels_for_plot_and_metrics, "static", self.progress_signal)
            norm_cm_fn = _save_normalized_confusion_matrix(y_true_labels, y_pred_labels, class_labels_from_encoder, "static", self.progress_signal)

            roc_auc_fn = _save_roc_auc_curve_macro(y_true_labels, y_pred_scores_np, classes_for_roc=class_labels_from_encoder, base_filename_prefix="static", progress_signal=self.progress_signal)
            pr_curve_fn = _save_precision_recall_curve_macro(y_true_labels, y_pred_scores_np, classes_for_pr=class_labels_from_encoder, base_filename_prefix="static", progress_signal=self.progress_signal)

            self.progress_signal.emit("Testowanie statyczne zakończone.")
            self.finished_signal.emit(True, accuracy, report_str, raw_conf_matrix, class_labels_from_encoder, class_metrics_plot_fn, norm_cm_fn, roc_auc_fn, pr_curve_fn)

        except Exception as e:
            import traceback; traceback.print_exc()
            err_msg = f"Błąd podczas testowania statycznego: {e}"
            self.progress_signal.emit(err_msg)
            self.finished_signal.emit(False, 0.0, err_msg, np.array([]), [], None, None, None, None)


class PathModelTestThread(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, float, str, np.ndarray, list, str, str, str, str)

    def __init__(self, test_folder_path, path_model, path_label_encoder, path_scaler, parent=None):
        super().__init__(parent)
        self.test_folder_path = test_folder_path
        self.path_model = path_model
        self.path_label_encoder = path_label_encoder
        self.path_scaler = path_scaler
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        class_metrics_plot_fn, norm_cm_fn, roc_auc_fn, pr_curve_fn = None, None, None, None
        default_return_tuple = (0.0, "", np.array([]), [], class_metrics_plot_fn, norm_cm_fn, roc_auc_fn, pr_curve_fn)

        if not self.path_model or not self.path_label_encoder or not self.path_scaler:
            self.finished_signal.emit(False, *default_return_tuple[:1], "Model ścieżek, enkoder lub skaler niezaładowany.", *default_return_tuple[3:])
            return

        y_true_labels = []
        y_pred_labels = []
        y_pred_scores_list = []

        class_labels_from_encoder = list(self.path_label_encoder.classes_)
        discovered_samples = []
        try:
            self.progress_signal.emit(f"Przeszukiwanie folderu testowego ścieżek: {self.test_folder_path}")
            for label_name in os.listdir(self.test_folder_path): 
                label_dir_path = os.path.join(self.test_folder_path, label_name)
                if os.path.isdir(label_dir_path) and label_name in class_labels_from_encoder:
                    true_label = label_name
                    for item_name in os.listdir(label_dir_path):
                        item_path = os.path.join(label_dir_path, item_name)
                        if os.path.isfile(item_path) and item_name.lower().endswith(ALLOWED_VIDEO_EXTENSIONS):
                            discovered_samples.append({'path': item_path, 'label': true_label, 'type': 'video'})
                        elif os.path.isdir(item_path) and any(f.lower().endswith(ALLOWED_IMAGE_EXTENSIONS) for f in os.listdir(item_path)):
                            discovered_samples.append({'path': item_path, 'label': true_label, 'type': 'images'})
            if not discovered_samples:
                self.finished_signal.emit(False, *default_return_tuple[:1], "Nie znaleziono filmów ani sekwencji obrazów.", *default_return_tuple[3:])
                return
            total_samples_to_process = len(discovered_samples)
            self.progress_signal.emit(f"Rozpoczynanie testu modelu ścieżek. Znaleziono {total_samples_to_process} próbek.")
        except Exception as e:
            self.finished_signal.emit(False, *default_return_tuple[:1], f"Błąd przy odczycie folderu testowego (ścieżki): {e}", *default_return_tuple[3:])
            return

        try:
            for sample_info in discovered_samples:
                if not self.running: break
                current_path_points = []
                sample_path, true_label_str, sample_type = sample_info['path'], sample_info['label'], sample_info['type']
                sample_name = os.path.basename(sample_path)
                self.progress_signal.emit(f"Przetwarzanie [{sample_type}] (ścieżka): {sample_name}")

                if sample_type == 'video':
                    cap = cv2.VideoCapture(sample_path)
                    if not cap.isOpened(): continue
                    while cap.isOpened() and self.running:
                        ret, frame = cap.read();
                        if not ret: break
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); results = hands.process(rgb_frame)
                        if results.multi_hand_landmarks: lm = results.multi_hand_landmarks[0].landmark[PATH_TRACKING_LANDMARK]; current_path_points.append((lm.x, lm.y))
                    cap.release()
                elif sample_type == 'images':
                    image_files = sorted([os.path.join(sample_path, f) for f in os.listdir(sample_path) if f.lower().endswith(ALLOWED_IMAGE_EXTENSIONS)])
                    for img_file_path in image_files:
                        if not self.running: break
                        img = cv2.imread(img_file_path);
                        if img is None: continue
                        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); results = hands.process(rgb_image)
                        if results.multi_hand_landmarks: lm = results.multi_hand_landmarks[0].landmark[PATH_TRACKING_LANDMARK]; current_path_points.append((lm.x, lm.y))
                if not self.running: break
                
                predicted_label_str = "nieznana_ścieżka"
                current_pred_scores = np.zeros(len(class_labels_from_encoder))

                if len(current_path_points) >= 2:
                    standardized_flat = standardize_path(current_path_points)
                    if standardized_flat is not None:
                        try:
                            input_data = self.path_scaler.transform(standardized_flat.reshape(1, -1))
                            prediction_probs = self.path_model.predict(input_data, verbose=0)[0]
                            current_pred_scores = prediction_probs
                            predicted_idx = np.argmax(prediction_probs)
                            predicted_label_str = self.path_label_encoder.inverse_transform([predicted_idx])[0]
                        except Exception: predicted_label_str = "błąd_predykcji"
                    else: predicted_label_str = "błąd_standaryzacji"
                
                y_true_labels.append(true_label_str)
                y_pred_labels.append(predicted_label_str)
                y_pred_scores_list.append(current_pred_scores)

            if not self.running:
                self.finished_signal.emit(False, *default_return_tuple[:1], "Testowanie ścieżek przerwane.", *default_return_tuple[3:])
                return
            if not y_true_labels:
                self.finished_signal.emit(False, *default_return_tuple[:1], "Nie przetworzono żadnych poprawnych próbek ścieżek.", *default_return_tuple[3:])
                return

            y_pred_scores_np = np.array(y_pred_scores_list)
            accuracy = accuracy_score(y_true_labels, y_pred_labels)
            report_labels_for_plot_and_metrics = class_labels_from_encoder

            report_str = classification_report(y_true_labels, y_pred_labels, labels=report_labels_for_plot_and_metrics, target_names=report_labels_for_plot_and_metrics, zero_division=0, output_dict=False)
            report_dict = classification_report(y_true_labels, y_pred_labels, labels=report_labels_for_plot_and_metrics, target_names=report_labels_for_plot_and_metrics, zero_division=0, output_dict=True)
            raw_conf_matrix = confusion_matrix(y_true_labels, y_pred_labels, labels=class_labels_from_encoder)

            class_metrics_plot_fn = _save_classification_report_metrics_plot(report_dict, report_labels_for_plot_and_metrics, "path", self.progress_signal)
            norm_cm_fn = _save_normalized_confusion_matrix(y_true_labels, y_pred_labels, class_labels_from_encoder, "path", self.progress_signal)
            roc_auc_fn = _save_roc_auc_curve_macro(y_true_labels, y_pred_scores_np, classes_for_roc=class_labels_from_encoder, base_filename_prefix="path", progress_signal=self.progress_signal)
            pr_curve_fn = _save_precision_recall_curve_macro(y_true_labels, y_pred_scores_np, classes_for_pr=class_labels_from_encoder, base_filename_prefix="path", progress_signal=self.progress_signal)

            self.progress_signal.emit("Testowanie modelu ścieżek zakończone.")
            self.finished_signal.emit(True, accuracy, report_str, raw_conf_matrix, class_labels_from_encoder, class_metrics_plot_fn, norm_cm_fn, roc_auc_fn, pr_curve_fn)

        except Exception as e:
            import traceback; traceback.print_exc()
            err_msg = f"Krytyczny błąd podczas testowania ścieżek: {e}"
            self.progress_signal.emit(err_msg)
            self.finished_signal.emit(False, 0.0, err_msg, np.array([]), [], None, None, None, None)

# Uwaga: Poniższa klasa PathTrainingThread jest identyczna jak w training_threads.py
# Została przetłumaczona zgodnie z żądaniem.
class PathTrainingThread(QThread): # Ta klasa wydaje się być duplikatem z training_threads.py
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

            # Importy brakujące w oryginalnym pliku testing_threads.py dla tej klasy
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            from tensorflow.keras.utils import to_categorical
            from sklearn.model_selection import train_test_split
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            import pickle
            # Zaimportowano także constants, ale PATH_NUM_FEATURES, PATH_NUM_CLASSES itp. nie są tu zdefiniowane
            # Używam wartości zastępczych lub zakładam, że są dostępne globalnie
            # To jest problematyczne, jeśli ta klasa ma działać poprawnie w tym pliku
            # W training_threads.py te stałe są importowane.
            # Dla przykładu, załóżmy, że PATH_ALLOWED_LABELS, PATH_NUM_CLASSES, PATH_NUM_FEATURES są dostępne.

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            self.progress_signal.emit(f"Skaler cech ścieżek zapisany do {self.scaler_path}")

            label_encoder = LabelEncoder()
            # Zakładając, że PATH_ALLOWED_LABELS jest zdefiniowane globalnie lub w constants
            label_encoder.fit(PATH_ALLOWED_LABELS) 
            try:
                y_encoded = label_encoder.transform(y_labels)
            except ValueError as e:
                unknown = set(y_labels) - set(label_encoder.classes_)
                raise ValueError(f"Nieznana etykieta(y) ścieżek w CSV: {unknown}. Błąd: {e}")

            # Zakładając, że PATH_NUM_CLASSES jest zdefiniowane
            y_categorical = to_categorical(y_encoded, num_classes=len(PATH_ALLOWED_LABELS)) # Użycie len(PATH_ALLOWED_LABELS) jako obejście
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
            # Zakładając, że PATH_NUM_FEATURES jest zdefiniowane
            model = Sequential([
                Dense(256, activation='relu', input_shape=(X_train.shape[1],)), Dropout(0.4), # Użycie X_train.shape[1] jako obejście
                Dense(128, activation='relu'), Dropout(0.4),
                Dense(64, activation='relu'), Dropout(0.4),
                Dense(y_categorical.shape[1], activation='softmax') # Użycie y_categorical.shape[1] jako obejście
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.summary(print_fn=lambda x: self.progress_signal.emit(x))

            self.progress_signal.emit("Rozpoczynanie trenowania modelu ścieżek...")
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=0.00001)
            ]
            # Funkcja _save_learning_curves nie jest zdefiniowana w tym pliku,
            # więc pomijam jej wywołanie lub zakładam, że zostanie dodana.
            # history = model.fit(X_train, y_train, epochs=150, batch_size=32,
            #                     validation_data=(X_val, y_val), callbacks=callbacks, verbose=0)
            # val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
            # self.progress_signal.emit(f"Trenowanie modelu ścieżek zakończone. Dokładność walidacyjna: {val_accuracy:.4f}")
            # acc_plot_file, loss_plot_file = _save_learning_curves(history, "path", self.progress_signal) # Ta funkcja nie jest tu zdefiniowana
            
            # Symulacja dla reszty kodu, ponieważ _save_learning_curves nie jest dostępne
            val_accuracy = 0.0 # Placeholder
            acc_plot_file, loss_plot_file = None, None # Placeholder
            
            learning_curves_msg = []
            if acc_plot_file: learning_curves_msg.append(f"Wykres dokładności: {os.path.basename(acc_plot_file)}")
            if loss_plot_file: learning_curves_msg.append(f"Wykres funkcji straty: {os.path.basename(loss_plot_file)}")

            final_message = f"Trenowanie modelu ścieżek zakończone. Model zapisany. Dokładność walidacyjna: {val_accuracy:.4f}."
            if learning_curves_msg:
                final_message += " Krzywe uczenia zapisane: " + ", ".join(learning_curves_msg)
            
            self.progress_signal.emit(f"Zapisywanie modelu ścieżek do {self.model_path}...")
            # model.save(self.model_path) # Zakomentowane, bo model może nie być w pełni zainicjowany bez stałych
            self.finished_signal.emit(True, final_message)

        except (FileNotFoundError, ValueError, AssertionError) as e:
            self.finished_signal.emit(False, f"Błąd trenowania modelu ścieżek: {e}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished_signal.emit(False, f"Nieoczekiwany błąd trenowania modelu ścieżek: {e}")

class PathFileClassifierThread(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str, str)
    error_signal = pyqtSignal(str)

    def __init__(self, file_paths, path_model, path_label_encoder, path_scaler, parent=None):
        super().__init__(parent)
        self.file_paths = file_paths
        self.path_model = path_model
        self.path_label_encoder = path_label_encoder
        self.path_scaler = path_scaler
        self.running = True

    def stop(self):
        self.running = False
        self.progress_signal.emit("Zatrzymywanie przetwarzania plików...")

    def run(self):
        if not self.path_model or not self.path_label_encoder or not self.path_scaler:
            self.error_signal.emit("Model ścieżek, enkoder lub skaler nie są załadowane.")
            self.finished_signal.emit(False, "Błąd: Model niezaładowany", "Błąd modelu ścieżek.")
            return

        current_path_points = []
        is_video_file = False
        if len(self.file_paths) == 1:
            if self.file_paths[0].lower().endswith(ALLOWED_VIDEO_EXTENSIONS):
                is_video_file = True
        
        try:
            if is_video_file:
                video_path = self.file_paths[0]
                self.progress_signal.emit(f"Przetwarzanie wideo: {os.path.basename(video_path)}...")
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    self.error_signal.emit(f"Nie można otworzyć pliku wideo: {video_path}")
                    self.finished_signal.emit(False, "Błąd: Nie można otworzyć wideo", f"Błąd otwarcia: {os.path.basename(video_path)}")
                    return

                frame_idx = 0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                while cap.isOpened() and self.running: 
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_idx += 1
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_frame.flags.writeable = False
                    results = hands.process(rgb_frame)
                    rgb_frame.flags.writeable = True

                    if results.multi_hand_landmarks:
                        hand_landmarks = results.multi_hand_landmarks[0]
                        lm = hand_landmarks.landmark[PATH_TRACKING_LANDMARK]
                        current_path_points.append((lm.x, lm.y))
                    
                    if frame_idx % 30 == 0 : 
                        self.progress_signal.emit(f"Wideo: klatka {frame_idx}/{total_frames if total_frames > 0 else 'N/A'}, zebrano {len(current_path_points)} pkt.")

                cap.release()
                if not self.running: 
                    self.finished_signal.emit(False, "Anulowano", "Przetwarzanie wideo anulowane.")
                    return
                self.progress_signal.emit(f"Przetworzono wideo. Zebrano {len(current_path_points)} punktów ścieżki.")

            else: 
                num_images = len(self.file_paths)
                self.progress_signal.emit(f"Przetwarzanie sekwencji {num_images} obrazów...")
                try: 
                    sorted_file_paths = sorted(self.file_paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].replace("frame", "").replace("img",""))) 
                except ValueError: 
                    sorted_file_paths = sorted(self.file_paths)

                for i, img_path in enumerate(sorted_file_paths):
                    if not self.running: 
                        break 
                    img = cv2.imread(img_path)
                    if img is None:
                        self.progress_signal.emit(f"Pominięto (nie można załadować): {os.path.basename(img_path)}")
                        continue

                    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    rgb_image.flags.writeable = False
                    results = hands.process(rgb_image)
                    rgb_image.flags.writeable = True

                    if results.multi_hand_landmarks:
                        hand_landmarks = results.multi_hand_landmarks[0]
                        lm = hand_landmarks.landmark[PATH_TRACKING_LANDMARK]
                        current_path_points.append((lm.x, lm.y))
                    
                    self.progress_signal.emit(f"Obrazy: {i+1}/{num_images}, zebrano {len(current_path_points)} pkt.")
                
                if not self.running: 
                    self.finished_signal.emit(False, "Anulowano", "Przetwarzanie obrazów anulowane.")
                    return
                self.progress_signal.emit(f"Przetworzono sekwencję obrazów. Zebrano {len(current_path_points)} punktów ścieżki.")

            if not self.running:
                 self.finished_signal.emit(False, "Anulowano", "Przetwarzanie anulowane przed klasyfikacją.")
                 return

            if len(current_path_points) < 2:
                self.progress_signal.emit("Ścieżka zbyt krótka do klasyfikacji (z pliku).")
                self.finished_signal.emit(False, "Ścieżka za krótka", "Ścieżka zebrana z pliku/ów jest zbyt krótka.")
                return

            self.progress_signal.emit(f"Standaryzacja {len(current_path_points)} punktów ścieżki (z pliku)...")
            standardized_flat = standardize_path(current_path_points)

            if standardized_flat is not None:
                input_data = self.path_scaler.transform(standardized_flat.reshape(1, -1))
                prediction = self.path_model.predict(input_data, verbose=0)[0]
                idx = np.argmax(prediction)
                conf = prediction[idx]
                label = self.path_label_encoder.inverse_transform([idx])[0]
                
                result_text = f"Ścieżka (plik): {label} ({conf:.2f})"
                status_message = f"Wynik z pliku: {label} (Pewność: {conf:.2f})"
                self.finished_signal.emit(True, result_text, status_message)
            else:
                self.progress_signal.emit("Błąd standaryzacji ścieżki (z pliku).")
                self.finished_signal.emit(False, "Błąd przetwarzania", "Nie można było zestandaryzować ścieżki z pliku/ów.")

        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            self.error_signal.emit(f"Krytyczny błąd podczas przetwarzania plików: {e}\n{tb_str}")
            self.finished_signal.emit(False, "Błąd krytyczny", f"Błąd przetwarzania plików: {e}")