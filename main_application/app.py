# app.py

import sys
import mediapipe
from PyQt5.QtWidgets import QApplication

from main_window import MainWindow
from mp_setup import close_hands

if __name__ == '__main__':
    app = QApplication(sys.argv)

    main_window = MainWindow()
    main_window.show()
    exit_code = app.exec_()
    sys.exit(exit_code)

