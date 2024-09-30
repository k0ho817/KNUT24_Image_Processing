import sys, cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QLabel, QMainWindow, QPushButton)

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.label = QLabel("Image Here")
        self.label.setFixedSize(640, 480)
        self.setCentralWidget(self.label)

        img = cv2.imread("/Users/mungyeongho/Documents/study/KNUT24/2학기/영상처리/week2/images/Lenna.jpg", cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, ch = img.shape

        img = QImage(img.data, w, h, ch * w, QImage.Format_RGB888)
        scaled_img = img.scaled(640, 480, Qt.KeepAspectRatio)

        self.label.setPixmap(QPixmap.fromImage(scaled_img))
        print("update image")

if __name__ == "__main__":
    app = QApplication()
    w = Window()
    w.show()
    sys.exit(app.exec())