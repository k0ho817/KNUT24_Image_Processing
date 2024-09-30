import sys, cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QLabel, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QWidget)

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.label = QLabel("Image Here")
        self.label.setFixedSize(640, 480)

        self.edit = QLineEdit("Write image path here")
        self.button1 = QPushButton("Load Image")
        self.button2 = QPushButton("이진화")

        self.layout_h = QHBoxLayout()
        self.layout_h.addWidget(self.edit)
        self.layout_h.addWidget(self.button1)
        self.button1.clicked.connect(self.image_load)
        self.layout_h.addWidget(self.button2)
        self.button2.clicked.connect(self.binary_image)

        self.widget1 = QWidget(self)
        self.widget1.setLayout(self.layout_h)
        self.setCentralWidget(self.widget1)

        self.layout_v = QVBoxLayout()
        self.layout_v.addWidget(self.label)
        self.layout_v.addWidget(self.widget1)
        
        self.widget = QWidget(self)
        self.widget.setLayout(self.layout_v)
        self.setCentralWidget(self.widget)

    def image_load(self):
        self.img = cv2.imread(self.edit.text(), cv2.IMREAD_COLOR)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        h, w, ch = self.img.shape

        self.img = QImage(self.img.data, w, h, ch * w, QImage.Format_RGB888)
        scaled_img = self.img.scaled(640, 480, Qt.KeepAspectRatio)

        self.label.setPixmap(QPixmap.fromImage(scaled_img))
        print("update image")
    
    def binary_image(self):
        self.bin_img = cv2.imread(self.edit.text(), cv2.IMREAD_GRAYSCALE)
        _, self.dst = cv2.threshold(self.bin_img, 100, 255, cv2.THRESH_BINARY)
        h, w = self.bin_img.shape

        self.dst = QImage(self.dst.data, w, h, QImage.Format_Grayscale8)
        self.bin_scaled_img = self.dst.scaled(640, 480, Qt.KeepAspectRatio)

        self.label.setPixmap(QPixmap.fromImage(self.bin_scaled_img))
        print("update image")
if __name__ == "__main__":
    app = QApplication()
    w = Window()
    w.show()
    sys.exit(app.exec())