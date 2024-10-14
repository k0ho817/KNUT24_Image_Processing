import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLineEdit, QSizePolicy)
from PySide6.QtGui import QImage, QPixmap
import cv2
import numpy as np

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.edit = QLineEdit('Wrtie image path here')

        self.load_btn = QPushButton('Load Image')
        self.bin_btn = QPushButton('Binary Image')
        self.geo_btn = QPushButton('Geometry Image')

        self.load_btn.clicked.connect(self.image_load)
        self.bin_btn.clicked.connect(self.binary_image)
        self.geo_btn.clicked.connect(self.geometry_image)

        self.label = QLabel()
        self.label.setFixedSize(640, 480)
        self.geo_label = QLabel("Geometry Type:")

        self.combobox = QComboBox()
        self.combobox.addItem('flip')
        self.combobox.addItem('translation')
        self.combobox.addItem('rotation')

        h_layout1 = QHBoxLayout()
        h_layout1.addWidget(self.edit)
        h_layout1.addWidget(self.load_btn)
        h_layout1.addWidget(self.bin_btn)
        self.edit.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.load_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.bin_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        h_layout2 = QHBoxLayout()
        h_layout2.addWidget(self.geo_label, alignment=Qt.AlignmentFlag.AlignRight)
        h_layout2.addWidget(self.combobox)
        h_layout2.addWidget(self.geo_btn)
        self.geo_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.combobox.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.geo_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        v_layout = QVBoxLayout()
        v_layout.addWidget(self.label)
        v_layout.addLayout(h_layout1)
        v_layout.addLayout(h_layout2)

        widget = QWidget(self)
        widget.setLayout(v_layout)
        self.setCentralWidget(widget)

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

    def geometry_image(self):
        self.img = cv2.imread(self.edit.text(), cv2.IMREAD_COLOR)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        h, w, ch = self.img.shape

        if self.combobox.currentIndex() == 0:
            self.img = cv2.flip(self.img, 1)
        elif self.combobox.currentIndex() == 1:
            rows, cols = self.img.shape[:2]
            Mat = np.float32([[1, 0, 50],[0, 1, 20]])
            self.img = cv2.warpAffine(self.img, Mat, (cols, rows),
                                      borderMode=cv2.BORDER_REPLICATE)
        elif self.combobox.currentIndex() == 2:
            rows, cols = self.img.shape[:2]
            Mat = cv2.getRotationMatrix2D((w/2, h/2), 60, 1.0)
            self.img = cv2.warpAffine(self.img, Mat, (cols, rows),
                                      borderMode=cv2.BORDER_REPLICATE)

        self.img = QImage(self.img.data, w, h, ch * w, QImage.Format_RGB888)
        scaled_img = self.img.scaled(640, 480, Qt.KeepAspectRatio)

        self.label.setPixmap(QPixmap.fromImage(scaled_img))
        print("update image")

if __name__ == '__main__':
    app = QApplication()
    w = Window()
    w.show()
    sys.exit(app.exec())