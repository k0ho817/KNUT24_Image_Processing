import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLineEdit, QSizePolicy, QGridLayout)
from PySide6.QtGui import QImage, QPixmap
import cv2
import numpy as np

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cnt = 0
        self.edit = QLineEdit('Wrtie image path here')
        self.pos1_x = QLineEdit('')
        self.pos1_y = QLineEdit('')
        self.pos2_x = QLineEdit('')
        self.pos2_y = QLineEdit('')
        self.pos3_x = QLineEdit('')
        self.pos3_y = QLineEdit('')
        self.pos4_x = QLineEdit('')
        self.pos4_y = QLineEdit('')

        self.load_btn = QPushButton('Load Image')
        self.bin_btn = QPushButton('Binary Image')
        self.geo_btn = QPushButton('Geometry Image')
        self.init_pos = QPushButton('Initialize Pos')
        self.perspective = QPushButton('Perspective Image')

        self.load_btn.clicked.connect(self.image_load)
        self.bin_btn.clicked.connect(self.binary_image)
        self.geo_btn.clicked.connect(self.geometry_image)
        self.perspective.clicked.connect(self.perspective_image)
        self.init_pos.clicked.connect(self.initialize_pos)

        self.label = QLabel()
        self.label.setFixedSize(640, 480)
        self.geo_label = QLabel("Geometry Type:")
        self.pos1_label = QLabel('Pos1')
        self.pos2_label = QLabel('Pos2')
        self.pos3_label = QLabel('Pos3')
        self.pos4_label = QLabel('Pos4')

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
        
        pos_grid = QGridLayout()
        pos_grid.addWidget(self.pos1_label, 0,0)
        pos_grid.addWidget(self.pos1_x, 0,1)
        pos_grid.addWidget(self.pos1_y, 0,2)
        pos_grid.addWidget(self.pos2_label, 0,3)
        pos_grid.addWidget(self.pos2_x, 0,4)
        pos_grid.addWidget(self.pos2_y, 0,5)
        pos_grid.addWidget(self.pos3_label, 1,0)
        pos_grid.addWidget(self.pos3_x, 1,1)
        pos_grid.addWidget(self.pos3_y, 1,2)
        pos_grid.addWidget(self.pos4_label, 1,3)
        pos_grid.addWidget(self.pos4_x, 1,4)
        pos_grid.addWidget(self.pos4_y, 1,5)

        h_layout3 = QHBoxLayout()
        h_layout3.addLayout(pos_grid)
        h_layout3.addWidget(self.init_pos)
        h_layout3.addWidget(self.perspective)
        self.init_pos.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.perspective.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        v_layout = QVBoxLayout()
        v_layout.addWidget(self.label)
        v_layout.addLayout(h_layout1)
        v_layout.addLayout(h_layout2)
        v_layout.addLayout(h_layout3)

        widget = QWidget(self)
        widget.setLayout(v_layout)
        self.setCentralWidget(widget)

    def image_load(self):
        self.load_img_func()
        print("update image")

    def binary_image(self):
        self.bin_img = cv2.cvtColor(self.m_proc_img, cv2.COLOR_BGR2GRAY)
        _, self.dst = cv2.threshold(self.bin_img, 100, 255, cv2.THRESH_BINARY)
        self.update_image(self.dst)
        print("update image")

    def geometry_image(self):
        if self.combobox.currentIndex() == 0:
            self.m_proc_img = cv2.flip(self.m_proc_img, 1)
        elif self.combobox.currentIndex() == 1:
            rows, cols = self.m_proc_img.shape[:2]
            Mat = np.float32([[1, 0, 50],[0, 1, 20]])
            self.m_proc_img = cv2.warpAffine(self.m_proc_img, Mat, (cols, rows),
                                      borderMode=cv2.BORDER_REPLICATE)
        elif self.combobox.currentIndex() == 2:
            rows, cols = self.m_proc_img.shape[:2]
            Mat = cv2.getRotationMatrix2D((cols/2, rows/2), 60, 1.0)
            self.m_proc_img = cv2.warpAffine(self.m_proc_img, Mat, (cols, rows),
                                      borderMode=cv2.BORDER_REPLICATE)
        self.update_image(self.m_proc_img)
        print("update image")
    
    def update_image(self, img):
        if len(img.shape) < 3:
            h, w = img.shape
            ch = 1
            img_format = QImage.Format_Grayscale8
        else :
            h, w, ch = img.shape
            img_format = QImage.Format_RGB888
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = QImage(img.data, w, h, ch * w, img_format)
        scaled_img = img.scaled(640, 480, Qt.KeepAspectRatio)

        self.label.setPixmap(QPixmap.fromImage(scaled_img))
    
    def load_img_func(self):
        self.m_main_img = cv2.imread(f"{self.edit.text()}", cv2.IMREAD_COLOR)
        self.m_main_img = cv2.resize(self.m_main_img, (640, 480))
        self.m_proc_img = self.m_main_img.copy()
        self.update_image(self.m_proc_img)
    
    def mousePressEvent(self, event):
        x = event.position().x() - self.label.x()
        y = event.position().y() - self.label.y()
        x,y = int(x), int(y)

        if self.cnt == 0:
            self.pos1_x.setText(f'{x}')
            self.pos1_y.setText(f'{y}')
            self.cnt += 1
            cv2.circle(self.m_proc_img, (x,y), 5, (255,0,0), -1)
        
        elif self.cnt == 1:
            self.pos2_x.setText(f'{x}')
            self.pos2_y.setText(f'{y}')
            self.cnt += 1
            cv2.circle(self.m_proc_img, (x,y), 5, (0,255,0), -1)
        
        elif self.cnt == 2:
            self.pos3_x.setText(f'{x}')
            self.pos3_y.setText(f'{y}')
            self.cnt += 1
            cv2.circle(self.m_proc_img, (x,y), 5, (0,0,255), -1)
        
        elif self.cnt == 3:
            self.pos4_x.setText(f'{x}')
            self.pos4_y.setText(f'{y}')
            self.cnt += 1
            cv2.circle(self.m_proc_img, (x,y), 5, (255,0,255), -1)
        self.update_image(self.m_proc_img)
    
    def initialize_pos(self):
        self.cnt = 0
        self.pos1_x.setText('')
        self.pos1_y.setText('')
        self.pos2_x.setText('')
        self.pos2_y.setText('')
        self.pos3_x.setText('')
        self.pos3_y.setText('')
        self.pos4_x.setText('')
        self.pos4_y.setText('')
        self.m_proc_img = self.m_main_img.copy()
        self.update_image(self.m_proc_img)

    def perspective_image(self):
        rows, cols = self.m_proc_img.shape[:2]
        x1 = self.pos1_x.text()
        y1 = self.pos1_y.text()
        x2 = self.pos2_x.text()
        y2 = self.pos2_y.text()
        x3 = self.pos3_x.text()
        y3 = self.pos3_y.text()
        x4 = self.pos4_x.text()
        y4 = self.pos4_y.text()

        pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        pts2 = np.float32([[0,0], [cols-1, 0], [cols-1, rows-1], [0, rows-1]])
        Mat1 = cv2.getPerspectiveTransform(pts1, pts2)
        self.m_proc_img = cv2.warpPerspective(self.m_proc_img, Mat1, (cols, rows))

        self.update_image(self.m_proc_img)

if __name__ == '__main__':
    app = QApplication()
    w = Window()
    w.show()
    sys.exit(app.exec())