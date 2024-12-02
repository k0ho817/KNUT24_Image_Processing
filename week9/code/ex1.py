import sys, time, queue
from PySide6.QtCore import Qt, QTimer, QSize, QThread, Signal
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout,
                               QWidget, QLineEdit, QSizePolicy, QGridLayout, QRadioButton)
from PySide6.QtGui import QImage, QPixmap
import cv2
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib


class Thread_in(QThread):
    def __init__(self, img_queue, parent=None):
        QThread.__init__(self, parent)
        self.status = True
        self.capture = None
        self.qu = img_queue
        self.vid = None
    def run(self):
        self.capture = cv2.VideoCapture(self.vid)
        if not self.capture.isOpened():
            print("Camera open failed")
        prevTime = 0
        fps = 24
        while self.status:
            curTime = time.time() # 현재 시간
            sec = curTime - prevTime
            if sec > 1/fps:
                prevTime = curTime
                ret, frame = self.capture.read()
                if not ret:
                    continue
                self.qu.put(frame)
        sys.exit(-1)


class Thread_out(QThread):
    updateFrame = Signal(object)
    updatePlot = Signal(object)

    def __init__(self, img_queue, parent=None):
        QThread.__init__(self, parent)
        self.status = True
        self.qu = img_queue
        self.EDGE_TYPE = None

    def run(self):
        while self.status:
            frame = self.qu.get()
            if self.EDGE_TYPE == 'Sobel XY':
                frame = cv2.Sobel(frame, cv2.CV_8U, 1, 1, ksize=3)
                self.updatePlot.emit(frame)
            elif self.EDGE_TYPE == 'Scharr X':
                frame = cv2.Scharr(frame, cv2.CV_8U, 1, 0)
                self.updatePlot.emit(frame)
            elif self.EDGE_TYPE == 'Scharr Y':
                frame = cv2.Scharr(frame, cv2.CV_8U, 0, 1)
                self.updatePlot.emit(frame)
            elif self.EDGE_TYPE == 'Laplacian':
                frame = cv2.Laplacian(frame, cv2.CV_8U, ksize=3)
                self.updatePlot.emit(frame)
            elif self.EDGE_TYPE == 'Canny':
                frame = cv2.Canny(frame, 50, 100)
                self.updatePlot.emit(frame)

            self.updateFrame.emit(frame)
        sys.exit(-1)


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.m_proc_img = None

        self.qu = queue.Queue()

        self.th_in = Thread_in(self.qu)
        self.th_out = Thread_out(self.qu)

        self.th_out.updateFrame.connect(self.update_image)
        self.th_out.updatePlot.connect(self.update_plot)

        self.img_size = QSize(640, 480)
        self.cnt = 0
        self.MODE_VIDEO = False
        self.EDGE_TYPE = None
        self.previous_plot = None

        self.canvas = FigureCanvas(Figure(figsize=(0, 0.5)))
        self.axes = self.canvas.figure.subplots()

        self.axes.set_ylim([10000, 20000])

        n_data = 50
        self.xdata = list(range(n_data))
        self.axes.set_xticks(self.xdata, [])
        self.ydata = [0 for i in range(n_data)]

        # create LineEdit Object
        self.edit = QLineEdit('Write image path here')
        self.pos1_x = QLineEdit('')
        self.pos1_y = QLineEdit('')
        self.pos2_x = QLineEdit('')
        self.pos2_y = QLineEdit('')
        self.pos3_x = QLineEdit('')
        self.pos3_y = QLineEdit('')
        self.pos4_x = QLineEdit('')
        self.pos4_y = QLineEdit('')

        # create button object
        load_btn = QPushButton('Load Image')
        bin_btn = QPushButton('Binary Image')
        geo_btn = QPushButton('Geometry Image')
        init_pos = QPushButton('Initialize Pos')
        perspective = QPushButton('Perspective Image')
        edge_btn = QPushButton('Edge Detection')

        # link function
        load_btn.clicked.connect(self.load_img_func)
        bin_btn.clicked.connect(self.binary_image)
        geo_btn.clicked.connect(self.geometry_image)
        perspective.clicked.connect(self.perspective_image)
        init_pos.clicked.connect(self.initialize_pos)
        edge_btn.clicked.connect(self.filtering)

        # crlabel object
        self.label = QLabel()
        self.label.setFixedSize(640, 480)
        geo_label = QLabel("Geometry Type:")
        pos1_label = QLabel('Pos1')
        pos2_label = QLabel('Pos2')
        pos3_label = QLabel('Pos3')
        pos4_label = QLabel('Pos4')
        filter_label = QLabel('Filter type')

        # create combobox object
        self.combobox = QComboBox()
        self.combobox.addItem('flip')
        self.combobox.addItem('translation')
        self.combobox.addItem('rotation')

        self.filter_combobox = QComboBox()
        self.filter_combobox.addItem('None')
        self.filter_combobox.addItem('Sobel_XY')
        self.filter_combobox.addItem('Scharr_X')
        self.filter_combobox.addItem('Scharr_Y')
        self.filter_combobox.addItem('Laplacian')
        self.filter_combobox.addItem('Canny')

        self.radio_image = QRadioButton('Image')
        self.radio_video = QRadioButton('Video')
        self.radio_webcam = QRadioButton('Webcam')
        self.radio_image.setChecked(True)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.display_video_stream)

        h_layout0 = QHBoxLayout()
        h_layout0.addWidget(self.label)
        h_layout0.addWidget(self.canvas)

        h_layout_radio = QHBoxLayout()
        h_layout_radio.addWidget(self.radio_image)
        h_layout_radio.addWidget(self.radio_video)
        h_layout_radio.addWidget(self.radio_webcam)

        h_layout1 = QHBoxLayout()
        h_layout1.addWidget(self.edit)
        h_layout1.addLayout(h_layout_radio)
        h_layout1.addWidget(load_btn)
        h_layout1.addWidget(bin_btn)
        self.edit.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        load_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        bin_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        h_layout2 = QHBoxLayout()
        h_layout2.addWidget(geo_label, alignment=Qt.AlignmentFlag.AlignRight)
        h_layout2.addWidget(self.combobox)
        h_layout2.addWidget(geo_btn)
        geo_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.combobox.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        geo_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        pos_grid = QGridLayout()
        pos_grid.addWidget(pos1_label, 0, 0)
        pos_grid.addWidget(self.pos1_x, 0, 1)
        pos_grid.addWidget(self.pos1_y, 0, 2)
        pos_grid.addWidget(pos2_label, 0, 3)
        pos_grid.addWidget(self.pos2_x, 0, 4)
        pos_grid.addWidget(self.pos2_y, 0, 5)
        pos_grid.addWidget(pos3_label, 1, 0)
        pos_grid.addWidget(self.pos3_x, 1, 1)
        pos_grid.addWidget(self.pos3_y, 1, 2)
        pos_grid.addWidget(pos4_label, 1, 3)
        pos_grid.addWidget(self.pos4_x, 1, 4)
        pos_grid.addWidget(self.pos4_y, 1, 5)

        h_layout3 = QHBoxLayout()
        h_layout3.addLayout(pos_grid)
        h_layout3.addWidget(init_pos)
        h_layout3.addWidget(perspective)
        init_pos.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        perspective.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        filter_layout = QHBoxLayout()
        filter_layout.addWidget(filter_label, alignment=Qt.AlignmentFlag.AlignRight)
        filter_layout.addWidget(self.filter_combobox)
        filter_layout.addWidget(edge_btn)
        filter_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.filter_combobox.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        edge_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        v_layout = QVBoxLayout()
        v_layout.addLayout(h_layout0)
        v_layout.addLayout(h_layout1)
        v_layout.addLayout(h_layout2)
        v_layout.addLayout(h_layout3)
        v_layout.addLayout(filter_layout)

        widget = QWidget(self)
        widget.setLayout(v_layout)
        self.setCentralWidget(widget)

    def binary_image(self):
        if len(self.m_proc_img.shape) > 2:
            self.m_proc_img = cv2.cvtColor(self.m_proc_img, cv2.COLOR_BGR2GRAY)

        _, self.m_proc_img = cv2.threshold(self.m_proc_img, 100, 255, cv2.THRESH_BINARY)

        self.update_image(self.m_proc_img)
        print("update image")

    def geometry_image(self):       
        if self.combobox.currentIndex() == 0:
            self.m_proc_img = cv2.flip(self.m_proc_img, 1)
        elif self.combobox.currentIndex() == 1:
            rows, cols = self.m_proc_img.shape[:2]
            Mat = np.float32([[1, 0, 50], [0, 1, 20]])
            self.m_proc_img = cv2.warpAffine(self.m_proc_img, Mat, (cols, rows),
                                             borderMode=cv2.BORDER_REPLICATE)
        elif self.combobox.currentIndex() == 2:
            rows, cols = self.m_proc_img.shape[:2]
            Mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), 60, 1.0)
            self.m_proc_img = cv2.warpAffine(self.m_proc_img, Mat, (cols, rows),
                                             borderMode=cv2.BORDER_REPLICATE)
        self.update_image(self.m_proc_img)
        print("update image")

    def filtering(self):
        if self.m_proc_img is not None:
            if len(self.m_proc_img.shape) > 2:
                self.m_proc_img = cv2.cvtColor(self.m_proc_img, cv2.COLOR_BGR2GRAY)

        if self.filter_combobox.currentText() == 'None':
            if self.MODE_VIDEO is True:
                self.EDGE_TYPE = None
                self.th_out.EDGE_TYPE = None
            return
        elif self.filter_combobox.currentIndex() == 1:
            if self.MODE_VIDEO is True:
                self.EDGE_TYPE = 'Sobel XY'
                self.th_out.EDGE_TYPE = 'Sobel XY'
                return
            edge_img = cv2.Sobel(self.m_proc_img, cv2.CV_8U, 1, 1, ksize=3)
        elif self.filter_combobox.currentIndex() == 2:
            if self.MODE_VIDEO is True:
                self.EDGE_TYPE = 'Scharr X'
                self.th_out.EDGE_TYPE = 'Scharr X'
                return
            edge_img = cv2.Scharr(self.m_proc_img, cv2.CV_8U, 1, 0)
        elif self.filter_combobox.currentIndex() == 3:
            if self.MODE_VIDEO is True:
                self.EDGE_TYPE = 'Scharr Y'
                self.th_out.EDGE_TYPE = 'Scharr Y'
                return
            edge_img = cv2.Scharr(self.m_proc_img, cv2.CV_8U, 0, 1)
        elif self.filter_combobox.currentIndex() == 4:
            if self.MODE_VIDEO is True:
                self.EDGE_TYPE = 'Laplacian'
                self.th_out.EDGE_TYPE = 'Laplacian'
                return
            edge_img = cv2.Laplacian(self.m_proc_img, cv2.CV_8U, ksize=3)
        elif self.filter_combobox.currentIndex() == 5:
            if self.MODE_VIDEO is True:
                self.EDGE_TYPE = 'Canny'
                self.th_out.EDGE_TYPE = 'Canny'
                return
            edge_img = cv2.Canny(self.m_proc_img, 10, 50)

        self.update_image(edge_img)

    def update_image(self, img):
        if len(img.shape) < 3:
            h, w = img.shape
            ch = 1
            img_format = QImage.Format_Grayscale8
        else:
            h, w, ch = img.shape
            img_format = QImage.Format_RGB888
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = QImage(img.data, w, h, ch * w, img_format)
        scaled_img = img.scaled(self.img_size.width(), self.img_size.height(), Qt.KeepAspectRatio)

        self.label.setPixmap(QPixmap.fromImage(scaled_img))

    def load_img_func(self):
        if self.radio_image.isChecked() is True:
            self.MODE_VIDEO = False
            self.m_main_img = cv2.imread(f"{self.edit.text()}", cv2.IMREAD_COLOR)
            self.m_main_img = cv2.resize(self.m_main_img, (self.img_size.width(), self.img_size.height()),
                                         interpolation=cv2.INTER_CUBIC)
            self.m_proc_img = self.m_main_img.copy()
            self.update_image(self.m_proc_img)
            print('update image')
        elif self.radio_video.isChecked() is True:
            self.MODE_VIDEO = True
            self.th_in.vid = self.edit.text()
            self.th_in.start()
            self.th_out.start()
        elif self.radio_webcam.isChecked() is True:
            self.MODE_VIDEO = True
            self.th_in.vid = 0
            self.th_in.start()
            self.th_out.start()
            # self.setup_camera(0)

    def setup_camera(self, vid):
        self.capture = cv2.VideoCapture(vid)
        if not self.capture.isOpened():
            print("Camera open failed")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_size.width())
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_size.height())

        self.timer.start(30)

    def display_video_stream(self):
        retval, self.m_proc_img = self.capture.read()
        if not retval:
            return

        if self.EDGE_TYPE == 'Sobel XY':
            self.m_proc_img = cv2.Sobel(self.m_proc_img, cv2.CV_8U, 1, 1, ksize=3)
            self.update_plot(self.m_proc_img)
        elif self.EDGE_TYPE == 'Scharr X':
            self.m_proc_img = cv2.Scharr(self.m_proc_img, cv2.CV_8U, 1, 0)
            self.update_plot(self.m_proc_img)
        elif self.EDGE_TYPE == 'Scharr Y':
            self.m_proc_img = cv2.Scharr(self.m_proc_img, cv2.CV_8U, 0, 1)
            self.update_plot(self.m_proc_img)
        elif self.EDGE_TYPE == 'Laplacian':
            self.m_proc_img = cv2.Laplacian(self.m_proc_img, cv2.CV_8U, ksize=3)
            self.update_plot(self.m_proc_img)
        elif self.EDGE_TYPE == 'Canny':
            self.m_proc_img = cv2.Canny(self.m_proc_img, 50, 100)
            self.update_plot(self.m_proc_img)

        self.update_image(self.m_proc_img)

        if self.MODE_VIDEO is False:
            self.timer.stop()

    def mousePressEvent(self, event):
        x = event.position().x() - self.label.x()
        y = event.position().y() - self.label.y()
        x, y = int(x), int(y)

        if self.cnt == 0:
            self.pos1_x.setText(f'{x}')
            self.pos1_y.setText(f'{y}')
            self.cnt += 1
            cv2.circle(self.m_proc_img, (x, y), 5, (255, 0, 0), -1)

        elif self.cnt == 1:
            self.pos2_x.setText(f'{x}')
            self.pos2_y.setText(f'{y}'
            self.cnt += 1
            cv2.circle(self.m_proc_img, (x, y), 5, (0, 255, 0), -1)

        elif self.cnt == 2:
            self.pos3_x.setText(f'{x}')
            self.pos3_y.setText(f'{y}')
            self.cnt += 1
            cv2.circle(self.m_proc_img, (x, y), 5, (0, 0, 255), -1)

        elif self.cnt == 3:
            self.pos4_x.setText(f'{x}')
            self.pos4_y.setText(f'{y}')
            self.cnt += 1
            cv2.circle(self.m_proc_img, (x, y), 5, (255, 0, 255), -1)
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
        pts2 = np.float32([[0, 0], [cols - 1, 0], [cols - 1, rows - 1], [0, rows - 1]])
        Mat1 = cv2.getPerspectiveTransform(pts1, pts2)
        self.m_proc_img = cv2.warpPerspective(self.m_proc_img, Mat1, (cols, rows))

        self.update_image(self.m_proc_img)

    def update_plot(self, img):
        temp = img > 250
        sum_value = temp.sum()

        self.ydata = self.ydata[1:] + [sum_value]

        if self.previous_plot is None:
            self.previous_plot = self.axes.plot(self.xdata, self.ydata, 'r')[0]
        else:
            self.previous_plot.set_ydata(self.ydata)
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication()
    w = Window()
    w.show()
    sys.exit(app.exec())
