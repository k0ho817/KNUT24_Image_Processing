import sys, time, queue
from PySide6.QtCore import Qt, QTimer, QSize, QThread
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout,
                               QWidget, QLineEdit, QSizePolicy, QGridLayout, QRadioButton)
from PySide6.QtGui import QImage, QPixmap
import cv2
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
import threading
import pyautogui

sema0_1 = threading.Semaphore(0)
sema0_2 = threading.Semaphore(0)
sema1 = threading.Semaphore(0)
sema2 = threading.Semaphore(0)

Processing_stop = False

class Thread_in(threading.Thread):
    def __init__(self, img_queue):
        threading.Thread.__init__(self)
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
            global Processing_stop
            if Processing_stop is True:
                sema1.release()
                sema0_1.acquire()

            curTime = time.time()  # 현재 시간
            sec = curTime - prevTime
            if sec > 1 / fps:
                prevTime = curTime
                ret, frame = self.capture.read()
                if not ret:
                    continue
                self.qu.put(frame)


class Thread_out(threading.Thread):
    def __init__(self, img_queue, proc_queue):
        threading.Thread.__init__(self)
        self.status = True
        self.qu = img_queue
        self.qu_img_to_app = proc_queue
        self.EDGE_TYPE = None
        self.cnt = 0

    def run(self):
        global Processing_stop
        while self.status:
            if Processing_stop is True:
                sema2.release()
                sema0_2.acquire()

            if self.qu.qsize() > 0:
                # print("qu size: ", self.qu.qsize())

                cnt_edge = 10
                frame = self.qu.get()

                if self.EDGE_TYPE == 'Laplacian':
                    frame = cv2.Laplacian(frame, cv2.CV_8U, ksize=3)

                elif self.EDGE_TYPE == 'Canny':
                    frame = cv2.Canny(frame, 150, 300)
                    cnt_edge = self.sum_edge(frame)

                if len(frame.shape) < 3:
                    h, w = frame.shape
                    ch = 1
                    img_format = QImage.Format_Grayscale8
                else:
                    h, w, ch = frame.shape
                    img_format = QImage.Format_RGB888
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frame = QImage(frame.data, w, h, ch * w, img_format)
                frame = frame.scaled(640, 480, Qt.KeepAspectRatio)

                qu_val = [frame, cnt_edge]

                self.qu_img_to_app.put_nowait(qu_val)
        
    def sum_edge(self, frame):
        ratio = 480 / frame.shape[0]
        img = cv2.resize(frame, None, fx=ratio, fy=ratio)

        temp = img > 0

        sum_value = temp.sum()
        return sum_value



class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.th_in = None
        self.th_out = None

        self.img_size = QSize(640, 480)
        self.cnt = 0
        self.MODE_VIDEO = False
        self.EDGE_TYPE = None
        self.previous_plot = None

        self.canvas = FigureCanvas(Figure(figsize=(0, 0.5)))
        self.axes = self.canvas.figure.subplots()
        self.y_max = 0

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
        display_change_btn = QPushButton('Display Change')

        # link function
        load_btn.clicked.connect(self.load_img_func)
        bin_btn.clicked.connect(self.binary_image)
        geo_btn.clicked.connect(self.geometry_image)
        perspective.clicked.connect(self.perspective_image)
        init_pos.clicked.connect(self.initialize_pos)
        edge_btn.clicked.connect(self.filtering)
        display_change_btn.clicked.connect(self.dp_change)

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
        v_layout.addWidget(display_change_btn)

        widget = QWidget(self)
        widget.setLayout(v_layout)
        self.setCentralWidget(widget)
    
    def dp_change(self):
        pyautogui.hotkey('alt', 'tab')
        print("display changed")

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
        if self.MODE_VIDEO is True:
            if self.filter_combobox.currentText() == 'None':
                self.th_out.EDGE_TYPE = None
                return
            elif self.filter_combobox.currentText() == 'Canny':
                self.th_out.EDGE_TYPE ='Canny'
                self.EDGE_TYPE ='Canny'
                return
            elif self.filter_combobox.currentText() == 'Laplacian':
                self.th_out.EDGE_TYPE ='Laplacian'
                self.EDGE_TYPE ='Laplacian'
                return
        if self.m_proc_img is not None:
            if len(self.m_proc_img.shape) >= 3:
                self.m_proc_img = cv2.cvtColor(self.m_proc_img, cv2.COLOR_BGR2GRAY)
            if self.filter_combobox.currentText() == 'Laplacian':
                print("Laplacian")
                l_image = cv2.Laplacian(self.m_proc_img, cv2.CV_8U, ksize=3)
                self.update_image(l_image)
            elif self.filter_combobox.currentText() == 'Canny':
                print("Canny")
                c_image1 = cv2.Canny(self.m_proc_img, 150, 300)
                self.update_image(c_image1)

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
            path = self.edit.text()
            self.createThread_start(path)
        elif self.radio_webcam.isChecked() is True:
            self.createThread_start(0)
            # self.setup_camera(0)

    def setup_camera(self, vid):
        self.capture = cv2.VideoCapture(vid)
        if not self.capture.isOpened():
            print("Camera open failed")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_size.width())
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_size.height())

        self.timer.start(30)

    def display_video_stream(self):
        if self.qu_img_to_app.empty() is False:
            qu_val = self.qu_img_to_app.get_nowait()

            frame = qu_val[0]
            cnt_edge = qu_val[1]

            self.update_image2(frame)

            if self.EDGE_TYPE == 'Canny':
                if cnt_edge is not None:
                    print(cnt_edge) 
                    if cnt_edge > 2375:
                        self.dp_change()
                    self.update_plot2(cnt_edge)
    
    def update_image2(self, scaled_img):
        self.label.setFixedSize(scaled_img.width(), scaled_img.height())
        self.label.setPixmap(QPixmap.fromImage(scaled_img))
    
    def update_plot2(self, sum_value):
        self.ydata = self.ydata[1:] + [sum_value]
        if sum_value > self.y_max:
            self.y_max = sum_value
            self.axes.set_ylim([0, self.y_max + 10])
        if self.previous_plot is None:
            self.previous_plot = self.axes.plot(self.xdata, self.ydata,'r')[0]
        else:
            self.previous_plot.set_ydata(self.ydata)

        global Processing_stop
        Processing_stop = True
        sema1.acquire()
        sema2.acquire()
        
        prevTime = time.time() # 현재 시간
        self.canvas.draw()
        Processing_stop = False
        sema0_1.release()
        sema0_2.release()
        curTime = time.time() # 현재 시간
        sec = curTime - prevTime
        print(sec%60)

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
            self.pos2_y.setText(f'{y}')
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
    
    def createThread_start(self, vid):
        self.MODE_VIDEO = True
        self.qu = queue.Queue()
        self.qu_img_to_app = queue.Queue()
        self.th_in = Thread_in(self.qu)
        self.th_out = Thread_out(self.qu, self.qu_img_to_app)

        self.th_in.vid = vid
        self.th_in.start()
        self.th_out.start()
        self.timer.start(30)
    
    def kill_thread(self):
        self.timer.stop()
        if self.th_in is not None:
            if self.th_in.is_alive() is True:
                self.th_in.status = False
                self.th_in.join()
                print("Thread_in END")
            if self.th_in.capture is not None:
                if self.th_in.capture.isOpened is True:
                    self.th_in.capture.release()

        if self.th_out is not None:
            if self.th_out.is_alive() is True:
                self.th_out.status = False
                self.th_out.join()
                print("Thread_out END")

if __name__ == '__main__':
    app = QApplication()
    w = Window()
    w.show()
    sys.exit(app.exec())
    w.kill_thread()
