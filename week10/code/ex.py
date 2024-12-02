import threading
import sys
import time
import os
import pyautogui

import queue

import numpy as np
import cv2
from PySide6.QtCore import Qt, QThread, Signal, Slot, QSize, QTimer
from PySide6.QtGui import QAction, QImage, QPixmap, QCloseEvent
from PySide6.QtWidgets import (QApplication, QLineEdit, QComboBox, QGridLayout, QWidget,
                               QHBoxLayout, QLabel, QMainWindow, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget, QRadioButton, QScrollBar, QFrame)
import random
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

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
        sys.exit(-1)


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
                cnt_edge = 10
                frame = self.qu.get()

                if self.EDGE_TYPE == 'Laplacian':
                    frame = cv2.Laplacian(frame, cv2.CV_8U, ksize=3)
                    # self.updatePlot.emit(frame)
                elif self.EDGE_TYPE == 'Canny':
                    frame = cv2.Canny(frame, 100, 150)
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
        try:
            self.model = tf.keras.models.load_model('model_ex6.h5')
        except:
            self.model = None
            pass
        # Title and dimensions
        self.setWindowTitle("Abnormal detection")
        self.setGeometry(0, 0, 1200, 500)

        self.y_max = 0
        self.canvas = FigureCanvas(Figure(figsize=(0, 0.5)))
        self.axes = self.canvas.figure.subplots()
        # self.axes.set_ylim([0,5000])

        n_data = 50
        self.xdata = list(range(n_data))
        self.axes.set_xticks(self.xdata, [])
        self.ydata = [0 for i in range(n_data)]

        # self.m_main_img = None
        self.m_proc_img = None
        self.MODE_VIDEO = False
        self.th_in = None
        self.th_out = None
        self.EDGE_TYPE = None
        self.previous_plot = None
        self.labeling_capture = None
        self.labeling_Ground_truth = []
        self.cnt = 0

        self.label_image = QLabel(self)
        self.img_size = QSize(640, 480)
        self.label_image.setFixedSize(self.img_size)

        self.scroll_bar = QScrollBar(Qt.Horizontal)  # 수평 스크롤바
        self.scroll_bar.setMinimum(0)  # 최소값 설정
        self.scroll_bar.setMaximum(100)  # 최대값 설정
        self.scroll_bar.setValue(0)  # 초기 값 설정
        self.scroll_bar.setVisible(False)
        self.scroll_bar.valueChanged.connect(self.change_frame)

        self.label_text_scroll = QLabel("Index Number: ")

        self.label_idx_scroll = QLabel()
        self.label_idx_scroll.setFrameStyle(QFrame.Box | QFrame.Raised)

        self.label_text_labeling = QLabel("Labeling(0 or 1): ")
        self.edit_text_labeling = QLineEdit()  #
        self.edit_text_labeling.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.edit_text_labeling.setFixedWidth(20)
        self.edit_text_labeling.setMaxLength(1)

        # 구간 레이블링 버튼
        self.button_start_labeling = QPushButton("section(Start)")
        self.button_end_labeling = QPushButton("section(End)")
        self.button_end_labeling.setEnabled(False)

        self.edit_section_label = QLineEdit()
        self.edit_section_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.edit_section_label.setFixedWidth(20)
        self.edit_section_label.setMaxLength(1)

        self.button_save_label = QPushButton("Save Label")
        self.button_training = QPushButton("Training")

        layout_save_training = QHBoxLayout()
        layout_save_training.addWidget(self.button_save_label)
        layout_save_training.addWidget(self.button_training)

        layout_scroll_idx = QHBoxLayout()
        layout_scroll_idx.addWidget(self.label_text_scroll, alignment=Qt.AlignmentFlag.AlignRight)
        layout_scroll_idx.addWidget(self.label_idx_scroll)
        layout_scroll_idx.addWidget(self.label_text_labeling, alignment=Qt.AlignmentFlag.AlignRight )
        layout_scroll_idx.addWidget(self.edit_text_labeling)
        layout_scroll_idx.addWidget(self.button_start_labeling)
        layout_scroll_idx.addWidget(self.edit_section_label)
        layout_scroll_idx.addWidget(self.button_end_labeling)


        self.label_clip_image = QLabel(self)

        # init widgets for perspective image
        self.m_pos_cnt = 0

        # init widgets for edge detection
        self.label_filter = QLabel("Filter type")
        self.button_edge_detection = QPushButton("Edge Detection")
        self._edgeType_combo_box = QComboBox()
        self._edgeType_combo_box.addItem("None")
        self._edgeType_combo_box.addItem("Sobel_XY")
        self._edgeType_combo_box.addItem("Scharr_X")
        self._edgeType_combo_box.addItem("Scharr_Y")
        self._edgeType_combo_box.addItem("Laplacian")
        self._edgeType_combo_box.addItem("Canny")

        # layout for edge detection
        edge_layout = QHBoxLayout()
        edge_layout.addWidget(self.label_filter, alignment=Qt.AlignmentFlag.AlignRight)
        edge_layout.addWidget(self._edgeType_combo_box)
        edge_layout.addWidget(self.button_edge_detection)

        # Load image buttons layout
        self.radiobutton_1 = QRadioButton("Image")
        self.radiobutton_2 = QRadioButton("Video")
        self.radiobutton_3 = QRadioButton("Webcam")
        self.radiobutton_1.setChecked(True)

        layout_loading_type = QHBoxLayout()
        layout_loading_type.addWidget(self.radiobutton_1)
        layout_loading_type.addWidget(self.radiobutton_2)
        layout_loading_type.addWidget(self.radiobutton_3)

        self.button_load_Img = QPushButton("Load Image")
        self.edit = QLineEdit("images/moving_dark.mp4")
        self.button_REC = QPushButton("REC")
        self.button_labeling = QPushButton("Labeling")

        self.button_load_Img.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.edit.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.button_REC.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.button_labeling.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.edit)
        bottom_layout.addLayout(layout_loading_type)
        bottom_layout.addWidget(self.button_load_Img)
        bottom_layout.addWidget(self.button_REC)
        bottom_layout.addWidget(self.button_labeling)

        layout_img_scroll = QVBoxLayout()
        layout_img_scroll.addWidget(self.label_image)
        layout_img_scroll.addWidget(self.scroll_bar)
        layout_img_scroll.addLayout(layout_scroll_idx)
        layout_img_scroll.addLayout(layout_save_training)


        # layout for image and graph
        layout_img_canvas = QHBoxLayout()
        layout_img_canvas.addLayout(layout_img_scroll)
        layout_img_canvas.addWidget(self.label_clip_image)

        # Main layout
        layout = QVBoxLayout()
        layout.addLayout(layout_img_canvas)
        layout.addLayout(bottom_layout)
        layout.addLayout(edge_layout)

        # Central widget
        widget = QWidget(self)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Connections
        self.button_load_Img.clicked.connect(self.load_img_func)
        self.button_REC.clicked.connect(self.recording)
        self.button_labeling.clicked.connect(self.labeling)
        self.button_edge_detection.clicked.connect(self.method_edge_detection)


        self.timer = QTimer(self)
        self.timer.timeout.connect(self.display_video_stream_v2)
    
    def dp_change(self):
        pyautogui.hotkey('command', 'tab')
        print("display changed")

    def training_perceptron(self):
        print("Training...")
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(3, activation='relu', input_shape=(1,)))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        # model.add(tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(1,)))
        # optimizer = tf.keras.optimizers.SGD(lr=0.001)
        # optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999)
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        path_split = self.edit.text().split('/')
        self.file_name = path_split[-1].split('.')
        train_y = np.load(f"{self.file_name[0]}.npy")
        print("train y len: ", len(train_y))
        print("train y size: ", train_y.size)
        train_x = np.array([[]])
        for i in range(len(self.frame_list)):
            frame = cv2.Canny(self.frame_list[i], 100, 150)

            ratio = 480 / frame.shape[0]
            img = cv2.resize(frame, None, fx=ratio, fy=ratio)

            temp = img > 0

            cnt_edge = temp.sum()
            train_x = np.append(train_x, cnt_edge)
        scaler = MinMaxScaler()
        scaler.fit(train_x)
        self.data_max = scaler.data_max_()
        self.data_min = scaler.data_min_()
        train_x = scaler.transform(train_x)
        train_x = train_x[:,np.newaxis]

        model.fit(train_x, train_y, epochs=2000, batch_size=50, shuffle=True)
        test_loss, test_acc = model.evaluate(train_x, train_y)

        print('테스트 정확도:', test_acc)

        model.save('model_ex6.h5')
        self.model = tf.keras.models.load_model('model_ex6.h5')

    def save_label(self):
        print("save label")
        save_Ground_truth = np.array(self.labeling_Ground_truth)
        path_split = self.edit.text().split('/')
        self.file_name = path_split[-1].split('.')
        np.save(f"./{self.file_name[0]}.npy", save_Ground_truth)

    def set_section_start(self):
        print("section start")
        self.start_idx = self.scroll_bar.sliderPosition()
        self.button_end_labeling.setEnabled(True)
        self.button_start_labeling.setEnabled(False)

    def set_section_end(self):
        print("section end")
        idx_i = self.start_idx
        idx_j = self.scroll_bar.sliderPosition()

        if idx_j > idx_i:
            label_value = [int(self.edit_section_label.text()) for i in range(idx_j-idx_i+1)]
            self.labeling_Ground_truth[idx_i:idx_j+1] = label_value
        elif idx_j < idx_i:
            label_value = [int(self.edit_section_label.text()) for i in range(idx_i - idx_j + 1)]
            self.labeling_Ground_truth[idx_j:idx_i+1] = label_value

        self.button_end_labeling.setEnabled(False)
        self.button_start_labeling.setEnabled(True)
    def set_label(self):

        if self.edit_text_labeling.text() == '0' or self.edit_text_labeling.text() == '1':
            self.labeling_Ground_truth[self.scroll_bar.sliderPosition()] = int(self.edit_text_labeling.text())
            print("changed label: ", self.labeling_Ground_truth[self.scroll_bar.sliderPosition()])

        else:
            print("Warning: input only [0 or 1]")

    def change_frame(self):
        cur_idx_of_scroller = self.scroll_bar.sliderPosition()
        # print("changed frame: ", cur_idx_of_scroller)
        self.update_image(self.frame_list[cur_idx_of_scroller])
        self.label_idx_scroll.setText(f"{cur_idx_of_scroller}")
        self.edit_text_labeling.setText(f"{self.labeling_Ground_truth[cur_idx_of_scroller]}")

    def labeling(self):
        print("labeling")
        self.kill_thread()
        self.labeling_capture = cv2.VideoCapture(self.edit.text())

        ret = True
        cnt = 0
        self.frame_list = []
        while ret:
            ret, frame = self.labeling_capture.read()

            if not ret:
                break
            cnt += 1
            self.frame_list.append(frame)
            self.labeling_Ground_truth.append(0)


        print("Loading video is complete")
        print(f"The number of frame: {len(self.frame_list)}  Ground-truth len: {len(self.labeling_Ground_truth)}" )
        self.scroll_bar.setMaximum(len(self.frame_list)-1)  # 최대값 설정
        self.scroll_bar.setVisible(True)

        # scroll의 index number QLabel과 Labeling의 QLineEdit을 초기값 설정
        self.label_idx_scroll.setText("0")
        self.edit_text_labeling.setText(f"{self.labeling_Ground_truth[0]}")

        self.labeling_capture.release()
        self.update_image(self.frame_list[0])

        self.edit_text_labeling.editingFinished.connect(self.set_label)
        self.button_start_labeling.clicked.connect(self.set_section_start)
        self.button_end_labeling.clicked.connect(self.set_section_end)
        self.button_save_label.clicked.connect(self.save_label)
        self.button_training.clicked.connect(self.training_perceptron)

    def recording(self):
        cap = cv2.VideoCapture(0)
        time.sleep(1)

        if not cap.isOpened():
            print("Camera open failed!")
            sys.exit()

        w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # *'DIVX' == 'D', 'I', 'V', 'X'
        delay = round(1000 / fps)

        out = cv2.VideoWriter('output_1.avi', fourcc, fps, (w, h))

        if not out.isOpened():
            print('File open failed!')
            cap.release()
            sys.exit()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            out.write(frame)
            self.update_image(frame)
            cv2.imshow('frame', frame)

            if cv2.waitKey(delay) == 27:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def closeEvent(self, event):
        self.kill_thread()

    def setup_camera(self, vid):
        self.capture = cv2.VideoCapture(vid)

        if not self.capture.isOpened():
            print("Camera open failed")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_size.width())
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_size.height())

        self.timer.start(50)

    def display_video_stream_v2(self):
        if self.qu_img_to_app.empty() is False:

            qu_val = self.qu_img_to_app.get_nowait()

            frame = qu_val[0]
            cnt_edge = qu_val[1]

            self.update_image2(frame)

            # print(self.qu_img_to_app.qsize())
            if self.EDGE_TYPE == 'Canny':
                if cnt_edge is not None:
                    self.update_plot2(cnt_edge)
                    print([cnt_edge])
                    predicted_y = self.model.predict(np.array([cnt_edge]))
                    print(predicted_y)
                    # print(type(cnt_edge))
                    # print(cnt_edge.shape)

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

    def method_edge_detection(self):
        if self.MODE_VIDEO is True:
            if self._edgeType_combo_box.currentText() == 'None':
                self.th_out.EDGE_TYPE = None
                return
            elif self._edgeType_combo_box.currentText() == 'Canny':
                self.th_out.EDGE_TYPE = 'Canny'
                self.EDGE_TYPE = 'Canny'
                return
            elif self._edgeType_combo_box.currentText() == 'Laplacian':
                self.th_out.EDGE_TYPE = 'Laplacian'
                self.EDGE_TYPE = 'Laplacian'
                return

        if self.m_proc_img is not None:
            if len(self.m_proc_img.shape) >= 3:
                self.m_proc_img = cv2.cvtColor(self.m_proc_img, cv2.COLOR_BGR2GRAY)

            if self._edgeType_combo_box.currentText() == 'Sobel_XY':
                print("Sobel_XY")
                sobel_img = cv2.Sobel(self.m_proc_img, cv2.CV_8U, 1, 1, ksize=3)
                self.update_image(sobel_img)
            elif self._edgeType_combo_box.currentText() == 'Scharr_X':
                print("Sobel_X")
                s_imageX = cv2.Scharr(self.m_proc_img, cv2.CV_8U, 1, 0)
                self.update_image(s_imageX)
            elif self._edgeType_combo_box.currentText() == 'Scharr_Y':
                print("Sobel_Y")
                s_imageY = cv2.Scharr(self.m_proc_img, cv2.CV_8U, 0, 1)
                self.update_image(s_imageY)
            elif self._edgeType_combo_box.currentText() == 'Laplacian':
                print("Laplacian")
                l_image = cv2.Laplacian(self.m_proc_img, cv2.CV_8U, ksize=3)
                self.update_image(l_image)
            elif self._edgeType_combo_box.currentText() == 'Canny':
                print("Canny")
                c_image1 = cv2.Canny(self.m_proc_img, 100, 150)
                self.update_image(c_image1)

    def update_image(self, img):
        # Creating and scaling QImage
        if len(img.shape) < 3:
            h, w = img.shape
            ch = 1
            img_format = QImage.Format_Grayscale8
        else:
            h, w, ch = img.shape
            img_format = QImage.Format_RGB888
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = QImage(img.data, w, h, ch * w, img_format)
        scaled_img = img.scaled(640, 480, Qt.KeepAspectRatio)
        self.label_image.setFixedSize(scaled_img.width(), scaled_img.height())
        self.label_image.setPixmap(QPixmap.fromImage(scaled_img))

    def load_img_func(self):
        self.kill_thread()
        if self.radiobutton_1.isChecked() is True:
            self.MODE_VIDEO = False
            self.m_main_img = cv2.imread(f"{self.edit.text()}", cv2.IMREAD_COLOR)
            self.m_main_img = cv2.resize(self.m_main_img, (640, 480), interpolation=cv2.INTER_CUBIC)
            self.m_proc_img = self.m_main_img.copy()
            self.update_image(self.m_proc_img)
            print("update image")
        elif self.radiobutton_2.isChecked() is True:
            path = self.edit.text()
            self.createThread_start(path)

        elif self.radiobutton_3.isChecked() is True:
            self.createThread_start(0)

    def setup_camera(self, vid):
        self.capture = cv2.VideoCapture(vid)
        if not self.capture.isOpened():
            print("Camera open failed")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_size.width())
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_size.height())

        self.timer.start(30)
    
    def update_image2(self, scaled_img):
        # Creating and scaling QImage
        self.label_image.setFixedSize(scaled_img.width(), scaled_img.height())
        self.label_image.setPixmap(QPixmap.fromImage(scaled_img))
    
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
        self.timer.start(15)
    
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

        if self.labeling_capture is not None:
            self.labeling_capture.release()

if __name__ == '__main__':
    app = QApplication()
    w = Window()
    w.show()
    sys.exit(app.exec())
    w.kill_thread()
