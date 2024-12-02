import threading
import time
import pyautogui
import queue
from PySide6.QtCore import QThread, Signal, QTimer
import cv2

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
                print("qu size: ", self.qu.qsize())

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

        self.m_proc_img = None
        self.MODE_VIDEO = False
        self.EDGE_TYPE = None
        self.previous_plot = None
        self.cnt = 0

        self.th_in = None
        self.th_out = None


        self.y_max = 0
        self.canvas = FigureCanvas(Figure(figsize=(0, 0.5)))
        self.axes = self.canvas.figure.subplots()
        # self.axes.set_ylim([0,5000])

        n_data = 50
        self.xdata = list(range(n_data))
        self.axes.set_xticks(self.xdata, [])
        self.ydata = [0 for i in range(n_data)]

    def update_plot2(self, sum_value):
        self.ydata = self.ydata[1:] + [sum_value]
        if sum_value > self.y_max:
            self.y_max = sum_value
            self.axes.set_ylim([0, self.y_max + 10])

        if self.previous_plot is None:
            self.previous_plot = self.axes.plot(self.xdata, self.ydata, 'r')[0]
        else:
            self.previous_plot.set_ydata(self.ydata)

        # Trigger the canvas to update and redraw.
        global Processing_stop
        Processing_stop = True
        sema1.acquire()
        sema2.acquire()

        prevTime = time.time()  # 현재 시간
        self.canvas.draw()
        Processing_stop = False
        sema0_1.release()
        sema0_2.release()
        curTime = time.time()  # 현재 시간
        sec = curTime - prevTime
        print(sec%60)

    def display_video_stream(self):
        if self.qu_img_to_app.empty() is False:
            qu_val = self.qu_img_to_app.get_nowait()
            frame = qu_val[0]
            cnt_edge = qu_val[1]
            self.update_image2(frame)

            if self.EDGE_TYPE == 'Canny':
                if cnt_edge is not None:
                    self.update_plot2(cnt_edge)

    def update_image2(self, scaled_img):
        # Creating and scaling QImage
        self.label_image.setFixedSize(scaled_img.width(), scaled_img.height())
        self.label_image.setPixmap(QPixmap.fromImage(scaled_img))

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
                c_image1 = cv2.Canny(self.m_proc_img, 150, 300)
                self.update_image(c_image1)

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
