# 功能基本完善
# 1、声音提醒，可调节声音大小
# 2、车辆状态检测，可调节时间间隔、姿态阈值
# 3、灵敏度调节 10-50
#当使用的训练模型不是同一个系统的时候，需要添加
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import os
import sys
import time
from pathlib import Path
import cv2
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import qdarkstyle
# import dlib
# import math

from Sender import Sample
from Driver import Driver, Worker

import shutil

from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt5.QtCore import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtCore import QUrl

from utils.general import check_img_size, non_max_suppression, scale_boxes, increment_path
from utils.augmentations import letterbox
from utils.plots import plot_one_box
from models.common import DetectMultiBackend

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
audio_path = ['drink', 'focus', 'phone', 'smoke', 'tired']


class Ui_MainWindow(QtWidgets.QMainWindow):
    ser = pyqtSignal(object)

    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)

        # 参数初始化
        # 驾驶员状态初始化
        self.driver = Driver(20)
        self.num_dangerous = 0
        self.num_tired = 0

        # 测试标识
        self.test_flag = False

        # 检测延时
        self.delay_time = 0.0
        self.start_times = {}
        # 声音播放初始化
        self.player = QMediaPlayer()
        self.player.setVolume(80)
        self.player_audio = [QMediaContent(QUrl.fromLocalFile(QFileInfo(f'audio/{path}.mp3').absoluteFilePath()))
                             for path in audio_path]

        # 短信发送
        # 15517327613
        self.phone_num = "15517327613"
        self.sender = Sample()

        # ui初始化
        self.timer_video = QtCore.QTimer()
        self.setupUi(self)
        self.init_logo()
        self.init_slots()

        # 检测模型初始化
        self.cap = cv2.VideoCapture()
        self.out = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.half = False
        name = 'exp'
        self.save_root = ROOT / 'result'
        self.save_file = increment_path(Path(self.save_root) / name, exist_ok=False, mkdir=True)
        cudnn.benchmark = True
        weights = 'weights/car.pt'  # 模型加载路径2
        imgsz = 640  # 预测图尺寸大小
        self.conf_thres = 0.25  # NMS置信度
        self.iou_thres = 0.45  # IOU阈值

        # 载入模型
        self.model = DetectMultiBackend(weights, device=self.device)
        stride = self.model.stride
        self.imgsz = check_img_size(imgsz, s=stride)
        if self.half:
            self.model.half()  # to FP16

        # 从模型中获取各类别名称
        self.names = self.model.names
        # 给每一个类别初始化颜色
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

    def font(self) -> QtGui.QFont:
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(12)
        return font

    def set_button_style(self, button, font, name, tips):
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(button.sizePolicy().hasHeightForWidth())
        button.setSizePolicy(sizePolicy)
        button.setMinimumSize(QtCore.QSize(150, 40))
        button.setMaximumSize(QtCore.QSize(150, 40))
        button.setStyleSheet(
            "QPushButton{background-color: rgb(111, 180, 219)}"  # 按键背景色
            "QPushButton{color: rgb(93, 109, 126); font-weight: bold}"  # 字体颜色形状
            "QPushButton{border-radius: 6px}"  # 圆角半径
            "QPushButton:hover{color: rgb(39, 55, 70); font-weight: bold;}"  # 光标移动到上面后的字体颜色形状
            "QPushButton:pressed{background-color: rgb(23, 32, 42); font-weight: bold; color: rgb(135, 54, 0)}"
            # 按下时的样式
        )
        button.setFont(font)
        button.setObjectName(name)
        self.verticalLayout.addWidget(button, 0, QtCore.Qt.AlignHCenter)
        self.verticalLayout.addStretch(3)  # 增加垂直盒子内部对象间距
        button.setToolTip(tips)  # 创建提示框

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(900, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setStyleSheet("#centralwidget{border-image:url(./UI/carui.jpg)}")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)  # 布局的左、上、右、下到窗体边缘的距离
        # self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")

        font = self.font()

        # 打开单图片按钮
        self.pushButton_img = QtWidgets.QPushButton(self.centralwidget)
        self.set_button_style(self.pushButton_img, font, 'pushButton_img',
                              '<b>请选择一张图片进行检测</b>')

        # 打开视频按钮
        self.pushButton_video = QtWidgets.QPushButton(self.centralwidget)
        self.set_button_style(self.pushButton_video, font, 'pushButton_video',
                              '<b>请选择一个视频进行检测</b>')

        # 打开摄像头按钮
        self.pushButton_camera = QtWidgets.QPushButton(self.centralwidget)
        self.set_button_style(self.pushButton_camera, font, 'pushButton_camera',
                              '<b>请确保摄像头设备正常</b>')

        # 发送文件按钮
        self.pushButton_send_video = QtWidgets.QPushButton(self.centralwidget)
        self.set_button_style(self.pushButton_send_video, font, 'pushButton_send_video',
                              '<b>请选择需要的发送视频文件</b>')

        # 模型更新按钮
        self.pushButton_update_model = QtWidgets.QPushButton(self.centralwidget)
        self.set_button_style(self.pushButton_update_model, font, 'pushButton_update_model',
                              '<b>请选择需要更新的模型</b>')

        # 测试按钮
        self.pushButton_test = QtWidgets.QPushButton(self.centralwidget)
        self.set_button_style(self.pushButton_test, font, 'pushButton_test',
                              '<b>进行移动与停车测试</b>')

        # 检测设备状态
        self.dangerous_label = QtWidgets.QLabel(self)
        self.dangerous_label.setText(f"违停车辆车牌: {self.num_dangerous}")
        self.verticalLayout.addWidget(self.dangerous_label, 0, QtCore.Qt.AlignHCenter)

        # 驾驶状态
        self.tired_label = QtWidgets.QLabel(self)
        self.tired_label.setText(f"违停地点: {self.num_tired}")
        self.verticalLayout.addWidget(self.tired_label, 0, QtCore.Qt.AlignHCenter)

        # 检测设备状态
        self.car_status = QtWidgets.QLabel(self)
        self.car_status.setText("车辆状态检测: 停止")
        self.verticalLayout.addWidget(self.car_status, 0, QtCore.Qt.AlignHCenter)

        # 检测延时状态
        self.delay_time_lable = QtWidgets.QLabel(self)
        self.delay_time_lable.setText(f"违停时间: {self.delay_time} ms")
        self.verticalLayout.addWidget(self.delay_time_lable, 0, QtCore.Qt.AlignHCenter)

        # 阈值调节滑动条
        self.title = QtWidgets.QLabel(self)
        self.title.setText("区域调整：")
        self.verticalLayout.addWidget(self.title, 0, QtCore.Qt.AlignHCenter)
        self.s = QtWidgets.QSlider(Qt.Horizontal)  # 水平方向
        self.s.setMinimum(1)  # 设置最小值
        self.s.setMaximum(40)  # 设置最大值
        self.s.setSingleStep(5)  # 设置步长值
        self.s.setValue(20)  # 设置当前值
        self.s.setTickPosition(QtWidgets.QSlider.TicksBelow)  # 设置刻度位置，在下方
        self.s.setTickInterval(5)  # 设置刻度间隔
        self.verticalLayout.addWidget(self.s, 0, QtCore.Qt.AlignHCenter)

        # 右侧图片/视频填充区域
        self.verticalLayout.setStretch(2, 1)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 3)
        self.horizontalLayout_2.addLayout(self.horizontalLayout)
        self.label.setStyleSheet("border: 1px solid white;")  # 添加显示区域边框

        # 底部美化导航条
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "EAI LAB 违停检测v1.0"))
        self.pushButton_img.setText(_translate("MainWindow", "单图片检测"))
        self.pushButton_video.setText(_translate("MainWindow", "视频检测"))
        self.pushButton_camera.setText(_translate("MainWindow", "摄像头检测"))
        self.pushButton_send_video.setText(_translate("MainWindow", "上传视频文件"))
        self.pushButton_update_model.setText(_translate("MainWindow", "更新模型"))
        self.pushButton_test.setText(_translate("MainWindow", "测试"))
        self.label.setText(_translate("MainWindow", "TextLabel"))

    # 初始化
    def init_slots(self):
        self.pushButton_img.clicked.connect(self.button_image_open)
        self.pushButton_video.clicked.connect(self.button_video_open)
        self.pushButton_camera.clicked.connect(self.button_camera_open)
        self.pushButton_send_video.clicked.connect(self.button_send_video)
        self.pushButton_update_model.clicked.connect(self.button_update_model)
        self.pushButton_test.clicked.connect(self.button_test)
        self.timer_video.timeout.connect(self.show_video_frame)
        self.s.valueChanged.connect(self.valueChange)

    def init_logo(self):
        pix = QtGui.QPixmap('')  # 绘制初始化图片
        self.label.setScaledContents(True)
        self.label.setPixmap(pix)

    # 退出提示
    def closeEvent(self, event):
        reply = QtWidgets.QMessageBox.question(self, 'Message',
                                               "Are you sure to quit?", QtWidgets.QMessageBox.Yes |
                                               QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            # 保留最新的两个结果
            res = self.keep_latest_folder()
            if res:
                event.accept()
        else:
            event.ignore()

    # 按钮事件
    def button_image_open(self):
        print('打开图片')
        name_list = []

        img_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        if not img_name:
            self.empty_information()
            print('empty!')
            return
        img = cv2.imread(img_name)
        print(img_name)
        showimg = img
        with torch.no_grad():
            img = letterbox(img, new_shape=self.imgsz)[0]
            # Convert
            # BGR to RGB, to 3x416x416
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            pred = self.model(img)[0]
            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
            # Process detections
            for i, det in enumerate(pred):
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(
                        img.shape[2:], det[:, :4], showimg.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        label = '%s %.2f' % (self.names[int(cls)], conf)
                        # print(label.split()[0])  # 打印各目标名称
                        name_list.append(self.names[int(cls)])
                        plot_one_box(xyxy, showimg, label=label,
                                     color=self.colors[int(cls)], line_thickness=2)

        cv2.imwrite(str(Path(self.save_file / 'prediction.jpg')), showimg)
        self.result = cv2.cvtColor(showimg, cv2.COLOR_BGR2BGRA)
        self.result = cv2.resize(self.result, (640, 480), interpolation=cv2.INTER_AREA)
        self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                  QtGui.QImage.Format_RGB32)
        self.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
        print('单图片检测完成')

    def button_images_open(self):
        print('打开图片')

        img_names, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        if len(img_names) == 0:
            self.empty_information()
            print('empty!')
            return
        index = 0
        for img_name in img_names:
            name_list = []
            img = cv2.imread(img_name)
            print(img_name)
            showimg = img
            with torch.no_grad():
                img = letterbox(img, new_shape=self.imgsz)[0]
                # Convert
                # BGR to RGB, to 3x416x416
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Inference
                pred = self.model(img)[0]
                # Apply NMS
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
                # Process detections
                for i, det in enumerate(pred):
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(
                            img.shape[2:], det[:, :4], showimg.shape).round()

                        for *xyxy, conf, cls in reversed(det):
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            # print(label.split()[0])  # 打印各目标名称
                            name_list.append(self.names[int(cls)])
                            plot_one_box(xyxy, showimg, label=label,
                                         color=self.colors[int(cls)], line_thickness=2)

            cv2.imwrite(str(Path(self.save_file / 'prediction_imgs{}.jpg'.format(index))), showimg)
            self.result = cv2.cvtColor(showimg, cv2.COLOR_BGR2BGRA)
            self.result = cv2.resize(self.result, (640, 480), interpolation=cv2.INTER_AREA)
            self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                      QtGui.QImage.Format_RGB32)
            self.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
            index += 1
        print('多图片检测完成')

    def button_imagefile_open(self):
        print('打开图片文件夹')

        file_name = QtWidgets.QFileDialog.getExistingDirectory(
            self, "打开图片文件夹", "")
        if not file_name:
            self.empty_information()
            print('empty!')
            return
        print(file_name)
        img_names = os.listdir(file_name)
        if len(img_names) == 0:
            self.empty_information()
            print('empty!')
            return
        index = 0
        for img_name in img_names:
            if img_name.split('.')[-1] not in ('jpg', 'png', 'jpeg'):
                continue
            name_list = []
            img = cv2.imread(os.path.join(file_name, img_name))
            print(img_name)
            showimg = img
            with torch.no_grad():
                img = letterbox(img, new_shape=self.imgsz)[0]
                # Convert
                # BGR to RGB, to 3x416x416
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Inference
                pred = self.model(img)[0]
                # Apply NMS
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
                # Process detections
                for i, det in enumerate(pred):
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(
                            img.shape[2:], det[:, :4], showimg.shape).round()

                        for *xyxy, conf, cls in reversed(det):
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            # print(label.split()[0])  # 打印各目标名称
                            name_list.append(self.names[int(cls)])
                            plot_one_box(xyxy, showimg, label=label,
                                         color=self.colors[int(cls)], line_thickness=2)

            cv2.imwrite(str(Path(self.save_file / 'prediction_file{}.jpg'.format(index))), showimg)
            self.result = cv2.cvtColor(showimg, cv2.COLOR_BGR2BGRA)
            self.result = cv2.resize(self.result, (640, 480), interpolation=cv2.INTER_AREA)
            self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                      QtGui.QImage.Format_RGB32)
            self.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
            index += 1
        print('文件夹图片检测完成')

    def button_video_open(self):
        video_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开视频", "", "*.mp4;;*.avi;;All Files(*)")

        if not video_name:
            self.empty_information()
            print('empty!')
            return

        flag = self.cap.open(video_name)
        if flag == False:
            QtWidgets.QMessageBox.warning(
                self, u"Warning", u"打开视频失败", buttons=QtWidgets.QMessageBox.Ok, defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            self.out = cv2.VideoWriter(str(Path(self.save_file / 'vedio_prediction.avi')), cv2.VideoWriter_fourcc(
                *'MJPG'), 20, (int(self.cap.get(3)), int(self.cap.get(4))))
            self.timer_video.start(30)
            self.pushButton_video.setDisabled(True)
            self.pushButton_img.setDisabled(True)
            self.pushButton_camera.setDisabled(True)

    def button_camera_open(self):
        if not self.timer_video.isActive():
            # 默认使用第一个本地camera
            flag = self.cap.open(0)
            # 2023.7.14 试试第二个
            # flag = self.cap.open(1)
            if flag == False:
                QtWidgets.QMessageBox.warning(
                    self, u"Warning", u"打开摄像头失败", buttons=QtWidgets.QMessageBox.Ok,
                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.out = cv2.VideoWriter(str(Path(self.save_file / 'camera_prediction.avi')), cv2.VideoWriter_fourcc(
                    *'MJPG'), 20, (int(self.cap.get(3)), int(self.cap.get(4))))
                self.timer_video.start(30)
                self.pushButton_video.setDisabled(True)
                self.pushButton_img.setDisabled(True)
                self.pushButton_camera.setText(u"关闭摄像头")
        else:
            self.timer_video.stop()
            self.cap.release()
            self.out.release()
            self.label.clear()
            self.init_logo()
            self.pushButton_video.setDisabled(False)
            self.pushButton_img.setDisabled(False)
            self.pushButton_camera.setText(u"摄像头检测")

    def show_video_frame(self):
        name_list = []
        flag, img = self.cap.read()
        if img is not None:
            showimg = img

            with torch.no_grad():
                img = letterbox(img, new_shape=self.imgsz)[0]
                # Convert
                # BGR to RGB, to 3x416x416
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Inference
                pred = self.model(img)[0]

                # Apply NMS
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

                # 改
                # -----进入循环：ESC退出-----
                num_boxes_set = set()  # 用集合来存储已经记录过开始时间的 num_boxes
                num_boxes = 0
                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if det is not None and len(det):
                        for j, (*xyxy, conf, cls) in enumerate(reversed(det)):
                            num_boxes += 1  # 计算检测到的框的数量
                            num_boxes_set.add(num_boxes)
                            if num_boxes not in self.start_times:
                                self.start_times[num_boxes] = time.time()  # 记录开始时间
                            elapsed_time = time.time() - self.start_times[num_boxes]  # 计算已过时间
                            minutes = int(elapsed_time // 60)
                            seconds = int(elapsed_time % 60)
                            timer_text = f"Time: {minutes:02d}:{seconds:02d}"

                            # Rescale boxes from img_size to im0 size
                            xyxy = [int(xy) for xy in xyxy]  # 将坐标转换为整数
                            #print(xyxy)
                            det[j, :4] = scale_boxes(
                                img.shape[2:], det[j, :4], showimg.shape).round()
                            xyxy_scaled = [int(xy) for xy in det[j, :4]]  # 将缩放后的坐标转换
                            if elapsed_time <= 5:
                                plot_one_box(xyxy_scaled, showimg, label=timer_text, color=(0, 128, 0),
                                             line_thickness=2)
                                print(timer_text)
                            else:
                                plot_one_box(xyxy_scaled, showimg, label="Stopped illegally " + timer_text,
                                             color=(0, 0, 128), line_thickness=2)


            self.out.write(showimg)
            show = cv2.resize(showimg, (640, 480))
            self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                     QtGui.QImage.Format_RGB888)
            self.label.setPixmap(QtGui.QPixmap.fromImage(showImage))

        else:
            self.timer_video.stop()
            self.cap.release()
            self.out.release()
            self.label.clear()
            self.pushButton_video.setDisabled(False)
            self.pushButton_img.setDisabled(False)
            self.pushButton_camera.setDisabled(False)
            self.init_logo()

    def button_send_video(self):
        print("上传视频文件")
        img_names, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "打开视频文件", "", "All Files(*)")
        if len(img_names) == 0:
            self.empty_information()
            # print('empty!')
            return

    def button_update_model(self):
        print("模型更新")
        weight_path, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "选择模型文件", "", "All Files(*)")
        if len(weight_path) == 0:
            self.empty_information()
            print('empty!')
            return

        filename = os.path.basename(weight_path[0])
        # 判断文件名是否以 ".pt" 或 ".pth" 结尾
        if not (filename.endswith('.pt') or filename.endswith('.pth')):
            self.msg_information("提示", "模型文件类型错误。请选择pt或pth格式文件。", QtWidgets.QMessageBox.Ok)
            print('选错文件了!')
            return
        else:
            self.model = None
            # 载入模型
            imgsz = 640  # 预测图尺寸大小
            self.model = DetectMultiBackend(weight_path[0], device=self.device)
            stride = self.model.stride
            self.imgsz = check_img_size(imgsz, s=stride)
            if self.half:
                self.model.half()  # to FP16
            self.msg_information("更换模型", "模型加载完毕", QtWidgets.QMessageBox.Ok)
            return

    def button_test(self):
        if self.test_flag:
            self.test_flag = False
        else:
            self.test_flag = True

        # 恢复
        self.num_tired = 0
        self.num_dangerous = 0
        self.delay_time = 0.0
        self.dangerous_label.setText("违停车辆车牌: **")#危险驾驶次数
        self.tired_label.setText("违停地点: 0")#疲劳驾驶次数
        self.delay_time_lable.setText(f"违停时长: 0.0 ms")#检测延时:
        
    # 提示
    def empty_information(self):
        QtWidgets.QMessageBox.information(self, '提示', '未选择文件或选择文件为空!', QtWidgets.QMessageBox.Cancel)

    def msg_information(self, title, msg, box_type):
        QtWidgets.QMessageBox.information(self, title, msg, box_type)

    # 检测算法
    # def detect(self, img):
    #     name_list = []
    #     showimg = img
    #     with torch.no_grad():
    #         img = letterbox(img, new_shape=self.imgsz)[0]
    #         # Convertpy
    #         # BGR to RGB, to 3x416x416
    #         img = img[:, :, ::-1].transpose(2, 0, 1)
    #         img = np.ascontiguousarray(img)
    #         img = torch.from_numpy(img).to(self.device)
    #         img = img.half() if self.half else img.float()  # uint8 to fp16/32
    #         img /= 255.0  # 0 - 255 to 0.0 - 1.0
    #         if img.ndimension() == 3:
    #             img = img.unsqueeze(0)
    #         # Inference
    #         pred = self.model(img)[0]
    #         # Apply NMS
    #         pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
    #         # Process detections
    #         for i, det in enumerate(pred):
    #             if det is not None and len(det):
    #                 # Rescale boxes from img_size to im0 size
    #                 det[:, :4] = scale_boxes(
    #                     img.shape[2:], det[:, :4], showimg.shape).round()
    #
    #                 for *xyxy, conf, cls in reversed(det):
    #                     label = '%s %.2f' % (self.names[int(cls)], conf)
    #                     # print(label.split()[0])  # 打印各目标名称
    #                     name_list.append(self.names[int(cls)])
    #                     plot_one_box(xyxy, showimg, label=label,
    #                                  color=self.colors[int(cls)], line_thickness=2)
    #     return showimg, name_list

    # 监控子线程
    # def start_worker(self):
    #     self.worker = Worker(1)
    #     self.worker.data_signal.connect(self.update_data)
    #     self.worker.start()

    # 标签数据更新
    def update_data(self, state):
        if self.test_flag:
            self.car_status.setText(f"车辆状态检测: {'运动' if not state else '停止'}")
        else:
            self.car_status.setText(f"车辆状态检测: {'运动' if state else '停止'}")

    # 更换驾驶员状态判定阈值更新
    def valueChange(self):
        self.driver.distracted_threshold = self.s.value()
        self.driver.close_eye_threshold = self.s.value()
        self.driver.open_mouth_threshold = self.s.value()
        self.driver.smoke_threshold = self.s.value()
        self.driver.phone_threshold = self.s.value()
        self.driver.drink_threshold = self.s.value()

    # 保留文件
    def keep_latest_folder(self, num_folders=2):
        """
        该函数将删除指定文件夹中旧的文件，只保留最新的num_folders个文件夹。
        如果文件夹数量少于num_folders，则保留所有文件夹。
        """
        # 获取文件夹中所有文件的列表
        files = os.listdir(self.save_root)

        if len(files) <= num_folders:
            return True

        # 按照修改时间排序文件列表（最新的文件在前面）
        files.sort(key=lambda x: os.path.getmtime(os.path.join(self.save_root, x)), reverse=True)

        # 保留最新的num_folders个文件夹，删除其余的文件夹
        for i in range(len(files) - num_folders, len(files)):
            shutil.rmtree(os.path.join(self.save_root, files[i]))  # 删除文件夹
        return True


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    ui = Ui_MainWindow()
    # 设置窗口透明度
    ui.setWindowOpacity(1)
    # 设置窗口图标
    icon = QIcon()
    icon.addPixmap(QPixmap("./UI/icon.ico"), QIcon.Normal, QIcon.Off)
    # 设置应用图标
    ui.setWindowIcon(icon)
    ui.show()
#    ui.start_worker()
    # 默认打开摄像头检测
    #ui.button_camera_open()
    sys.exit(app.exec_())
