# -*- coding: utf-8 -*-
# @Time    : 2023/7/21 10:52
# @Name    : Driver.py
# @Author  : zyk
# @Introduce  : 驾驶员状态监控

import math

import cv2
import numpy as np
import dlib
from serial import Serial
from PyQt5.QtCore import *
from IMU import DueData


class Driver:
    def __init__(self, threshold=20):
        # 标记
        self.close_eye_counter = 0
        self.open_mouth_counter = 0
        self.smoke_counter = 0
        self.drink_counter = 0
        self.phone_counter = 0
        self.distracted_counter = 0

        # 阈值
        self.distracted_threshold = threshold
        self.close_eye_threshold = threshold
        self.open_mouth_threshold = threshold
        self.smoke_threshold = threshold
        self.phone_threshold = threshold
        self.drink_threshold = threshold

        self.num_dangerous = 0
        self.num_tired = 0

        # 人脸检测器
        # 加载人脸检测器
        self.detector = dlib.get_frontal_face_detector()
        # 加载关键点检测器
        self.predictor = dlib.shape_predictor("landmarks/shape_predictor_68_face_landmarks.dat")

    # 驾驶状态
    def get_state(self):
        if self.close_eye_counter > self.close_eye_threshold:
            print("您已疲劳！")
            self.close_eye_counter = 0
            self.num_tired += 1
            return "tired"

        if self.open_mouth_counter > self.open_mouth_threshold:
            print("您已疲劳！")
            self.open_mouth_counter = 0
            self.num_tired += 1
            return "tired"

        # if self.smoke_counter > self.smoke_threshold:
        #     print("驾驶时请勿抽烟！")
        #     self.smoke_counter = 0
        #     self.num_dangerous += 1
        #     return "smoke"

        if self.phone_counter > self.phone_threshold:
            print("驾驶时请勿玩手机！")
            self.phone_counter = 0
            self.num_dangerous += 1
            return "phone"

        if self.drink_counter > self.drink_threshold:
            print("驾驶时请勿饮水！")
            self.drink_counter = 0
            self.num_dangerous += 1
            return "drink"

        if self.distracted_counter > self.distracted_threshold:
            print("请专心驾驶")
            self.distracted_counter = 0
            self.num_dangerous += 1
            return "focus_drive"

        return "ok"

    # 标签

    def counter_labels(self, labels):
        self.close_eye_counter = self.close_eye_counter + 1 if 'closed_eye' in labels else 0
        self.open_mouth_counter = self.open_mouth_counter + 1 if 'open_mouth' in labels else 0
        self.smoke_counter = self.smoke_counter + 1 if 'smoke' in labels else 0
        self.phone_counter = self.phone_counter + 1 if 'phone' in labels else 0
        self.drink_counter = self.drink_counter + 1 if 'drink' in labels else 0


# 车辆状态监控子线程
class Worker(QThread):
    data_signal = pyqtSignal(int)

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def run(self):
        port = '/dev/ttyUSB0'  # USB serial port
        baud = 9600  # Same baud rate as the INERTIAL navigation module
        ser = Serial(port, baud, timeout=0.5)
        ser_data = ser.read(33)
        last_angel = DueData(ser_data)
        state = False
        while True:
            ser_data = ser.read(33)
            now_angle = DueData(ser_data)
            if now_angle is not None and last_angel is not None:
                for x, y in zip(now_angle, last_angel):
                    state = True if round(abs(x - y), 3) > self.threshold * 5 else False
                self.data_signal.emit(state)
            last_angel = now_angle
