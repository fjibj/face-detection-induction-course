#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
from time import sleep
import sys
import cv2
import numpy as np
from PIL import Image
from imutils import face_utils, resize
from threading import Thread
from multiprocessing import Pool

try:
    from dlib import get_frontal_face_detector, shape_predictor
except ImportError:
    raise

def async(f):
    def wrapper(*args, **kwargs):
        thr = Thread(target = f, args = args, kwargs = kwargs)
        thr.start()
    return wrapper


class DynamicStreamMaskService(object):
    """
    动态黏贴面具服务
    """

    def __init__(self, video, saved=False):
        self.saved = saved  # 是否保存图片
        self.dangering = False # 是否正在危险警告
        self.listener = True  # 启动参数
        self.video_capture = cv2.VideoCapture(video)
        self.doing = False  # 是否进行面部面具
        self.detector = get_frontal_face_detector()  # 面部识别器
        self.predictor = shape_predictor("shape_predictor_68_face_landmarks.dat")  # 面部分析器
        self.max_width = 500  # 图像大小
        self.max_height = 500
        self.deal, self.text, self.cigarette = None, None, None  # 面具对象

    def read_data(self):
        """
        从摄像头获取视频流，并转换为一帧一帧的图像
        :return: 返回一帧一帧的图像信息
        """
        _, data = self.video_capture.read()
        return data
    
    @async
    def danger_beep(self):
        self.dangering = True
        os.system('say "危险！危险！"')
        self.dangering = False

    def save_data(self, draw_img):
        """
        保存图片到本地
        :param draw_img:
        :return:
        """
        if not self.saved:
            return
        self.console("保存证据...")
        draw_img.save("images/%s.png" % time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())))
        self.saved = False

    def init_mask(self):
        """
        加载面具
        :return:
        """
        self.console("加载面具...")
        self.deal, self.text, self.cigarette = (
            Image.open(x) for x in ["images/deals.png", "images/text.png", "images/cigarette.png"]
        )

    def get_glasses_info(self, face_shape, face_width):
        """
        获取当前面部的眼镜信息
        :param face_shape:
        :param face_width:
        :return:
        """
        left_eye = face_shape[36:42]
        right_eye = face_shape[42:48]

        left_eye_center = left_eye.mean(axis=0).astype("int")
        right_eye_center = right_eye.mean(axis=0).astype("int")

        y = left_eye_center[1] - right_eye_center[1]
        x = left_eye_center[0] - right_eye_center[0]
        eye_angle = np.rad2deg(np.arctan2(y, x))

        deal = self.deal.resize(
            (face_width, int(face_width * self.deal.size[1] / self.deal.size[0])),
            resample=Image.LANCZOS)

        deal = deal.rotate(eye_angle, expand=True)
        deal = deal.transpose(Image.FLIP_TOP_BOTTOM)

        left_eye_x = left_eye[0, 0] - face_width // 4
        left_eye_y = left_eye[0, 1] - face_width // 6

        return {"image": deal, "pos": (left_eye_x, left_eye_y)}

    def get_cigarette_info(self, face_shape, face_width):
        """
        获取当前面部的烟卷信息
        :param face_shape:
        :param face_width:
        :return:
        """
        mouth = face_shape[49:68]
        mouth_center = mouth.mean(axis=0).astype("int")
        cigarette = self.cigarette.resize(
            (face_width, int(face_width * self.cigarette.size[1] / self.cigarette.size[0])),
            resample=Image.LANCZOS)
        x = mouth[0, 0] - face_width + int(16 * face_width / self.cigarette.size[0])
        y = mouth_center[1]
        return {"image": cigarette, "pos": (x, y)}

    def orientation(self, rects, img_gray):
        """
        人脸定位
        :return:
        """
        faces = []
        for rect in rects:
            face = {}
            face_shades_width = rect.right() - rect.left()
            predictor_shape = self.predictor(img_gray, rect)
            face_shape = face_utils.shape_to_np(predictor_shape)
            face['cigarette'] = self.get_cigarette_info(face_shape, face_shades_width)
            face['glasses'] = self.get_glasses_info(face_shape, face_shades_width)

            faces.append(face)
        
        return faces

    def start(self):
        """
        启动程序
        :return:
        """
        #p = Pool()
        self.console("程序启动成功.")
        self.init_mask()
        while self.listener:
            frame = self.read_data()
            frame = resize(frame, width=self.max_width, height=self.max_height)
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.detector(img_gray, 0)
            faces = self.orientation(rects, img_gray)
            draw_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.drawing(draw_img, faces)
            self.save_data(draw_img)
            frame = cv2.cvtColor(np.asarray(draw_img), cv2.COLOR_RGB2BGR)
            cv2.imshow("hello mask", frame)
            #p.map(self.danger_beep,faces)
            if not self.dangering and len(faces)>0:
                self.danger_beep()
            self.listener_keys()

    def listener_keys(self):
        """
        设置键盘监听事件
        :return:
        """
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.listener = False
            self.console("程序退出")
            sleep(1)
            self.exit()

        if key == ord("s"):    #按S键保存画面
            self.saved = not self.saved

    def exit(self):
        """
        程序退出
        :return:
        """
        self.video_capture.release()
        cv2.destroyAllWindows()

    def drawing(self, draw_img, faces):
        """
        画图
        :param draw_img:
        :param faces:
        :return:
        """
        for face in faces:
            draw_img.paste(face["glasses"]["image"], face["glasses"]["pos"], face["glasses"]["image"])
            draw_img.paste(face["cigarette"]["image"], face["cigarette"]["pos"], face["cigarette"]["image"])
            draw_img.paste(self.text, (20, draw_img.height // 2 ), self.text)


    @classmethod
    def console(cls, s):
        print("{} !".format(s))

def main(argv):
    if len(argv)<2:
        video = 0
    else:
        video = argv[1]
    print(video)
    ms = DynamicStreamMaskService(video)
    ms.start()

if __name__ == '__main__':
    main(sys.argv)
