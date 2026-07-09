import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

# 模型文件路径（相对于本模块所在目录）
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_MODEL_DIR, 'hand_landmarker.task')

# 手部关键点连线定义 (21个点的连接关系)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # 大拇指
    (0, 5), (5, 6), (6, 7), (7, 8),       # 食指
    (0, 9), (9, 10), (10, 11), (11, 12),   # 中指
    (0, 13), (13, 14), (14, 15), (15, 16), # 无名指
    (0, 17), (17, 18), (18, 19), (19, 20), # 小指
    (5, 9), (9, 13), (13, 17),             # 手指间横向连接
]

# 关键点绘制颜色 (BGR)
LANDMARK_COLOR = (0, 255, 0)
CONNECTION_COLOR = (0, 200, 200)


class HandDetector:
    '''
    手势识别类 — 基于 MediaPipe 0.10+ Tasks API
    '''
    def __init__(self, mode=False, max_hands=2, complexity=1,
                 detection_con=0.5, track_con=0.5):
        '''
        手势识别初始化
        :param mode: 是否为静态图片。默认为False(不是静态图片)
        :param max_hands: 最多检测几只手 默认为2
        :param complexity: 模型复杂度 默认为1
        :param detection_con: 最小检测置信度 默认为0.5
        :param track_con: 最小追踪置信度 默认为0.5
        '''
        self.mode = mode
        self.max_hands = max_hands
        self.complexity = complexity
        self.detection_con = detection_con
        self.track_con = track_con

        # 使用 MediaPipe 0.10+ Tasks API
        # 注意: MediaPipe 底层 C 库不支持中文路径，改用 model_asset_buffer 直接加载
        with open(_MODEL_PATH, 'rb') as f:
            model_data = f.read()
        base_options = python.BaseOptions(model_asset_buffer=model_data)
        running_mode = vision.RunningMode.IMAGE
        if not mode:
            running_mode = vision.RunningMode.VIDEO

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_con,
            min_hand_presence_confidence=detection_con,
            min_tracking_confidence=track_con,
            running_mode=running_mode,
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)

        # 存储结果和手部标签
        self.results = None
        self.hand_labels = []
        self._frame_timestamp = 0

    def find_hands(self, img):
        '''
        检测手势
        :param img: 视频帧图片 (BGR格式)
        :return: 处理过的视频帧图片 (BGR格式)
        '''
        # BGR -> RGB
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # 创建 MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)

        # 检测
        if self.mode:
            self.results = self.hand_landmarker.detect(mp_image)
        else:
            self._frame_timestamp += 1
            self.results = self.hand_landmarker.detect_for_video(mp_image, self._frame_timestamp)

        # 提取手部标签
        self.hand_labels = []
        if self.results.handedness:
            for handness in self.results.handedness:
                if handness:
                    label = handness[0].category_name  # "Left" or "Right"
                    score = handness[0].score
                    self.hand_labels.append((label, score))

        # 绘制手部关键点和连线
        img = self._draw_landmarks(img)

        return img

    def _draw_landmarks(self, img):
        """在图像上绘制手部关键点和连线"""
        if not self.results or not self.results.hand_landmarks:
            return img

        h, w, _ = img.shape
        overlay = img.copy()

        for hand_lms in self.results.hand_landmarks:
            # 计算像素坐标
            points = []
            for lm in hand_lms:
                px, py = int(lm.x * w), int(lm.y * h)
                points.append((px, py))

            # 绘制连线
            for (i, j) in HAND_CONNECTIONS:
                cv.line(overlay, points[i], points[j], CONNECTION_COLOR, 2)

            # 绘制关键点
            for i, (px, py) in enumerate(points):
                radius = 5 if i in [4, 8, 12, 16, 20] else 3  # 指尖稍大
                color = (0, 255, 255) if i in [4, 8, 12, 16, 20] else LANDMARK_COLOR
                cv.circle(overlay, (px, py), radius, color, -1)

        # 半透明叠加
        cv.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        return img

    def find_positions(self, img, hand_no=0):
        '''
        获取手势数据
        :param img: 视频帧图片
        :param hand_no: 手编号（默认第1只手）
        :return: 关键点列表，每个元素为 [id, cx, cy]
        '''
        self.lmslist = []
        if self.results and self.results.hand_landmarks:
            if hand_no < len(self.results.hand_landmarks):
                hand = self.results.hand_landmarks[hand_no]
                h, w, _ = img.shape
                for id, lm in enumerate(hand):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lmslist.append([id, cx, cy])
                return self.lmslist

        return self.lmslist

    def close(self):
        """释放资源"""
        self.hand_landmarker.close()
