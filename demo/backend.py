import cv2 as cv
import numpy as np
import os
import json
import time
from datetime import datetime
from handutil import HandDetector
from RK_face import (
    capture_faces, train_model, names_mapping_path, put_chinese_text,
    dataset_path, gesture_password_path, attendance_records_path,
    model_file, init_data_environment
)


class TerminalBackend:
    def __init__(self):
        init_data_environment()
        self.win_name = "人脸考勤系统"
        self.cap = None
        self.detector = HandDetector(detection_con=0.7)
        self.tip_ids = [4, 8, 12, 16, 20]
        self.confirm_frames = 20
        self.current_gesture_count = 0
        self.gesture_hold_timer = 0
        self.is_busy = False

        # ========== 手势防抖配置与状态 ==========
        self.gesture_stable_frames = 2    # 连续N帧相同才认定为稳定手势
        self.gesture_enter_delay = 2      # 手势从无到有时，额外延迟N帧再开始计时
        self.gesture_fault_tolerance = 1  # 允许连续N帧跳变不重置保持计时
        self._last_raw_finger = 0
        self._same_raw_count = 0
        self.stable_finger_count = 0
        self._stable_enter_counter = 0
        self._fault_remain = 0

        # ========== 分层反馈系统 ==========
        self.FEEDBACK_SUCCESS = "success"
        self.FEEDBACK_WARNING = "warning"
        self.FEEDBACK_ERROR = "error"
        self.FEEDBACK_INFO = "info"
        self.feedback_stack = []
        self.feedback_colors = {
            "success": ((46, 204, 113), (255, 255, 255)),
            "warning": ((241, 196, 15), (0, 0, 0)),
            "error": ((231, 76, 60), (255, 255, 255)),
            "info": ((0, 0, 0), (189, 195, 199))
        }

        self.face_cascade = None
        self.recognizer = None
        self.model_ready = False  # 模型是否已训练并就绪
        self.id_name_map = {}
        self._init_face_model()
        self.captured_faces = []

        self.attendance_mode = False
        self.attendance_file = attendance_records_path
        self.today_attendance = {}
        self.last_sign_timestamp = {}
        self.sign_cooldown = 30
        self._load_attendance()

        self.gesture_passwords = {}
        self._load_gesture_passwords()
        self.verify_state = 'idle'
        self.verify_target_user = None
        self.verify_target_gesture = None
        self.verify_frame_counter = 0
        self.verify_timeout = 300
        self.verify_show_duration = 60

        self.menu_active = False
        self.menu_selected_index = 0
        self.main_menu_items = [
            ("数据采集", "1"),
            ("模型训练", "2"),
            ("考勤模式", "3"),
            ("退出系统", "4"),
        ]
        self.menu_gesture_timer = 0
        self.menu_last_finger = 0
        self.menu_trigger_frames = 15
        self.menu_action_locked = False

    # ===================== 手势防抖核心方法 =====================
    def _filter_gesture(self, raw_finger: int) -> int:
        if raw_finger == 0:
            self._last_raw_finger = 0
            self._same_raw_count = 0
            self.stable_finger_count = 0
            self._stable_enter_counter = 0
            self._fault_remain = 0
            return 0

        if raw_finger == self._last_raw_finger:
            self._same_raw_count += 1
        else:
            if self._fault_remain > 0:
                self._fault_remain -= 1
                return self.stable_finger_count
            self._same_raw_count = 1
            self._last_raw_finger = raw_finger

        if self._same_raw_count >= self.gesture_stable_frames:
            if raw_finger != self.stable_finger_count:
                self.stable_finger_count = raw_finger
                self._stable_enter_counter = 0
                self._fault_remain = self.gesture_fault_tolerance
            else:
                if self._stable_enter_counter < self.gesture_enter_delay:
                    self._stable_enter_counter += 1
                    return 0

        return self.stable_finger_count

    # ===================== 分层反馈核心方法 =====================
    def push_feedback(self, msg: str, level: str = "info", duration: int = 60):
        self.feedback_stack.append({
            "msg": msg,
            "level": level,
            "remain": duration
        })

    def _draw_feedback(self, img):
        self.feedback_stack = [f for f in self.feedback_stack if f["remain"] > 0]
        if not self.feedback_stack:
            return img

        priority = {"error": 3, "warning": 2, "success": 1, "info": 0}
        top_feedback = max(self.feedback_stack, key=lambda x: priority.get(x["level"], 0))
        top_feedback["remain"] -= 1

        bg_color, text_color = self.feedback_colors[top_feedback["level"]]
        msg = top_feedback["msg"]

        img_h, img_w = img.shape[:2]
        bar_height = 50
        bar_y = 20
        overlay = img.copy()
        cv.rectangle(overlay, (0, bar_y), (img_w, bar_y + bar_height), bg_color, -1)
        cv.addWeighted(overlay, 0.85, img, 0.15, 0, img)

        text_x = img_w // 2 - len(msg) * 10
        img = put_chinese_text(img, msg, (text_x, bar_y + 12), text_color=text_color, font_size=22)
        return img

    def _draw_menu(self, img):
        menu_x, menu_y = 30, 60
        item_height = 45
        menu_width = 280
        menu_height = len(self.main_menu_items) * item_height + 90
        overlay = img.copy()
        cv.rectangle(overlay, (menu_x, menu_y), (menu_x + menu_width, menu_y + menu_height), (0, 0, 0), -1)
        cv.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        img = put_chinese_text(img, "手势主菜单", (menu_x + 15, menu_y + 10), text_color=(255, 255, 255), font_size=22)
        for i, (name, _) in enumerate(self.main_menu_items):
            y = menu_y + 45 + i * item_height
            if i == self.menu_selected_index:
                cv.rectangle(img, (menu_x + 5, y - 5), (menu_x + menu_width - 5, y + item_height - 10), (46, 204, 113),
                             -1)
                text_color = (255, 255, 255)
                prefix = "▶ "
            else:
                text_color = (200, 200, 200)
                prefix = "  "
            img = put_chinese_text(img, f"{prefix}{name}", (menu_x + 15, y), text_color=text_color, font_size=20)
        hint_y = menu_y + menu_height - 50
        img = put_chinese_text(img, "1/2切换 | 3确认 | 4/5退出", (menu_x + 15, hint_y), text_color=(150, 150, 150),
                               font_size=16)
        state_y = hint_y + 25
        if getattr(self, 'menu_action_locked', False):
            img = put_chinese_text(img, "[锁定] 请握拳或放下手", (menu_x + 15, state_y), text_color=(231, 76, 60),
                                   font_size=15)
        else:
            img = put_chinese_text(img, "[就绪] 等待操作...", (menu_x + 15, state_y), text_color=(46, 204, 113),
                                   font_size=15)
        return img

    def _process_menu_gesture(self, finger_count, img):
        triggered_cmd = None
        if finger_count == 0:
            self.menu_action_locked = False
            self.menu_gesture_timer = 0
            self.menu_last_finger = 0
            return None, img
        if self.menu_action_locked:
            return None, img

        if finger_count == self.menu_last_finger and finger_count != 0:
            self.menu_gesture_timer += 1
        else:
            self.menu_gesture_timer = 1
            self.menu_last_finger = finger_count

        if self.menu_last_finger in [1, 2, 3, 4, 5]:
            bar_w = int(200 * (self.menu_gesture_timer / self.menu_trigger_frames))
            cv.rectangle(img, (340, 70), (540, 85), (100, 100, 100), 2)
            cv.rectangle(img, (340, 70), (340 + bar_w, 85), (46, 204, 113), -1)
            img = put_chinese_text(img, f"指令 {self.menu_last_finger}", (340, 40), text_color=(46, 204, 113),
                                   font_size=18)

        if self.menu_gesture_timer >= self.menu_trigger_frames:
            if finger_count == 1:
                self.menu_selected_index = (self.menu_selected_index - 1) % len(self.main_menu_items)
                self.menu_action_locked = True
            elif finger_count == 2:
                self.menu_selected_index = (self.menu_selected_index + 1) % len(self.main_menu_items)
                self.menu_action_locked = True
            elif finger_count == 3:
                _, cmd = self.main_menu_items[self.menu_selected_index]
                triggered_cmd = cmd
                self.menu_active = False
                self.menu_action_locked = True
            elif finger_count in [4, 5]:
                self.menu_active = False
                self.menu_action_locked = True
            self.menu_gesture_timer = 0
        return triggered_cmd, img

    def _init_face_model(self):
        try:
            self.face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
            self.recognizer = cv.face.LBPHFaceRecognizer_create()
            self.model_ready = False
            if os.path.exists(model_file):
                self.recognizer.read(model_file)
                self.model_ready = True  # 只有成功加载模型才标记为就绪
            self._refresh_name_mapping()
        except Exception as e:
            self.model_ready = False
            print(f"[模型加载失败] {e}")

    def _refresh_name_mapping(self):
        if os.path.exists(names_mapping_path):
            with open(names_mapping_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            self.id_name_map = {v: k for k, v in mapping.items()}

    def get_user_list(self):
        if os.path.exists(names_mapping_path):
            with open(names_mapping_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            return sorted([(v, k) for k, v in mapping.items()], key=lambda x: x[0])
        return []

    def delete_user(self, user_name):
        if not os.path.exists(names_mapping_path):
            return False, "用户映射文件不存在"
        with open(names_mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        if user_name not in mapping:
            return False, f"未找到用户：{user_name}"
        user_id = mapping.pop(user_name)
        with open(names_mapping_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        if os.path.exists(dataset_path):
            for filename in os.listdir(dataset_path):
                if filename.startswith(f"User.{user_id}."):
                    try:
                        os.remove(os.path.join(dataset_path, filename))
                    except:
                        pass
        if user_name in self.gesture_passwords:
            self.gesture_passwords.pop(user_name)
            self._save_gesture_passwords()
        if os.path.exists(model_file):
            try:
                os.remove(model_file)
                self.model_ready = False  # 删除模型后同步更新状态
            except:
                pass
        self._refresh_name_mapping()
        return True, f"用户 {user_name} 已删除，模型缓存已清理，请重新训练模型"

    def _load_gesture_passwords(self):
        if os.path.exists(gesture_password_path):
            with open(gesture_password_path, 'r', encoding='utf-8') as f:
                self.gesture_passwords = json.load(f)
        else:
            self.gesture_passwords = {}

    def _save_gesture_passwords(self):
        with open(gesture_password_path, 'w', encoding='utf-8') as f:
            json.dump(self.gesture_passwords, f, ensure_ascii=False, indent=2)

    def set_user_gesture(self, user_name, gesture_num):
        if not 1 <= gesture_num <= 5:
            return False, "手势密码必须为1-5根手指"
        self.gesture_passwords[user_name] = gesture_num
        self._save_gesture_passwords()
        return True, f"用户 {user_name} 的手势密码已设为 {gesture_num} 指"

    def start_verification(self, user_name):
        if not self.model_ready:
            return False, "模型未训练，无法启动验证"
        if user_name not in self.id_name_map.values():
            return False, "用户不存在，无法启动验证"
        if user_name not in self.gesture_passwords:
            self.gesture_passwords[user_name] = 1
            self._save_gesture_passwords()
        self.verify_target_user = user_name
        self.verify_target_gesture = self.gesture_passwords[user_name]
        self.verify_state = 'wait_face'
        self.verify_frame_counter = 0
        return True, "验证已启动，请正对摄像头"

    def cancel_verification(self):
        self.verify_state = 'idle'
        self.verify_target_user = None
        self.verify_target_gesture = None
        self.verify_frame_counter = 0

    def _process_verification_frame(self, img, finger_count):
        # 模型未就绪直接返回提示，不调用predict
        if not self.model_ready:
            cv.rectangle(img, (10, 10), (450, 120), (0, 0, 0), -1)
            img = put_chinese_text(img, "【双因子安全验证】", (20, 15), text_color=(255, 255, 255), font_size=24)
            img = put_chinese_text(img, "错误：未训练模型，请先训练", (20, 75),
                                   text_color=(0, 0, 255), font_size=20)
            return img

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
        cv.rectangle(img, (10, 10), (450, 120), (0, 0, 0), -1)
        img = put_chinese_text(img, "【双因子安全验证】", (20, 15), text_color=(255, 255, 255), font_size=24)

        if self.verify_state == 'wait_face':
            img = put_chinese_text(img, f"第一步：请正对摄像头 [{self.verify_target_user}]", (20, 75),
                                   text_color=(255, 255, 0), font_size=20)
            for (x, y, w, h) in faces:
                face_id, confidence = self.recognizer.predict(gray[y:y + h, x:x + w])
                if confidence < 100:
                    name = self.id_name_map.get(face_id, "未知")
                    if name == self.verify_target_user:
                        self.verify_state = 'wait_gesture'
                        self.verify_frame_counter = 0
                        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        break
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        elif self.verify_state == 'wait_gesture':
            self.verify_frame_counter += 1
            remain = max(0, self.verify_timeout - self.verify_frame_counter)
            img = put_chinese_text(img, f"第二步：请伸出 {self.verify_target_gesture} 根手指", (20, 75),
                                   text_color=(0, 255, 255), font_size=20)
            img = put_chinese_text(img, f"剩余时间：{remain / 60:.1f}s", (20, 100), text_color=(255, 255, 255),
                                   font_size=18)
            if finger_count == self.verify_target_gesture:
                self.verify_state = 'success'
                self.verify_frame_counter = 0
            elif self.verify_frame_counter >= self.verify_timeout:
                self.verify_state = 'failed'
                self.verify_frame_counter = 0

        elif self.verify_state == 'success':
            self.verify_frame_counter += 1
            img = put_chinese_text(img, "[通过] 验证成功", (20, 80), text_color=(0, 255, 0), font_size=24)
            if self.verify_frame_counter >= self.verify_show_duration:
                self.cancel_verification()

        elif self.verify_state == 'failed':
            self.verify_frame_counter += 1
            img = put_chinese_text(img, "[失败] 超时/手势错误", (20, 80), text_color=(0, 0, 255), font_size=24)
            if self.verify_frame_counter >= self.verify_show_duration:
                self.cancel_verification()

        return img

    def _load_attendance(self):
        today_str = datetime.now().strftime("%Y-%m-%d")
        if os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'r', encoding='utf-8') as f:
                all_records = json.load(f)
            self.today_attendance = all_records.get(today_str, {})
            for name, record in self.today_attendance.items():
                if isinstance(record, str):
                    self.today_attendance[name] = {"in": record, "out": "--:--:--"}
        else:
            self.today_attendance = {}

    def _save_attendance(self):
        today_str = datetime.now().strftime("%Y-%m-%d")
        all_records = {}
        if os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'r', encoding='utf-8') as f:
                all_records = json.load(f)
        all_records[today_str] = self.today_attendance
        with open(self.attendance_file, 'w', encoding='utf-8') as f:
            json.dump(all_records, f, ensure_ascii=False, indent=2)

    def _process_attendance_frame(self, img, finger_count):
        # 用 model_ready 判断替代 recognizer is None
        if not self.model_ready or self.face_cascade is None:
            img = put_chinese_text(img, "未训练模型，请先执行模型训练", (20, 40), text_color=(0, 0, 255), font_size=20)
            return img

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
        now_time = time.time()

        for (x, y, w, h) in faces:
            # 异常捕获，极端情况也不会崩溃
            try:
                face_id, confidence = self.recognizer.predict(gray[y:y + h, x:x + w])
            except Exception:
                name = "未知"
                color = (0, 0, 255)
            else:
                if confidence < 100:
                    name = self.id_name_map.get(face_id, "未知")
                    color = (0, 255, 0)
                else:
                    name = "未知"
                    color = (0, 0, 255)

            if name != "未知":
                cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
                img = put_chinese_text(img, f"[{name}] 1指签到 | 2指签退", (x, max(0, y - 35)),
                                       text_color=(255, 255, 0), font_size=18)

                if finger_count in [1, 2]:
                    last_time = self.last_sign_timestamp.get(name, 0)
                    if now_time - last_time > self.sign_cooldown:
                        date_str = datetime.now().strftime("%Y-%m-%d")
                        time_str = datetime.now().strftime("%H:%M:%S")
                        if name not in self.today_attendance:
                            self.today_attendance[name] = {"in": "--:--:--", "out": "--:--:--"}

                        if finger_count == 1:
                            if self.today_attendance[name]["in"] == "--:--:--":
                                self.today_attendance[name]["in"] = time_str
                                self.push_feedback(f"[成功] {name} 签到成功 {time_str}",
                                                   level=self.FEEDBACK_SUCCESS, duration=60)
                            else:
                                self.push_feedback(f"[警告] {name} 今日已签到，请勿重复打卡",
                                                   level=self.FEEDBACK_WARNING, duration=60)
                        elif finger_count == 2:
                            if self.today_attendance[name]["in"] == "--:--:--":
                                self.push_feedback(f"[警告] {name} 签退失败：请先完成签到",
                                                   level=self.FEEDBACK_WARNING, duration=60)
                            else:
                                self.today_attendance[name]["out"] = time_str
                                self.push_feedback(f"[成功] {name} 签退成功 {time_str}",
                                                   level=self.FEEDBACK_SUCCESS, duration=60)

                        self.last_sign_timestamp[name] = now_time
                        self._save_attendance()
                    else:
                        if not any(f["msg"].startswith("[冷却]") for f in self.feedback_stack):
                            self.push_feedback("[冷却] 操作过快，系统冷却中...",
                                               level=self.FEEDBACK_WARNING, duration=20)
            else:
                cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
                img = put_chinese_text(img, "未注册人员", (x, max(0, y - 30)), text_color=color, font_size=20)

        current_date_str = datetime.now().strftime("%Y-%m-%d")
        img = put_chinese_text(img, f"[{current_date_str}] 考勤动态: {len(self.today_attendance)}人", (20, 20),
                               text_color=(255, 255, 255), font_size=20)
        return img

    def toggle_attendance_mode(self):
        # 开启考勤前校验模型状态
        if not self.attendance_mode and not self.model_ready:
            return False, "请先完成数据采集与模型训练，再开启考勤模式"
        self.attendance_mode = not self.attendance_mode
        if self.attendance_mode:
            self._refresh_name_mapping()
            return True, "已进入双因子考勤模式"
        else:
            return True, "已退出考勤模式"

    def get_attendance_list(self):
        return self.today_attendance

    def open_camera(self):
        self.cap = cv.VideoCapture(0)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        return self.cap.isOpened()

    def close_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def get_frame_and_gesture(self):
        if self.is_busy or self.cap is None:
            return None, None
        success, img = self.cap.read()
        if not success:
            return None, None

        img = cv.flip(img, 1)
        img = self.detector.find_hands(img)
        lmslist = self.detector.find_positions(img)

        finger_count = 0
        if len(lmslist) > 0:
            fingers = []
            for tid in self.tip_ids:
                if tid == 4:
                    if lmslist[8][1] < lmslist[12][1]:
                        fingers.append(1 if lmslist[tid][1] < lmslist[tid - 1][1] else 0)
                    else:
                        fingers.append(1 if lmslist[tid][1] > lmslist[tid - 1][1] else 0)
                else:
                    fingers.append(1 if lmslist[tid][2] < lmslist[tid - 2][2] else 0)
            finger_count = fingers.count(1)

        # 全局手势防抖：后续所有业务统一使用 stable_finger
        stable_finger = self._filter_gesture(finger_count)
        triggered_command = None

        if self.verify_state != 'idle':
            img = self._process_verification_frame(img, stable_finger)
        else:
            if self.menu_active:
                triggered_command, img = self._process_menu_gesture(stable_finger, img)
                img = self._draw_menu(img)
            else:
                if self.attendance_mode:
                    img = self._process_attendance_frame(img, stable_finger)

                if 1 <= stable_finger <= 5:
                    if self.attendance_mode and stable_finger in [1, 2]:
                        self.current_gesture_count = 0
                        self.gesture_hold_timer = 0
                    else:
                        if stable_finger == self.current_gesture_count:
                            self.gesture_hold_timer += 1
                        else:
                            self.current_gesture_count = stable_finger
                            self.gesture_hold_timer = 1

                        if stable_finger == 5:
                            img = put_chinese_text(img, "唤出手势菜单...", (20, 70), text_color=(0, 255, 255),
                                                   font_size=20)
                        else:
                            img = put_chinese_text(img, f"Trigger Option {stable_finger}...", (20, 70),
                                                   text_color=(0, 255, 0), font_size=20)

                        bar_width = int(200 * (self.gesture_hold_timer / self.confirm_frames))
                        cv.rectangle(img, (20, 100), (220, 115), (100, 100, 100), 2)
                        cv.rectangle(img, (20, 100), (20 + bar_width, 115), (0, 255, 0), -1)

                        if self.gesture_hold_timer >= self.confirm_frames:
                            if stable_finger == 5:
                                self.menu_active = True
                                self.menu_selected_index = 0
                                self.menu_action_locked = True
                            else:
                                triggered_command = str(self.current_gesture_count)
                            self.gesture_hold_timer = 0
                            self.current_gesture_count = 0
                else:
                    self.current_gesture_count = 0
                    self.gesture_hold_timer = 0

        # 绘制全局分层反馈条
        img = self._draw_feedback(img)
        return img, triggered_command

    def execute_task(self, choice, user_name=""):
        if choice == '1' and not user_name:
            return False, "请先填写姓名！"
        self.is_busy = True
        try:
            cv.destroyWindow('image')
        except:
            pass
        try:
            if choice == '1':
                self.captured_faces.clear()
                is_new_user = user_name not in self.id_name_map.values()
                face_id = self._get_id(user_name)

                def on_face_captured(count, face_roi):
                    self.captured_faces.append(face_roi.copy())

                success = capture_faces(face_id, cam=self.cap, show_preview=False, capture_callback=on_face_captured)
                if not success and is_new_user:
                    self.delete_user(user_name)
                    return False, "采集被终止：检测到已有重复人脸，注册已撤销"
                return success, "采集完成" if success else "采集被终止"
            elif choice == '2':
                train_model()
                self._init_face_model()  # 训练完成后重新加载模型并更新就绪状态
                return True, "训练完成"
        except Exception as e:
            return False, f"执行异常: {str(e)}"
        finally:
            self.is_busy = False

    def _get_id(self, name):
        if os.path.exists(names_mapping_path):
            with open(names_mapping_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
        else:
            mapping = {}
        if name in mapping:
            return mapping[name]
        new_id = max(mapping.values(), default=0) + 1
        mapping[name] = new_id
        with open(names_mapping_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        if name not in self.gesture_passwords:
            self.gesture_passwords[name] = 1
            self._save_gesture_passwords()
        return new_id