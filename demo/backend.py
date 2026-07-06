import cv2 as cv
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

        self.cap = None
        self.detector = HandDetector(detection_con=0.7)
        self.tip_ids = [4, 8, 12, 16, 20]

        self.confirm_frames = 20
        self.current_gesture_count = 0
        self.gesture_hold_timer = 0
        self.is_busy = False

        self.face_cascade = None
        self.recognizer = None
        self.id_name_map = {}
        self._init_face_model()

        self.attendance_mode = False
        self.attendance_file = attendance_records_path
        self.today_attendance = {}
        self.last_sign_timestamp = {}
        self.sign_cooldown = 30  # 30秒防抖冷却
        self.attendance_feedback = {"msg": "", "timer": 0}
        self._load_attendance()

        self.gesture_passwords = {}
        self._load_gesture_passwords()
        self.verify_state = 'idle'
        self.verify_target_user = None
        self.verify_target_gesture = None
        self.verify_frame_counter = 0
        self.verify_timeout = 300
        self.verify_show_duration = 60

        # ========== 全手势菜单 ==========
        self.menu_active = False  # 菜单激活状态
        self.menu_selected_index = 0  # 当前选中的菜单项索引
        self.main_menu_items = [  # 主菜单配置：(显示名称, 触发命令)
            ("数据采集", "1"),
            ("模型训练", "2"),
            ("考勤模式", "3"),
            ("退出系统", "4"),
        ]
        self.menu_gesture_timer = 0  # 菜单手势保持计时器
        self.menu_last_finger = 0  # 上一帧识别的手指数
        self.menu_trigger_frames = 15  # 菜单操作触发所需保持帧数
        self.menu_action_locked = False  # 【新增】动作锁，防止连续触发和秒退

    def _draw_menu(self, img):
        """在画面上绘制手势悬浮菜单"""
        menu_x, menu_y = 30, 60
        item_height = 45
        menu_width = 280
        # 增加高度以容纳底部状态栏
        menu_height = len(self.main_menu_items) * item_height + 90

        # 绘制半透明黑色背景
        overlay = img.copy()
        cv.rectangle(overlay, (menu_x, menu_y),
                     (menu_x + menu_width, menu_y + menu_height),
                     (0, 0, 0), -1)
        cv.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        # 菜单标题
        img = put_chinese_text(img, "手势主菜单", (menu_x + 15, menu_y + 10),
                               text_color=(255, 255, 255), font_size=22)

        # 遍历绘制菜单项
        for i, (name, _) in enumerate(self.main_menu_items):
            y = menu_y + 45 + i * item_height
            if i == self.menu_selected_index:
                # 选中项绿色高亮
                cv.rectangle(img, (menu_x + 5, y - 5),
                             (menu_x + menu_width - 5, y + item_height - 10),
                             (46, 204, 113), -1)
                text_color = (255, 255, 255)
                prefix = "▶ "
            else:
                text_color = (200, 200, 200)
                prefix = "  "
            img = put_chinese_text(img, f"{prefix}{name}", (menu_x + 15, y),
                                   text_color=text_color, font_size=20)

        # 底部操作提示与状态锁展示
        hint_y = menu_y + menu_height - 50
        img = put_chinese_text(img, "1/2切换 | 3确认 | 4/5退出", (menu_x + 15, hint_y),
                               text_color=(150, 150, 150), font_size=16)

        state_y = hint_y + 25
        if getattr(self, 'menu_action_locked', False):
            img = put_chinese_text(img, "🔒 动作锁定: 请握拳或放下手", (menu_x + 15, state_y),
                                   text_color=(231, 76, 60), font_size=15)
        else:
            img = put_chinese_text(img, "🔓 等待操作...", (menu_x + 15, state_y),
                                   text_color=(46, 204, 113), font_size=15)

        return img

    def _process_menu_gesture(self, finger_count, img):
        """处理菜单内的手势交互，带边缘触发与防抖锁"""
        triggered_cmd = None

        # 【核心优化1】中立状态解锁：只要识别到0指（握拳，或者手不在画面内），就解除动作锁
        if finger_count == 0:
            self.menu_action_locked = False
            self.menu_gesture_timer = 0
            self.menu_last_finger = 0
            return None, img

        # 如果处于锁定状态，则忽略任何1-5指的输入
        if self.menu_action_locked:
            return None, img

        # 手势保持计时逻辑
        if finger_count == self.menu_last_finger and finger_count != 0:
            self.menu_gesture_timer += 1
        else:
            self.menu_gesture_timer = 1
            self.menu_last_finger = finger_count

        # 【核心优化2】菜单内独立进度条：给用户明确的确认感
        if self.menu_last_finger in [1, 2, 3, 4, 5]:
            bar_w = int(200 * (self.menu_gesture_timer / self.menu_trigger_frames))
            cv.rectangle(img, (340, 70), (540, 85), (100, 100, 100), 2)
            cv.rectangle(img, (340, 70), (340 + bar_w, 85), (46, 204, 113), -1)
            img = put_chinese_text(img, f"指令 {self.menu_last_finger}", (340, 40),
                                   text_color=(46, 204, 113), font_size=18)

        # 达到触发阈值执行对应操作
        if self.menu_gesture_timer >= self.menu_trigger_frames:
            if finger_count == 1:
                self.menu_selected_index = (self.menu_selected_index - 1) % len(self.main_menu_items)
                self.menu_action_locked = True  # 【核心优化3】执行后立即加锁
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

            self.menu_gesture_timer = 0  # 执行后重置计时

        return triggered_cmd, img

    def _init_face_model(self):
        try:
            self.face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
            self.recognizer = cv.face.LBPHFaceRecognizer_create()
            if os.path.exists(model_file):
                self.recognizer.read(model_file)
            self._refresh_name_mapping()
        except Exception as e:
            print(f"[警告] 人脸识别模型加载失败：{e}，请先训练模型")

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
        if not os.path.exists(names_mapping_path): return False, "用户映射文件不存在"
        with open(names_mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        if user_name not in mapping: return False, f"未找到用户：{user_name}"

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

        self._refresh_name_mapping()
        return True, f"用户 {user_name} 已删除，请重新训练模型以完全生效"

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
        if not 1 <= gesture_num <= 5: return False, "手势密码必须为1-5根手指"
        self.gesture_passwords[user_name] = gesture_num
        self._save_gesture_passwords()
        return True, f"用户 {user_name} 的手势密码已设为 {gesture_num} 指"

    def start_verification(self, user_name):
        if user_name not in self.id_name_map.values(): return False, "用户不存在，无法启动验证"
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
            img = put_chinese_text(img, "✅ 验证通过", (20, 80), text_color=(0, 255, 0), font_size=24)
            if self.verify_frame_counter >= self.verify_show_duration: self.cancel_verification()

        elif self.verify_state == 'failed':
            self.verify_frame_counter += 1
            img = put_chinese_text(img, "❌ 验证失败（超时/手势错误）", (20, 80), text_color=(0, 0, 255), font_size=24)
            if self.verify_frame_counter >= self.verify_show_duration: self.cancel_verification()
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
        if self.recognizer is None or self.face_cascade is None:
            img = put_chinese_text(img, "未训练模型，请先执行模型训练", (20, 40), text_color=(0, 0, 255), font_size=20)
            return img

        if self.attendance_feedback["timer"] > 0:
            img = put_chinese_text(img, self.attendance_feedback["msg"], (20, 150), text_color=(0, 255, 255),
                                   font_size=22)
            self.attendance_feedback["timer"] -= 1

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
        now_time = time.time()

        for (x, y, w, h) in faces:
            face_id, confidence = self.recognizer.predict(gray[y:y + h, x:x + w])
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
                                self.attendance_feedback = {"msg": f"✅ {name} 签到成功 ({date_str} {time_str})",
                                                            "timer": 60}
                            else:
                                self.attendance_feedback = {"msg": f"⚠️ {name} 今日已签到，请勿重复打卡", "timer": 60}

                        elif finger_count == 2:
                            if self.today_attendance[name]["in"] == "--:--:--":
                                self.attendance_feedback = {"msg": f"⚠️ {name} 签退失败：请先完成签到", "timer": 60}
                            else:
                                self.today_attendance[name]["out"] = time_str
                                self.attendance_feedback = {"msg": f"✅ {name} 签退成功 ({date_str} {time_str})",
                                                            "timer": 60}

                        self.last_sign_timestamp[name] = now_time
                        self._save_attendance()
                    else:
                        if self.attendance_feedback["timer"] == 0:
                            self.attendance_feedback = {"msg": f"⏳ 操作过快，系统冷却中...", "timer": 20}
            else:
                cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
                img = put_chinese_text(img, "未注册人员", (x, max(0, y - 30)), text_color=color, font_size=20)

        current_date_str = datetime.now().strftime("%Y-%m-%d")
        img = put_chinese_text(img, f"[{current_date_str}] 考勤动态: {len(self.today_attendance)}人", (20, 20),
                               text_color=(255, 255, 255), font_size=20)
        return img

    def toggle_attendance_mode(self):
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
        if self.is_busy or self.cap is None: return None, None

        success, img = self.cap.read()
        if not success: return None, None

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

        triggered_command = None
        if self.verify_state != 'idle':
            img = self._process_verification_frame(img, finger_count)
            triggered_command = None
        else:
            if self.menu_active:
                # ========== 菜单模式：带有安全锁和进度条的处理 ==========
                triggered_command, img = self._process_menu_gesture(finger_count, img)
                img = self._draw_menu(img)
            else:
                # ========== 非菜单模式 ==========
                if self.attendance_mode:
                    img = self._process_attendance_frame(img, finger_count)

                if 1 <= finger_count <= 5:
                    if self.attendance_mode and finger_count in [1, 2]:
                        self.current_gesture_count = 0
                        self.gesture_hold_timer = 0
                    else:
                        if finger_count == self.current_gesture_count:
                            self.gesture_hold_timer += 1
                        else:
                            self.current_gesture_count = finger_count
                            self.gesture_hold_timer = 1

                        if finger_count == 5:
                            img = put_chinese_text(img, "唤出手势菜单...", (20, 70),
                                                   text_color=(0, 255, 255), font_size=20)
                        else:
                            img = put_chinese_text(img, f"Trigger Option {finger_count}...", (20, 70),
                                                   text_color=(0, 255, 0), font_size=20)

                        bar_width = int(200 * (self.gesture_hold_timer / self.confirm_frames))
                        cv.rectangle(img, (20, 100), (220, 115), (100, 100, 100), 2)
                        cv.rectangle(img, (20, 100), (20 + bar_width, 115), (0, 255, 0), -1)

                        if self.gesture_hold_timer >= self.confirm_frames:
                            if finger_count == 5:
                                # 5指：激活菜单并【立即加锁】
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

        return img, triggered_command

    def execute_task(self, choice, user_name=""):
        if choice == '1' and not user_name: return False, "请先填写姓名！"
        self.is_busy = True
        self.close_camera()
        try:
            if choice == '1':
                face_id = self._get_id(user_name)
                capture_faces(face_id)
            elif choice == '2':
                train_model()
                self._init_face_model()
        except Exception as e:
            return False, f"执行异常: {str(e)}"
        finally:
            self.open_camera()
            self.is_busy = False
        return True, "执行完成"

    def _get_id(self, name):
        if os.path.exists(names_mapping_path):
            with open(names_mapping_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
        else:
            mapping = {}

        if name in mapping: return mapping[name]

        new_id = max(mapping.values(), default=0) + 1
        mapping[name] = new_id
        with open(names_mapping_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)

        if name not in self.gesture_passwords:
            self.gesture_passwords[name] = 1
            self._save_gesture_passwords()
        return new_id