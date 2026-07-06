import threading
from tkinter import messagebox, ttk, simpledialog
import customtkinter as ctk
import cv2 as cv
import numpy as np
from PIL import Image
from backend import TerminalBackend
from RK_face import put_chinese_text

# ==================== 全局主题设置 ====================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class SmartTerminalUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face2Hand V2.0")
        self.root.geometry("1150x680")

        self.backend = TerminalBackend()
        if not self.backend.open_camera():
            messagebox.showerror("错误", "无法打开摄像头！")
            self.root.quit()

        self.pending_action = None
        self.task_thread = None  # 后台任务线程
        self.task_result = None  # 后台任务执行结果

        self.setup_ui()
        self.refresh_user_list()
        self.update_loop()

    def setup_ui(self):
        # ========== 主布局：左右分栏 ==========
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=0)
        self.root.grid_rowconfigure(0, weight=1)

        # ========== 左侧：视频画面区域 ==========
        self.video_frame = ctk.CTkFrame(self.root, corner_radius=15)
        self.video_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        self.cam_title = ctk.CTkLabel(self.video_frame, text="🟢 LIVE CAMERA FEED",
                                      font=ctk.CTkFont(family="Consolas", size=14, weight="bold"), text_color="#2ECC71")
        self.cam_title.pack(pady=(15, 5))

        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack(expand=True, padx=15, pady=(0, 15))

        self.control_frame = ctk.CTkFrame(self.root, width=320, corner_radius=15)
        self.control_frame.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")

        self.app_logo = ctk.CTkLabel(self.control_frame, text="Face2Hand",
                                     font=ctk.CTkFont(size=24, weight="bold"), justify="center")
        self.app_logo.pack(pady=(20, 10))

        self.tabview = ctk.CTkTabview(self.control_frame, width=280)
        self.tabview.pack(padx=20, pady=10, fill="both", expand=True)

        self.tabview.add("操作终端")
        self.tabview.add("用户矩阵")
        self.tabview.add("考勤日志")

        self._build_operate_tab()
        self._build_user_tab()
        self._build_attendance_tab()

    def _build_operate_tab(self):
        tab = self.tabview.tab("操作终端")

        self.name_entry = ctk.CTkEntry(tab, placeholder_text="在此输入录入姓名...", height=40, corner_radius=8)
        self.name_entry.pack(fill="x", padx=10, pady=(15, 10))

        btn_kwargs = {"height": 40, "corner_radius": 8, "font": ctk.CTkFont(size=14)}
        ctk.CTkButton(tab, text="[1] 数据采集", command=lambda: self.on_action_triggered('1'), **btn_kwargs).pack(
            fill="x", padx=10, pady=6)
        ctk.CTkButton(tab, text="[2] 模型训练", command=lambda: self.on_action_triggered('2'), **btn_kwargs).pack(
            fill="x", padx=10, pady=6)
        ctk.CTkButton(tab, text="[3] 考勤模式", fg_color="#E67E22", hover_color="#D35400",
                      command=lambda: self.on_action_triggered('3'), **btn_kwargs).pack(fill="x", padx=10, pady=6)

        hint_label = ctk.CTkLabel(tab, text="💡 考勤模式下：伸出1指签到，伸出2指签退", font=ctk.CTkFont(size=12),
                                  text_color="#BDC3C7")
        hint_label.pack(pady=(0, 10))

        self.status_box = ctk.CTkFrame(tab, fg_color="#2C3E50", corner_radius=6)
        self.status_box.pack(fill="x", padx=10, pady=10)
        self.status_label = ctk.CTkLabel(self.status_box, text="SYS: STANDBY",
                                         font=ctk.CTkFont(family="Consolas", size=13), text_color="#2ECC71")
        self.status_label.pack(pady=8)

        ctk.CTkButton(tab, text="退出系统 (伸4指)", fg_color="#E74C3C", hover_color="#C0392B", command=self.quit_app,
                      **btn_kwargs).pack(side="bottom", fill="x", padx=10, pady=15)

    def _build_user_tab(self):
        tab = self.tabview.tab("用户矩阵")
        self._style_treeview()

        tree_frame = ctk.CTkFrame(tab, fg_color="transparent")
        tree_frame.pack(fill="both", expand=True, padx=5, pady=(10, 5))

        self.user_tree = ttk.Treeview(tree_frame, columns=("id", "name"), show="headings", height=10)
        self.user_tree.heading("id", text="UID")
        self.user_tree.heading("name", text="姓名")
        self.user_tree.column("id", width=60, anchor="center")
        self.user_tree.column("name", width=160, anchor="center")
        self.user_tree.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.user_tree.yview)
        scrollbar.pack(side="right", fill="y")
        self.user_tree.configure(yscrollcommand=scrollbar.set)

        btn_frame = ctk.CTkFrame(tab, fg_color="transparent")
        btn_frame.pack(fill="x", padx=5, pady=10)

        ctk.CTkButton(btn_frame, text="刷新", width=70, command=self.refresh_user_list).pack(side="left", padx=2)
        ctk.CTkButton(btn_frame, text="设手势", width=70, fg_color="#8E44AD", hover_color="#732D91",
                      command=self.set_user_gesture).pack(side="left", padx=2)
        ctk.CTkButton(btn_frame, text="删除", width=70, fg_color="transparent", border_width=1, text_color="#E74C3C",
                      border_color="#E74C3C", hover_color="#3A1919", command=self.delete_selected_user).pack(
            side="right", padx=2)

    def _build_attendance_tab(self):
        tab = self.tabview.tab("考勤日志")

        tree_frame = ctk.CTkFrame(tab, fg_color="transparent")
        tree_frame.pack(fill="both", expand=True, padx=5, pady=(10, 5))

        self.attendance_tree = ttk.Treeview(tree_frame, columns=("name", "sign_in", "sign_out"), show="headings",
                                            height=12)
        self.attendance_tree.heading("name", text="姓名")
        self.attendance_tree.heading("sign_in", text="签到时间")
        self.attendance_tree.heading("sign_out", text="签退时间")

        self.attendance_tree.column("name", width=60, anchor="center")
        self.attendance_tree.column("sign_in", width=80, anchor="center")
        self.attendance_tree.column("sign_out", width=80, anchor="center")
        self.attendance_tree.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.attendance_tree.yview)
        scrollbar.pack(side="right", fill="y")
        self.attendance_tree.configure(yscrollcommand=scrollbar.set)

        self.attendance_count_label = ctk.CTkLabel(tab, text="TOTAL RECORDS: 0",
                                                   font=ctk.CTkFont(family="Consolas", size=13, weight="bold"),
                                                   text_color="#3498DB")
        self.attendance_count_label.pack(anchor="w", padx=10, pady=10)

    def _style_treeview(self):
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview", background="#2B2B2B", foreground="white", rowheight=30, fieldbackground="#2B2B2B",
                        borderwidth=0)
        style.map('Treeview', background=[('selected', '#1F6AA5')])
        style.configure("Treeview.Heading", background="#333333", foreground="white", relief="flat",
                        font=("微软雅黑", 10, "bold"))
        style.map("Treeview.Heading", background=[('active', '#3E3E3E')])

    # ===================== 业务与画面刷新 =====================
    def _show_loading_frame(self):
        """后台处理时，主画面展示等待动画"""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img = put_chinese_text(img, "系统运行中...", (210, 200), text_color=(0, 255, 255), font_size=36)
        img = put_chinese_text(img, "请不要关闭系统", (240, 260), text_color=(200, 200, 200), font_size=22)
        cv2_im = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        ctk_img = ctk.CTkImage(light_image=pil_im, dark_image=pil_im, size=(640, 480))
        self.video_label.configure(image=ctk_img)
        self.video_label.image = ctk_img

    def update_loop(self):
        # 检测后台线程是否执行完毕
        if self.task_thread and not self.task_thread.is_alive():
            success, msg, choice = self.task_result
            self.task_thread = None
            self.task_result = None
            self._on_task_finished(success, msg, choice)

        if self.backend.is_busy or self.task_thread:
            self._show_loading_frame()
        else:
            # 只有系统空闲时，主线程才去读取摄像头
            img, cmd = self.backend.get_frame_and_gesture()
            if img is not None:
                img_h, img_w, _ = img.shape
                target_w = 640
                target_h = int(img_h * (target_w / img_w))

                cv2_im = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                pil_im = Image.fromarray(cv2_im)

                ctk_img = ctk.CTkImage(light_image=pil_im, dark_image=pil_im, size=(target_w, target_h))
                self.video_label.configure(image=ctk_img)
                self.video_label.image = ctk_img

            if cmd: self.on_action_triggered(cmd)
            if self.backend.attendance_mode: self._refresh_attendance_list()

            if self.pending_action:
                if self.backend.verify_state == 'success':
                    self._execute_pending_action()
                elif self.backend.verify_state == 'failed':
                    self._cancel_pending_action("验证失败，操作已取消")

        # 循环调用
        self.root.after(15, self.update_loop)

    def refresh_user_list(self):
        for item in self.user_tree.get_children(): self.user_tree.delete(item)
        for uid, uname in self.backend.get_user_list(): self.user_tree.insert("", "end", values=(uid, uname))

    def set_user_gesture(self):
        selected = self.user_tree.selection()
        if not selected:
            messagebox.showwarning("提示", "请先选中用户")
            return
        user_name = self.user_tree.item(selected[0], "values")[1]

        num_str = simpledialog.askstring("设置手势密码",
                                         f"请输入 {user_name} 的手势密码\n（1-5的整数，对应伸出手指数量）：",
                                         parent=self.root)
        if not num_str: return
        try:
            num = int(num_str)
        except ValueError:
            messagebox.showerror("错误", "请输入1-5的整数")
            return

        success, msg = self.backend.set_user_gesture(user_name, num)
        if success:
            messagebox.showinfo("提示", msg)
        else:
            messagebox.showerror("错误", msg)

    def delete_selected_user(self):
        selected = self.user_tree.selection()
        if not selected:
            messagebox.showwarning("提示", "请先选中要删除的用户")
            return
        user_name = self.user_tree.item(selected[0], "values")[1]
        if not messagebox.askyesno("确认操作", f"删除用户「{user_name}」需双因子验证\n是否开始验证？"): return

        success, msg = self.backend.start_verification(user_name)
        if not success:
            messagebox.showerror("错误", msg)
            return

        self.status_label.configure(text="SYS: VERIFYING...", text_color="#F39C12")
        self.pending_action = ('delete', user_name)

    def _execute_pending_action(self):
        action_type, target = self.pending_action
        if action_type == 'delete':
            success, msg = self.backend.delete_user(target)
            self.refresh_user_list()
            messagebox.showinfo("提示", msg)
        self.pending_action = None
        self.status_label.configure(text="SYS: STANDBY", text_color="#2ECC71")

    def _cancel_pending_action(self, msg):
        self.pending_action = None
        self.status_label.configure(text="SYS: STANDBY", text_color="#2ECC71")
        messagebox.showerror("操作取消", msg)

    def _refresh_attendance_list(self):
        for item in self.attendance_tree.get_children(): self.attendance_tree.delete(item)
        attendance_data = self.backend.get_attendance_list()

        for name, record in attendance_data.items():
            if isinstance(record, dict):
                in_time = record.get("in", "--:--:--")
                out_time = record.get("out", "--:--:--")
            else:
                in_time = record
                out_time = "--:--:--"
            self.attendance_tree.insert("", "end", values=(name, in_time, out_time))

        self.attendance_count_label.configure(text=f"TOTAL RECORDS: {len(attendance_data)}")

    def on_action_triggered(self, choice):
        # 拦截：如果当前已有挂起操作或正在执行后台线程，不响应新操作
        if self.pending_action or self.task_thread: return

        if choice == '3':
            success, msg = self.backend.toggle_attendance_mode()
            if self.backend.attendance_mode:
                self.status_label.configure(text="SYS: ATTENDANCE RUNNING", text_color="#E67E22")
                self.tabview.set("考勤日志")
            else:
                self.status_label.configure(text="SYS: STANDBY", text_color="#2ECC71")
            return

        if choice == '4':
            self.quit_app()
            return

        name = self.name_entry.get().strip()
        if choice == '1' and not name:
            messagebox.showwarning("提示", "请先填写姓名！")
            return

        # 启动后台线程执行核心任务，防止 UI 卡死
        self.status_label.configure(text="SYS: PROCESSING...", text_color="#F39C12")
        self.task_thread = threading.Thread(target=self._run_backend_task, args=(choice, name))
        self.task_thread.daemon = True
        self.task_thread.start()

    def _run_backend_task(self, choice, name):
        """后台实际调用的任务函数"""
        success, msg = self.backend.execute_task(choice, name)
        self.task_result = (success, msg, choice)

    def _on_task_finished(self, success, msg, choice):
        """后台任务完成后的主线程回调（恢复UI、弹窗提示）"""
        self.status_label.configure(text="SYS: STANDBY", text_color="#2ECC71")
        self.name_entry.delete(0, 'end')

        if success and choice in ('1', '2'):
            self.refresh_user_list()

        if not success:
            messagebox.showerror("系统提示", msg)
        else:
            messagebox.showinfo("系统提示", msg)

    def quit_app(self):
        self.backend.close_camera()
        self.root.quit()


if __name__ == '__main__':
    root = ctk.CTk()
    app = SmartTerminalUI(root)
    root.mainloop()