import threading
import os
import math
from tkinter import messagebox, ttk, simpledialog, filedialog
import customtkinter as ctk
import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from datetime import datetime, timedelta
from backend import TerminalBackend
from RK_face import put_chinese_text

# ==================== 主题 ====================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# ==================== 赛博配色方案 ====================
C = {
    "bg_deep":       "#080C18",  # 最深底色
    "bg_card":       "#0F1425",  # 卡片底色
    "bg_raised":     "#161D33",  # 悬浮层
    "bg_input":      "#0A0E1A",  # 输入框
    "cyan":          "#00E5FF",  # 主青色
    "cyan_dim":      "#006680",  # 暗青
    "purple":        "#9D4EDD",  # 紫色
    "purple_dim":    "#4A1A7A",  # 暗紫
    "green":         "#00FF88",  # 成功绿
    "green_dim":     "#006640",  # 暗绿
    "orange":        "#FF9F1C",  # 警告橙
    "red":           "#FF3860",  # 错误红
    "blue":          "#3B82F6",  # 信息蓝
    "text":          "#E8ECF4",  # 主文字
    "text_dim":      "#6B7394",  # 次要文字
    "text_muted":    "#3D4563",  # 暗文字
    "border":        "#1E2540",  # 边框
    "border_glow":   "#00E5FF33",  # 发光边框
    "white":         "#FFFFFF",
}

# 字体占位 — 等 root 创建后再初始化
FONT_TITLE = FONT_H2 = FONT_H3 = FONT_BODY = FONT_SMALL = None
FONT_MONO = FONT_MONO_BOLD = FONT_NUM = FONT_NUM_MD = None


def _init_fonts():
    """在 root 创建后初始化字体"""
    global FONT_TITLE, FONT_H2, FONT_H3, FONT_BODY, FONT_SMALL
    global FONT_MONO, FONT_MONO_BOLD, FONT_NUM, FONT_NUM_MD
    FONT_TITLE  = ctk.CTkFont(family="Segoe UI", size=28, weight="bold")
    FONT_H2     = ctk.CTkFont(family="Segoe UI", size=18, weight="bold")
    FONT_H3     = ctk.CTkFont(family="Segoe UI", size=14, weight="bold")
    FONT_BODY   = ctk.CTkFont(family="Segoe UI", size=13)
    FONT_SMALL  = ctk.CTkFont(family="Segoe UI", size=11)
    FONT_MONO   = ctk.CTkFont(family="Consolas", size=12)
    FONT_MONO_BOLD = ctk.CTkFont(family="Consolas", size=13, weight="bold")
    FONT_NUM    = ctk.CTkFont(family="Segoe UI", size=32, weight="bold")
    FONT_NUM_MD = ctk.CTkFont(family="Segoe UI", size=22, weight="bold")


class SmartTerminalUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face2Hand V3.0 · 智能考勤终端")
        self.root.geometry("1320x800")
        self.root.minsize(1200, 720)
        self.root.configure(fg_color=C["bg_deep"])

        self.backend = TerminalBackend()
        if not self.backend.open_camera():
            messagebox.showerror("错误", "无法打开摄像头！")
            self.root.quit()
            return

        self.pending_action = None
        self.task_thread = None
        self.task_result = None
        self.current_task = None
        self._attendance_view_date = None
        self._attendance_auto_refresh = True
        self._pulse_phase = 0.0  # 动画相位

        self.setup_ui()
        self.refresh_user_list()
        self.refresh_console()
        self._animate_pulse()
        self.update_loop()

    # ── 组件工厂 ──
    def _make_card(self, parent, corner_radius=12, fg_color=None, border_color=None, border_width=1, **kwargs):
        fg = fg_color if fg_color is not None else C["bg_card"]
        bc = border_color if border_color is not None else C["border"]
        return ctk.CTkFrame(parent, fg_color=fg, corner_radius=corner_radius,
                            border_width=border_width, border_color=bc, **kwargs)

    def _make_btn(self, parent, text, command=None, accent="cyan", size="md", corner_radius=8, **kwargs):
        colors = {
            "cyan":   (C["cyan_dim"], C["cyan"], C["bg_deep"]),
            "purple": (C["purple_dim"], C["purple"], C["white"]),
            "green":  (C["green_dim"], C["green"], C["bg_deep"]),
            "orange": (C["orange"], "#E08500", C["bg_deep"]),
            "red":    (C["red"], "#D03050", C["white"]),
            "blue":   (C["blue"], "#2563EB", C["white"]),
        }
        fg, hover, txt_color = colors.get(accent, colors["cyan"])
        h_map = {"sm": 30, "md": 38, "lg": 46}
        f_map = {"sm": FONT_SMALL, "md": FONT_BODY, "lg": FONT_H3}
        return ctk.CTkButton(parent, text=text, command=command,
                             fg_color=fg, hover_color=hover, text_color=txt_color,
                             height=h_map.get(size, 38), corner_radius=corner_radius,
                             font=f_map.get(size, FONT_BODY), **kwargs)

    def _make_kpi_card(self, parent, label, value, accent_color, icon=None):
        card = self._make_card(parent, corner_radius=10)
        header = ctk.CTkFrame(card, fg_color="transparent", height=4)
        header.pack(fill="x")
        accent_bar = ctk.CTkFrame(header, fg_color=accent_color, height=3, corner_radius=0)
        accent_bar.pack(fill="x")
        val_label = ctk.CTkLabel(card, text=str(value), font=FONT_NUM_MD,
                                 text_color=accent_color)
        val_label.pack(pady=(10, 0))
        lbl = ctk.CTkLabel(card, text=label, font=FONT_SMALL, text_color=C["text_dim"])
        lbl.pack(pady=(0, 10))
        return card, val_label

    # ════════════════════════════════════════════
    #  UI 布局
    # ════════════════════════════════════════════
    def setup_ui(self):
        self.root.grid_columnconfigure(0, weight=6)
        self.root.grid_columnconfigure(1, weight=4)
        self.root.grid_rowconfigure(0, weight=1)

        # ── 左侧 · 摄像头 ──
        self._build_camera_panel()
        # ── 右侧 · 控制面板 ──
        self._build_control_panel()

    # ════════════════════════════════════════════
    #  左侧 - 摄像头面板
    # ════════════════════════════════════════════
    def _build_camera_panel(self):
        self.cam_panel = ctk.CTkFrame(self.root, fg_color="transparent")
        self.cam_panel.grid(row=0, column=0, padx=(20, 10), pady=20, sticky="nsew")
        self.cam_panel.grid_columnconfigure(0, weight=1)
        self.cam_panel.grid_rowconfigure(0, weight=0)
        self.cam_panel.grid_rowconfigure(1, weight=1)
        self.cam_panel.grid_rowconfigure(2, weight=0)

        # 标题栏
        title_bar = ctk.CTkFrame(self.cam_panel, fg_color="transparent")
        title_bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        # 脉冲指示点 (用 Canvas 手动绘制)
        self.pulse_canvas = ctk.CTkCanvas(title_bar, width=14, height=14,
                                           bg=C["bg_deep"], highlightthickness=0)
        self.pulse_canvas.pack(side="left", padx=(2, 8))
        self.pulse_dot = self.pulse_canvas.create_oval(2, 2, 12, 12, fill=C["green"], outline="")

        ctk.CTkLabel(title_bar, text="LIVE FEED",
                     font=FONT_MONO_BOLD, text_color=C["cyan"]).pack(side="left")
        self.fps_label = ctk.CTkLabel(title_bar, text="",
                                      font=FONT_MONO, text_color=C["text_dim"])
        self.fps_label.pack(side="right")
        ctk.CTkLabel(title_bar, text="●", font=FONT_SMALL,
                     text_color=C["green"]).pack(side="right", padx=(0, 4))

        # 视频框 (发光边框效果用两层模拟)
        self.video_outer = ctk.CTkFrame(self.cam_panel, fg_color=C["cyan"],
                                        corner_radius=14)
        self.video_outer.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        self.video_inner = ctk.CTkFrame(self.video_outer, fg_color=C["bg_deep"],
                                        corner_radius=12)
        self.video_inner.pack(fill="both", expand=True, padx=2, pady=2)

        self.video_label = ctk.CTkLabel(self.video_inner, text="")
        self.video_label.pack(expand=True, padx=8, pady=8)

        # 底部状态条
        self.status_bar = ctk.CTkFrame(self.cam_panel, fg_color=C["bg_card"],
                                       corner_radius=6, height=32)
        self.status_bar.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        self.status_bar.grid_propagate(False)

        self.status_icon = ctk.CTkLabel(self.status_bar, text="◉", font=FONT_SMALL,
                                        text_color=C["green"], width=20)
        self.status_icon.pack(side="left", padx=(10, 0))
        self.status_text = ctk.CTkLabel(self.status_bar, text="STANDBY",
                                        font=FONT_MONO_BOLD, text_color=C["text"])
        self.status_text.pack(side="left", padx=(4, 0))

    # ════════════════════════════════════════════
    #  右侧 - 控制面板
    # ════════════════════════════════════════════
    def _build_control_panel(self):
        self.ctrl_panel = ctk.CTkFrame(self.root, fg_color="transparent")
        self.ctrl_panel.grid(row=0, column=1, padx=(10, 20), pady=20, sticky="nsew")
        self.ctrl_panel.grid_rowconfigure(0, weight=0)
        self.ctrl_panel.grid_rowconfigure(1, weight=1)
        self.ctrl_panel.grid_columnconfigure(0, weight=1)

        # ── Logo 区 ──
        logo_card = self._make_card(self.ctrl_panel, corner_radius=10)
        logo_card.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        logo_inner = ctk.CTkFrame(logo_card, fg_color="transparent")
        logo_inner.pack(fill="x", padx=16, pady=12)
        ctk.CTkLabel(logo_inner, text="Face2Hand", font=FONT_TITLE,
                     text_color=C["cyan"]).pack(side="left")
        ver_badge = ctk.CTkFrame(logo_inner, fg_color=C["purple_dim"], corner_radius=10)
        ver_badge.pack(side="right")
        ctk.CTkLabel(ver_badge, text="V3.0", font=FONT_SMALL,
                     text_color=C["purple"]).pack(padx=10, pady=2)

        # ── TabView ──
        self.tabview = ctk.CTkTabview(self.ctrl_panel,
                                      fg_color="transparent",
                                      segmented_button_fg_color=C["bg_card"],
                                      segmented_button_selected_color=C["cyan"],
                                      segmented_button_selected_hover_color=C["cyan_dim"],
                                      segmented_button_unselected_color=C["bg_raised"],
                                      segmented_button_unselected_hover_color=C["border"])
        self.tabview.grid(row=1, column=0, sticky="nsew")

        self.tabview.add("  操 作 台  ")
        self.tabview.add("  用 户 库  ")
        self.tabview.add("  打 卡 控 制 台  ")

        self._build_operate_tab()
        self._build_user_tab()
        self._build_console_tab()

    # ════════════════════════════════════════════
    #  操作台
    # ════════════════════════════════════════════
    def _build_operate_tab(self):
        tab = self.tabview.tab("  操 作 台  ")

        # 姓名输入卡片
        entry_card = self._make_card(tab)
        entry_card.pack(fill="x", padx=6, pady=(8, 6))
        ctk.CTkLabel(entry_card, text="▸ 用户登记", font=FONT_H3,
                     text_color=C["text"], anchor="w").pack(anchor="w", padx=14, pady=(12, 6))
        self.name_entry = ctk.CTkEntry(entry_card, placeholder_text="输入姓名...",
                                       height=40, corner_radius=8,
                                       border_color=C["border"],
                                       fg_color=C["bg_input"],
                                       text_color=C["text"],
                                       font=FONT_BODY)
        self.name_entry.pack(fill="x", padx=14, pady=(0, 12))

        # 功能按钮卡片
        btn_card = self._make_card(tab)
        btn_card.pack(fill="x", padx=6, pady=4)
        ctk.CTkLabel(btn_card, text="▸ 功能操作", font=FONT_H3,
                     text_color=C["text"], anchor="w").pack(anchor="w", padx=14, pady=(12, 4))

        btn_row1 = ctk.CTkFrame(btn_card, fg_color="transparent")
        btn_row1.pack(fill="x", padx=14, pady=(4, 6))
        self._make_btn(btn_row1, "数据采集", command=lambda: self.on_action_triggered('1'),
                  accent="cyan").pack(side="left", fill="x", expand=True, padx=(0, 4))
        self._make_btn(btn_row1, "模型训练", command=lambda: self.on_action_triggered('2'),
                  accent="purple").pack(side="right", fill="x", expand=True, padx=(4, 0))

        self.attendance_btn = self._make_btn(btn_card, "进入考勤模式",
                                        command=lambda: self.on_action_triggered('3'),
                                        accent="orange", size="lg")
        self.attendance_btn.pack(fill="x", padx=14, pady=(4, 6))

        # 手势指南
        guide_card = self._make_card(btn_card, fg_color=C["bg_raised"], corner_radius=8)
        guide_card.pack(fill="x", padx=14, pady=(2, 12))
        hints = [
            ("1 指", "签到"),
            ("2 指", "签退"),
            ("4 指长按", "唤出菜单"),
            ("5 指长按", "退出系统"),
            ("1/2/3 指", "导航/确认"),
        ]
        for gesture, action in hints:
            row = ctk.CTkFrame(guide_card, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=1)
            ctk.CTkLabel(row, text=gesture, font=FONT_MONO_BOLD,
                         text_color=C["cyan"], width=65, anchor="e").pack(side="left")
            ctk.CTkLabel(row, text=action, font=FONT_SMALL,
                         text_color=C["text_dim"]).pack(side="left", padx=8)

        # 退出
        exit_row = ctk.CTkFrame(tab, fg_color="transparent")
        exit_row.pack(side="bottom", fill="x", padx=6, pady=(0, 8))
        self._make_btn(exit_row, "退出系统", command=self.quit_app,
                  accent="red", size="sm").pack(side="right")

    # ════════════════════════════════════════════
    #  用户库
    # ════════════════════════════════════════════
    def _build_user_tab(self):
        tab = self.tabview.tab("  用 户 库  ")
        self._style_treeview()

        # KPI
        kpi_row = ctk.CTkFrame(tab, fg_color="transparent")
        kpi_row.pack(fill="x", padx=6, pady=(8, 4))
        self.user_kpi_card, self.user_kpi_val = self._make_kpi_card(
            kpi_row, "已注册用户", "0", C["cyan"], "👤")
        self.user_kpi_card.pack(fill="x", padx=0)

        # 表格
        tree_card = self._make_card(tab)
        tree_card.pack(fill="both", expand=True, padx=6, pady=4)

        self.user_tree = ttk.Treeview(tree_card,
                                      columns=("id", "name", "gesture"),
                                      show="headings", height=10)
        self.user_tree.heading("id", text="UID")
        self.user_tree.heading("name", text="姓名")
        self.user_tree.heading("gesture", text="手势")
        self.user_tree.column("id", width=50, anchor="center")
        self.user_tree.column("name", width=110, anchor="center")
        self.user_tree.column("gesture", width=70, anchor="center")
        self.user_tree.pack(side="left", fill="both", expand=True, padx=2, pady=2)

        sb = ttk.Scrollbar(tree_card, orient="vertical", command=self.user_tree.yview)
        sb.pack(side="right", fill="y")
        self.user_tree.configure(yscrollcommand=sb.set)

        # 按钮
        btn_bar = ctk.CTkFrame(tab, fg_color="transparent")
        btn_bar.pack(fill="x", padx=6, pady=(4, 6))
        self._make_btn(btn_bar, "刷新", command=self.refresh_user_list,
                  accent="blue", size="sm").pack(side="left", padx=2)
        self._make_btn(btn_bar, "手势密码", command=self.set_user_gesture,
                  accent="purple", size="sm").pack(side="left", padx=2)
        self._make_btn(btn_bar, "删除用户", command=self.delete_selected_user,
                  accent="red", size="sm").pack(side="right", padx=2)

    # ════════════════════════════════════════════
    #  打卡控制台
    # ════════════════════════════════════════════
    def _build_console_tab(self):
        tab = self.tabview.tab("  打 卡 控 制 台  ")

        # ── 今日概览 KPI ──
        kpi_frame = ctk.CTkFrame(tab, fg_color="transparent")
        kpi_frame.pack(fill="x", padx=6, pady=(8, 4))
        kpi_frame.grid_columnconfigure((0, 1, 2), weight=1, uniform="kpi")

        self.kpi_in, self.kpi_in_val = self._make_kpi_card(kpi_frame, "已签到", "0", C["green"])
        self.kpi_in.grid(row=0, column=0, padx=(0, 3), sticky="ew")

        self.kpi_out, self.kpi_out_val = self._make_kpi_card(kpi_frame, "已签退", "0", C["orange"])
        self.kpi_out.grid(row=0, column=1, padx=3, sticky="ew")

        self.kpi_users, self.kpi_users_val = self._make_kpi_card(kpi_frame, "总人数", "0", C["cyan"])
        self.kpi_users.grid(row=0, column=2, padx=(3, 0), sticky="ew")

        # 日期标题
        self.today_title = ctk.CTkLabel(tab, text="", font=FONT_H3,
                                        text_color=C["text_dim"], anchor="w")
        self.today_title.pack(anchor="w", padx=10, pady=(4, 2))

        # ── 操作栏 ──
        action_card = self._make_card(tab, corner_radius=8)
        action_card.pack(fill="x", padx=6, pady=4)
        action_inner = ctk.CTkFrame(action_card, fg_color="transparent")
        action_inner.pack(fill="x", padx=10, pady=8)

        ctk.CTkLabel(action_inner, text="查看日期", font=FONT_SMALL,
                     text_color=C["text_dim"]).pack(side="left", padx=(0, 6))
        self.date_combo = ctk.CTkComboBox(action_inner, values=["今天"], width=130, height=30,
                                          corner_radius=6,
                                          dropdown_fg_color=C["bg_card"],
                                          dropdown_text_color=C["text"],
                                          dropdown_hover_color=C["bg_raised"],
                                          button_color=C["cyan_dim"],
                                          button_hover_color=C["cyan"],
                                          text_color=C["text"],
                                          font=FONT_SMALL,
                                          command=self._on_date_selected)
        self.date_combo.set("今天")
        self.date_combo.pack(side="left", padx=4)

        self._make_btn(action_inner, "今天", command=lambda: self._set_date("今天"),
                  accent="blue", size="sm").pack(side="left", padx=4)

        ctk.CTkLabel(action_inner, text="").pack(side="left", padx=10)

        self._make_btn(action_inner, "导出 CSV", command=self._export_csv,
                  accent="green", size="sm").pack(side="right", padx=2)
        self._make_btn(action_inner, "刷新", command=self.refresh_console,
                  accent="blue", size="sm").pack(side="right", padx=2)

        # ── 考勤表格 ──
        table_card = self._make_card(tab)
        table_card.pack(fill="both", expand=True, padx=6, pady=4)

        self.attendance_tree = ttk.Treeview(table_card,
                                            columns=("name", "sign_in", "sign_out", "status"),
                                            show="headings", height=12)
        self.attendance_tree.heading("name", text="姓名")
        self.attendance_tree.heading("sign_in", text="签到")
        self.attendance_tree.heading("sign_out", text="签退")
        self.attendance_tree.heading("status", text="状态")
        self.attendance_tree.column("name", width=60, anchor="center")
        self.attendance_tree.column("sign_in", width=68, anchor="center")
        self.attendance_tree.column("sign_out", width=68, anchor="center")
        self.attendance_tree.column("status", width=60, anchor="center")
        self.attendance_tree.pack(side="left", fill="both", expand=True, padx=2, pady=2)

        ts = ttk.Scrollbar(table_card, orient="vertical", command=self.attendance_tree.yview)
        ts.pack(side="right", fill="y")
        self.attendance_tree.configure(yscrollcommand=ts.set)

        # 底部计数
        bottom = ctk.CTkFrame(tab, fg_color="transparent")
        bottom.pack(fill="x", padx=6, pady=(3, 6))
        self.attendance_count_label = ctk.CTkLabel(bottom, text="记录: 0 条",
                                                   font=FONT_MONO_BOLD,
                                                   text_color=C["cyan"])
        self.attendance_count_label.pack(side="left")

    # ════════════════════════════════════════════
    #  Treeview 样式
    # ════════════════════════════════════════════
    def _style_treeview(self):
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview",
                        background=C["bg_card"],
                        foreground=C["text"],
                        rowheight=32,
                        fieldbackground=C["bg_card"],
                        borderwidth=0,
                        font=("Segoe UI", 10))
        style.map("Treeview",
                  background=[("selected", C["cyan_dim"])],
                  foreground=[("selected", C["white"])])
        style.configure("Treeview.Heading",
                        background=C["bg_raised"],
                        foreground=C["text"],
                        relief="flat",
                        font=("Segoe UI", 10, "bold"),
                        borderwidth=0,
                        padding=(0, 4))
        style.map("Treeview.Heading",
                  background=[("active", C["border"])])

    # ════════════════════════════════════════════
    #  脉冲动画
    # ════════════════════════════════════════════
    def _animate_pulse(self):
        self._pulse_phase += 0.08
        if self._pulse_phase > math.pi * 2:
            self._pulse_phase -= math.pi * 2

        # 脉冲点大小振荡 (6~10px)
        r = 5 + int(math.sin(self._pulse_phase) * 2)
        self.pulse_canvas.coords(self.pulse_dot, 7 - r, 7 - r, 7 + r, 7 + r)
        # 颜色在绿和青之间切换
        t = (math.sin(self._pulse_phase) + 1) / 2
        if self.backend.attendance_mode:
            r_col = int(255 * t + 255 * (1 - t))
            g_col = int(159 * t + 56 * (1 - t))
            b_col = int(28 * t + 0 * (1 - t))
        else:
            r_col = int(0 * t + 0 * (1 - t))
            g_col = int(255 * t + 200 * (1 - t))
            b_col = int(136 * t + 100 * (1 - t))
        color = f'#{r_col:02x}{g_col:02x}{b_col:02x}'
        self.pulse_canvas.itemconfig(self.pulse_dot, fill=color)

        self.root.after(50, self._animate_pulse)

    # ════════════════════════════════════════════
    #  Loading 画面
    # ════════════════════════════════════════════
    def _show_loading_frame(self):
        if self.current_task == '1' and hasattr(self.backend, 'live_frame') and self.backend.live_frame is not None:
            img = self.backend.live_frame.copy()
            faces = list(self.backend.captured_faces)
            # 半透明渐变覆盖
            overlay = img.copy()
            cv.rectangle(overlay, (0, 0), (img.shape[1], 55), (8, 12, 24), -1)
            cv.addWeighted(overlay, 0.85, img, 0.15, 0, img)
            img = put_chinese_text(img, f"采集人脸 [{len(faces)}/10]", (200, 12),
                                   text_color=(0, 229, 255), font_size=22)
            # 底部进度
            for i in range(10):
                px = 50 + i * 55
                py = img.shape[0] - 80
                w, h = 48, 48
                if i < len(faces):
                    face_gray = faces[i]
                    face_rz = cv.resize(face_gray, (w, h))
                    face_bgr = cv.cvtColor(face_rz, cv.COLOR_GRAY2BGR)
                    img[py:py + h, px:px + w] = face_bgr
                    cv.rectangle(img, (px, py), (px + w, py + h), (0, 229, 255), 1)
                else:
                    cv.rectangle(img, (px, py), (px + w, py + h), C_color_to_bgr(C["border"]), 1)
            img = put_chinese_text(img, "请面对摄像头，自动采集 10 张人脸", (130, img.shape[0] - 26),
                                   text_color=C_bgr(C["text_dim"]), font_size=14)
        elif self.current_task == '1':
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            faces = list(self.backend.captured_faces)
            img = put_chinese_text(img, f"采集人脸 [{len(faces)}/10]", (220, 30),
                                   text_color=(0, 229, 255), font_size=24)
            for i in range(10):
                px = 50 + i * 55
                py = 100
                w, h = 48, 48
                cv.rectangle(img, (px, py), (px + w, py + h), C_color_to_bgr(C["border"]), 1)
                if i < len(faces):
                    fg = faces[i]
                    fr = cv.resize(fg, (w, h))
                    fb = cv.cvtColor(fr, cv.COLOR_GRAY2BGR)
                    img[py:py + h, px:px + w] = fb
                    cv.rectangle(img, (px, py), (px + w, py + h), (0, 229, 255), 1)
        else:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            img = put_chinese_text(img, "模型训练中...", (220, 200),
                                   text_color=(157, 78, 221), font_size=36)
            img = put_chinese_text(img, "请勿关闭系统", (240, 260),
                                   text_color=C_bgr(C["text_dim"]), font_size=22)

        self._render_frame(img)

    # ════════════════════════════════════════════
    #  主循环
    # ════════════════════════════════════════════
    _frame_times = []

    def update_loop(self):
        if self.task_thread and not self.task_thread.is_alive():
            success, msg, choice = self.task_result
            self.task_thread = None
            self.task_result = None
            self._on_task_finished(success, msg, choice)

        if self.backend.is_busy or self.task_thread:
            self._show_loading_frame()
        else:
            img, cmd = self.backend.get_frame_and_gesture()
            if img is not None:
                self._render_frame(img)
                now = datetime.now()
                self._frame_times.append(now)
                self._frame_times = [t for t in self._frame_times if (now - t).total_seconds() < 1]
                if self._frame_times:
                    fps = len(self._frame_times)
                    self.fps_label.configure(text=f"{fps} FPS")

            if cmd:
                self.on_action_triggered(cmd)
            if self.backend.attendance_mode:
                self._refresh_attendance_list()
                if self._attendance_view_date is None:
                    self.refresh_console()

            if self.pending_action:
                if self.backend.verify_state == 'success':
                    self._execute_pending_action()
                elif self.backend.verify_state == 'failed':
                    self._cancel_pending_action("验证失败，操作已取消")

        self.root.after(15, self.update_loop)

    def _render_frame(self, img):
        img_h, img_w, _ = img.shape
        target_w = 620
        target_h = int(img_h * (target_w / img_w))
        cv2_im = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        ctk_img = ctk.CTkImage(light_image=pil_im, dark_image=pil_im, size=(target_w, target_h))
        self.video_label.configure(image=ctk_img)
        self.video_label.image = ctk_img

    # ════════════════════════════════════════════
    #  用户管理
    # ════════════════════════════════════════════
    def refresh_user_list(self):
        for item in self.user_tree.get_children():
            self.user_tree.delete(item)
        users = self.backend.get_user_list()
        for uid, uname in users:
            gesture = self.backend.gesture_passwords.get(uname, 1)
            self.user_tree.insert("", "end", values=(uid, uname, f"{gesture} 指"))
        self.user_kpi_val.configure(text=str(len(users)))

    def set_user_gesture(self):
        selected = self.user_tree.selection()
        if not selected:
            messagebox.showwarning("提示", "请先选中用户")
            return
        user_name = self.user_tree.item(selected[0], "values")[1]
        num_str = simpledialog.askstring("手势密码",
                                         f"请输入 {user_name} 的手势密码\n(1-5的整数)：",
                                         parent=self.root)
        if not num_str:
            return
        try:
            num = int(num_str)
        except ValueError:
            messagebox.showerror("错误", "请输入 1-5 的整数")
            return
        success, msg = self.backend.set_user_gesture(user_name, num)
        if success:
            messagebox.showinfo("提示", msg)
            self.refresh_user_list()
        else:
            messagebox.showerror("错误", msg)

    def delete_selected_user(self):
        selected = self.user_tree.selection()
        if not selected:
            messagebox.showwarning("提示", "请先选中要删除的用户")
            return
        user_name = self.user_tree.item(selected[0], "values")[1]
        if not messagebox.askyesno("确认操作",
                                   f"删除用户「{user_name}」需双因子验证\n是否开始验证？"):
            return
        success, msg = self.backend.start_verification(user_name)
        if not success:
            messagebox.showerror("错误", msg)
            return
        self._set_status("VERIFYING", C["orange"])
        self.pending_action = ('delete', user_name)

    def _execute_pending_action(self):
        action_type, target = self.pending_action
        if action_type == 'delete':
            success, msg = self.backend.delete_user(target)
            self.refresh_user_list()
            messagebox.showinfo("提示", msg)
        self.pending_action = None
        self._set_status("STANDBY", C["green"])

    def _cancel_pending_action(self, msg):
        self.pending_action = None
        self._set_status("STANDBY", C["green"])
        messagebox.showerror("操作取消", msg)

    def _set_status(self, text, color):
        self.status_text.configure(text=text, text_color=color)
        self.status_icon.configure(text_color=color)

    # ════════════════════════════════════════════
    #  考勤列表
    # ════════════════════════════════════════════
    def _refresh_attendance_list(self):
        for item in self.attendance_tree.get_children():
            self.attendance_tree.delete(item)

        if self._attendance_view_date:
            data = self.backend.get_attendance_by_date(self._attendance_view_date)
        else:
            data = self.backend.get_attendance_list()

        for name, record in data.items():
            if isinstance(record, dict):
                in_t = record.get("in", "--:--:--")
                out_t = record.get("out", "--:--:--")
            else:
                in_t = record if record != "--:--:--" else "--:--:--"
                out_t = "--:--:--"

            if in_t == "--:--:--" and out_t == "--:--:--":
                status, sc = "未打卡", "#555566"
            elif in_t != "--:--:--" and out_t != "--:--:--":
                status, sc = "完成", C["green"]
            elif in_t != "--:--:--":
                status, sc = "已签到", C["blue"]
            else:
                status, sc = "仅签退", C["orange"]

            item_id = self.attendance_tree.insert("", "end", values=(name, in_t, out_t, status))
            self.attendance_tree.tag_configure(sc, foreground=sc)
            self.attendance_tree.item(item_id, tags=(sc,))

        prefix = "本日" if self._attendance_view_date is None else self._attendance_view_date
        self.attendance_count_label.configure(text=f"{prefix} · {len(data)} 条记录")

    # ════════════════════════════════════════════
    #  打卡控制台刷新
    # ════════════════════════════════════════════
    def refresh_console(self):
        summary = self.backend.get_attendance_summary()
        ti = summary["today"]
        now = datetime.now()
        wd = ["一","二","三","四","五","六","日"][now.weekday()]
        self.today_title.configure(text=f"{now.strftime('%Y年%m月%d日')} 星期{wd} · 今日概览")

        self.kpi_in_val.configure(text=str(ti["checked_in"]))
        self.kpi_out_val.configure(text=str(ti["checked_out"]))
        self.kpi_users_val.configure(text=str(summary["total_users"]))

        dates = self.backend.get_attendance_dates()
        today = now.strftime("%Y-%m-%d")
        opts = ["今天"]
        if today not in dates:
            opts.append(today)
        opts += dates
        cur = self.date_combo.get()
        self.date_combo.configure(values=opts)
        if self._attendance_view_date is None:
            self.date_combo.set("今天")
        elif cur not in opts:
            self.date_combo.set("今天")
            self._attendance_view_date = None

        self._refresh_attendance_list()

    def _set_date(self, d):
        self._attendance_view_date = None if d == "今天" else d
        self.date_combo.set(d)
        self._refresh_attendance_list()

    def _on_date_selected(self, choice):
        self._attendance_view_date = None if choice == "今天" else choice
        self._refresh_attendance_list()

    def _export_csv(self):
        fp = filedialog.asksaveasfilename(
            title="导出考勤记录",
            defaultextension=".csv",
            filetypes=[("CSV 文件", "*.csv")],
            initialfile=f"考勤_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        if not fp:
            return
        ok, msg = self.backend.export_attendance_csv(fp)
        if ok:
            messagebox.showinfo("导出成功", f"已导出到:\n{msg}")
        else:
            messagebox.showerror("导出失败", msg)

    # ════════════════════════════════════════════
    #  操作触发
    # ════════════════════════════════════════════
    def on_action_triggered(self, choice):
        if self.pending_action or self.task_thread:
            return
        if choice == '3':
            ok, msg = self.backend.toggle_attendance_mode()
            if self.backend.attendance_mode:
                self._set_status("ATTENDANCE", C["orange"])
                self.attendance_btn.configure(text="退出考勤模式", fg_color=C["red"],
                                              hover_color="#D03050")
                self.tabview.set("  打 卡 控 制 台  ")
            else:
                self._set_status("STANDBY", C["green"])
                self.attendance_btn.configure(text="进入考勤模式", fg_color=C["orange"],
                                              hover_color="#E08500")
            return
        if choice == '4':
            self.quit_app()
            return
        name = self.name_entry.get().strip()
        if choice == '1' and not name:
            messagebox.showwarning("提示", "请先填写姓名！")
            return
        self._set_status("PROCESSING", C["orange"])
        self.current_task = choice
        self.task_thread = threading.Thread(target=self._run_backend_task, args=(choice, name))
        self.task_thread.daemon = True
        self.task_thread.start()

    def _run_backend_task(self, choice, name):
        success, msg = self.backend.execute_task(choice, name)
        self.task_result = (success, msg, choice)

    def _on_task_finished(self, success, msg, choice):
        self.current_task = None
        self._set_status("STANDBY", C["green"])
        self.name_entry.delete(0, 'end')
        if success and choice in ('1', '2'):
            self.refresh_user_list()
        fn = messagebox.showerror if not success else messagebox.showinfo
        fn("系统提示", msg)

    def quit_app(self):
        self.backend.close_camera()
        self.root.quit()


# ════════════════════════════════════════════
#  辅助函数
# ════════════════════════════════════════════
def C_color_to_bgr(hex_color):
    """#RRGGBB → (B, G, R)"""
    h = hex_color.lstrip('#')
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))

def C_bgr(hex_color):
    return C_color_to_bgr(hex_color)


if __name__ == '__main__':
    root = ctk.CTk()
    _init_fonts()
    app = SmartTerminalUI(root)
    root.mainloop()
