import cv2 as cv
import numpy as np
import os
import json
from PIL import Image, ImageDraw, ImageFont

# ========== 统一数据目录配置 ==========
DATA_ROOT = 'data'
dataset_path = os.path.join(DATA_ROOT, 'dataset')  # 人脸样本目录
trainer_path = os.path.join(DATA_ROOT, 'trainer')  # 训练模型目录
names_mapping_path = os.path.join(DATA_ROOT, 'names_mapping.json')  # 姓名映射
gesture_password_path = os.path.join(DATA_ROOT, 'gesture_passwords.json')  # 手势密码
attendance_records_path = os.path.join(DATA_ROOT, 'attendance_records.json')  # 考勤记录
model_file = os.path.join(trainer_path, 'trainer.yml')  # 模型文件完整路径

# ========== 人脸识别优化参数 ==========
FACE_WIDTH = 200
FACE_HEIGHT = 200
FACE_SIZE = (FACE_WIDTH, FACE_HEIGHT)
CONFIDENCE_THRESHOLD = 80  # 识别置信度阈值（越低越严格，参考工作正常的项目用80）


def is_face_quality_ok(gray_face):
    """检测人脸图像质量，仅用于提示"""
    h, w = gray_face.shape
    if h < 50 or w < 50:
        return False, f"人脸过小 ({h}x{w})"
    face_resized = cv.resize(gray_face, FACE_SIZE)
    laplacian_var = cv.Laplacian(face_resized, cv.CV_64F).var()
    if laplacian_var < 5:
        return False, f"严重模糊 ({laplacian_var:.1f})"
    return True, f"质量({laplacian_var:.1f})"


def preprocess_face(gray_face, target_size=FACE_SIZE):
    """标准化人脸预处理：缩放 + 直方图均衡化"""
    face = cv.resize(gray_face, target_size)
    face = cv.equalizeHist(face)
    return face


def init_data_environment():
    """初始化数据目录：自动创建文件夹 + 迁移旧版本根目录文件"""
    if not os.path.exists(DATA_ROOT):
        os.mkdir(DATA_ROOT)
    for path in [dataset_path, trainer_path]:
        if not os.path.exists(path):
            os.mkdir(path)

    old_file_map = {
        'names_mapping.json': names_mapping_path,
        'gesture_passwords.json': gesture_password_path,
        'attendance_records.json': attendance_records_path,
    }
    for old_name, new_path in old_file_map.items():
        old_path = os.path.join('.', old_name)
        if os.path.isfile(old_path) and not os.path.exists(new_path):
            try:
                os.rename(old_path, new_path)
                print(f"[自动迁移] {old_name} 已移动到 {DATA_ROOT} 目录")
            except Exception as e:
                print(f"[警告] {old_name} 迁移失败: {e}")

    old_dir_map = {
        'dataset': dataset_path,
        'trainer': trainer_path,
    }
    for old_name, new_path in old_dir_map.items():
        old_path = os.path.join('.', old_name)
        if os.path.isdir(old_path) and not os.path.exists(new_path):
            try:
                os.rename(old_path, new_path)
                print(f"[自动迁移] {old_name} 目录已移动到 {DATA_ROOT} 目录")
            except Exception as e:
                print(f"[警告] {old_name} 目录迁移失败: {e}")


def put_chinese_text(img, text, position, text_color=(0, 255, 0), font_size=20):
    """使用 PIL 在 OpenCV 图像上绘制中文"""
    cv2_im = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    draw = ImageDraw.Draw(pil_im)
    try:
        font = ImageFont.truetype("msyh.ttc", font_size, encoding="utf-8")
    except IOError:
        try:
            font = ImageFont.truetype("simhei.ttf", font_size, encoding="utf-8")
        except IOError:
            print("警告: 找不到中文字体文件，将使用默认字体。")
            font = ImageFont.load_default()
    b, g, r = text_color
    draw.text(position, text, font=font, fill=(r, g, b))
    return cv.cvtColor(np.array(pil_im), cv.COLOR_RGB2BGR)


def input_names():
    if os.path.exists(names_mapping_path):
        with open(names_mapping_path, 'r', encoding='utf-8') as f:
            names_mapping = json.load(f)
    else:
        names_mapping = {}
    new_id = max(names_mapping.values(), default=0) + 1
    name = input(f"请输入名字（ID将自动分配为{new_id}）：")
    names_mapping[name] = new_id
    with open(names_mapping_path, 'w', encoding='utf-8') as f:
        json.dump(names_mapping, f, ensure_ascii=False)
    return new_id


def check_face_duplicate(gray_face, threshold=60):
    """
    检测人脸是否已被注册（使用标准化预处理）
    """
    if not os.path.exists(model_file):
        return False, None, 999
    try:
        # 使用与训练一致的预处理
        face = preprocess_face(gray_face)
        recognizer = cv.face.LBPHFaceRecognizer_create()
        recognizer.read(model_file)
        face_id, confidence = recognizer.predict(face)

        if os.path.exists(names_mapping_path):
            with open(names_mapping_path, 'r', encoding='utf-8') as f:
                names_mapping = json.load(f)
            id_name = {v: k for k, v in names_mapping.items()}
            name = id_name.get(face_id, None)
        else:
            name = None

        if name is not None and confidence < threshold:
            return True, name, confidence
        return False, None, confidence
    except Exception as e:
        print(f"[校验失败] 重复人脸检测异常: {e}")
        return False, None, 999


def capture_faces(face_id, cam=None, win_name=None, show_preview=True, capture_callback=None):
    external_cam = cam is not None
    if not external_cam:
        cam = cv.VideoCapture(0)
        if not cam.isOpened():
            print("错误: 无法打开摄像头。")
            return False

    cam.set(3, 640)
    cam.set(4, 480)
    face_detector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    print("\n[信息] 正在初始化人脸捕捉。看着摄像头并等待...")

    current_name = None
    if os.path.exists(names_mapping_path):
        with open(names_mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        for name, uid in mapping.items():
            if uid == face_id:
                current_name = name
                break

    count = 0
    detect_interval = 5
    frame_count = 0
    window_name = win_name if win_name else 'image'

    while True:
        ret, img = cam.read()
        if not ret:
            break
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        if frame_count % detect_interval == 0:
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                # 只处理最大的人脸（避免多人干扰）
                (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])
                face_roi = gray[y:y + h, x:x + w]

                # 质量检测（仅提示不阻塞）
                quality_ok, q_msg = is_face_quality_ok(face_roi)
                if not quality_ok:
                    print(f"[质量提醒] {q_msg}，仍将保存")

                cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                count += 1

                # 统一尺寸后保存，后续训练/预测都用相同预处理
                face_normalized = cv.resize(face_roi, FACE_SIZE)
                cv.imwrite(os.path.join(dataset_path, f"User.{face_id}.{count}.jpg"), face_normalized)

                if capture_callback:
                    capture_callback(count, face_normalized)

                print(f"[采集进度] {count}/10 张 | {q_msg}")
                if count >= 10:
                    break

            if show_preview:
                cv.imshow(window_name, img)
                k = cv.waitKey(1) & 0xff
                if k == 27 or count >= 10:
                    break
            else:
                if count >= 10:
                    break

        frame_count += 1

    if not external_cam:
        cam.release()
        if show_preview:
            cv.destroyAllWindows()

    print(f"\n[信息] 人脸采集完成：共 {count} 张有效样本")
    return count >= 10


def train_model():
    recognizer = cv.face.LBPHFaceRecognizer_create()

    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []
        for imagePath in imagePaths:
            img = cv.imread(imagePath, cv.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[警告] 无法读取图片: {imagePath}，已跳过")
                continue
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            face = preprocess_face(img)
            faceSamples.append(face)
            ids.append(id)
        return faceSamples, ids, imagePaths

    print("\n[信息] 正在训练人脸识别模型。请稍候...")
    faces, ids, image_paths = getImagesAndLabels(dataset_path)

    unique_ids = np.unique(ids)
    if len(faces) < 5:
        print(f"\n[错误] 有效训练样本过少（仅 {len(faces)} 张），请重新采集")
        return
    print(f"[信息] 共加载 {len(faces)} 张人脸样本，涉及 {len(unique_ids)} 位用户")

    recognizer.train(faces, np.array(ids))
    recognizer.write(model_file)

    print("\n[信息] 正在校验人脸样本重复性...")
    duplicate_warn = []
    if os.path.exists(names_mapping_path):
        with open(names_mapping_path, 'r', encoding='utf-8') as f:
            names_mapping = json.load(f)
        id_name = {v: k for k, v in names_mapping.items()}
    else:
        id_name = {}

    for idx, face in enumerate(faces):
        pred_id, conf = recognizer.predict(face)
        real_id = ids[idx]
        if pred_id != real_id and conf < 55:
            real_name = id_name.get(real_id, f"ID{real_id}")
            pred_name = id_name.get(pred_id, f"ID{pred_id}")
            warn_msg = f"样本 {os.path.basename(image_paths[idx])} 与用户 [{pred_name}] 高度相似(置信度:{conf:.1f})"
            if warn_msg not in duplicate_warn:
                duplicate_warn.append(warn_msg)

    if duplicate_warn:
        print("\n[警告] 检测到疑似重复人脸样本：")
        for msg in duplicate_warn:
            print(f"  - {msg}")
        print("建议清理重复样本后重新训练。")
    else:
        print("[信息] 样本校验完成，未发现明显重复人脸。")

    print(f"\n[信息] {len(unique_ids)} 位用户的人脸已训练完成")


def recognize_faces():
    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.read(model_file)
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv.CascadeClassifier(cascadePath)
    if os.path.exists(names_mapping_path):
        with open(names_mapping_path, 'r', encoding='utf-8') as f:
            names_mapping = json.load(f)
        names = {v: k for k, v in names_mapping.items()}
    else:
        print("错误: 未找到名字映射文件。")
        return

    cam = cv.VideoCapture(0)
    if not cam.isOpened():
        print("错误: 无法打开摄像头。")
        return
    cam.set(3, 640)
    cam.set(4, 480)
    while True:
        ret, img = cam.read()
        if not ret:
            break
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            # 标准化预处理后再预测
            face_roi = gray[y:y + h, x:x + w]
            face_preprocessed = preprocess_face(face_roi)
            id, confidence = recognizer.predict(face_preprocessed)
            if confidence < CONFIDENCE_THRESHOLD:
                name = names.get(id, "未知")
            else:
                name = "未知"
            img = put_chinese_text(img, name, (x, max(0, y - 30)), text_color=(0, 255, 0), font_size=20)
        cv.imshow('camera', img)
        k = cv.waitKey(10) & 0xff
        if k == 27:
            break
    cam.release()
    cv.destroyAllWindows()


def main():
    init_data_environment()
    while True:
        print("请选择一个操作：\n1. 数据采集\n2. 模型训练\n3. 实时识别\n4. 退出")
        choice = input("输入你的选择: ")
        if choice == '1':
            capture_faces(input_names())
        elif choice == '2':
            train_model()
        elif choice == '3':
            recognize_faces()
        elif choice == '4':
            break
        else:
            print("无效的选择，请重试。")


if __name__ == '__main__':
    main()