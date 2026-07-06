import cv2 as cv
import numpy as np
import os
import json
from PIL import Image, ImageDraw, ImageFont

# ========== 统一数据目录配置 ==========
DATA_ROOT = 'data'
dataset_path = os.path.join(DATA_ROOT, 'dataset')       # 人脸样本目录
trainer_path = os.path.join(DATA_ROOT, 'trainer')       # 训练模型目录
names_mapping_path = os.path.join(DATA_ROOT, 'names_mapping.json')  # 姓名映射
gesture_password_path = os.path.join(DATA_ROOT, 'gesture_passwords.json')  # 手势密码
attendance_records_path = os.path.join(DATA_ROOT, 'attendance_records.json')  # 考勤记录
model_file = os.path.join(trainer_path, 'trainer.yml')  # 模型文件完整路径


def init_data_environment():
    """初始化数据目录：自动创建文件夹 + 迁移旧版本根目录文件"""
    # 创建根数据目录
    if not os.path.exists(DATA_ROOT):
        os.mkdir(DATA_ROOT)
    # 创建业务子目录
    for path in [dataset_path, trainer_path]:
        if not os.path.exists(path):
            os.mkdir(path)

    # 单个JSON文件自动迁移（根目录 -> data目录）
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

    # 文件夹自动迁移
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


def capture_faces(face_id):
    cam = cv.VideoCapture(0)
    if not cam.isOpened():
        print("错误: 无法打开摄像头。")
        return
    cam.set(3, 640)
    cam.set(4, 480)
    face_detector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    print("\n[信息] 正在初始化人脸捕捉。看着摄像头并等待...")
    count = 0
    detect_interval = 5
    frame_count = 0
    while True:
        ret, img = cam.read()
        if not ret:
            break
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if frame_count % detect_interval == 0:
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                count += 1
                cv.imwrite(os.path.join(dataset_path, f"User.{face_id}.{count}.jpg"), gray[y:y + h, x:x + w])
                if count >= 10:
                    break
            cv.imshow('image', img)
            k = cv.waitKey(1) & 0xff
            if k == 27 or count >= 10:
                break
        frame_count += 1
    cam.release()
    cv.destroyAllWindows()


def train_model():
    recognizer = cv.face.LBPHFaceRecognizer_create()
    detector = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)
        return faceSamples, ids

    print("\n[信息] 正在训练人脸识别模型。请稍候...")
    faces, ids = getImagesAndLabels(dataset_path)
    recognizer.train(faces, np.array(ids))
    recognizer.write(model_file)
    print(f"\n[信息] {len(np.unique(ids))} 张人脸已训练。程序结束")


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
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)
    while True:
        ret, img = cam.read()
        if not ret:
            break
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)))
        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if confidence < 100:
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