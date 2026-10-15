import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# ==================================================
# ⚙️ PATHS & CONFIGURATION (الإعدادات والمسارات)
# ==================================================
INPUT_VIDEO = r"E:\elecrto pi\co\cam5.avi"
OUTPUT_VIDEO = r"E:\elecrto pi\co\cam5allcode_output.mp4"
# الموديلات المستعملة
MODEL1_PATH = r"bestv999.pt"   # الموديل الأول: كشف وقص المنطقة (الكود الثاني)
MODEL2_PATH = r"bestv1.11.pt"  # الموديل الثاني: الكشف عن (fol و hol) على القص (الكود الثالث)

# إعدادات الموديل الأول (Model 1)
MODEL1_CONF = 0.60
TARGET_CLASS1 = "cou"

# إعدادات الموديل الثاني (Model 2)
MODEL2_CONF = 0.40
MODEL2_IOU = 0.25
AGNOSTIC_NMS = True
TARGET_CLASSES2 = None  # ترك جميع الكلاسات الخاصة بالموديل الثاني بدون فلترة لضمان كشف كل شيء (fol, hol, cou...)

# إعدادات نافذة المعاينة المصغرة (Overlay منتصف اليسار)
PIP_SCALE_HEIGHT = 0.65 # جعل ارتفاع الفيديو المصغر = 65% من ارتفاع الفريم الرئيسي
PIP_PADDING = 15        # المسافة من حافة الفريم الرئيسي

# أبعاد المعاينة
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# سرعة تشغيل الفيديو النهائي والـ Live (تلقائياً None لتحديد سرعة الفيديو الحقيقية بالضبط)
TARGET_FPS = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF = (DEVICE == "cuda")  # تسريع الاستنتاج للضعف على كروت الشاشة

# ==================================================
# 📐 CALIBRATION MATRIX (قيم المعايرة للكود الأول)
# ==================================================
CAMERA_MATRIX = np.array([
    [8.8429467006e+03, 0.0000000000e+00, 1.9760256316e+03],
    [0.0000000000e+00, 8.0614318748e+03, 1.0553053205e+03],
    [0.0000000000e+00, 0.0000000000e+00, 1.0000000000e+00],
])

DIST_COEFFS = np.array([[
    -4.4506662832e+00,
     5.7302295871e+01,
    -7.6896190630e-02,
     9.8996868369e-02,
    -2.9595574195e+02,
]])


def build_undistort_maps(mtx, dist, width, height):
    """تعديل أبعاد العدسة وتصحيح التشويه (Resize & Undistort)"""
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (width, height), 1, (width, height)
    )
    mapx, mapy = cv2.initUndistortRectifyMap(
        mtx, dist, None, newcameramtx, (width, height), cv2.CV_32FC1
    )
    return mapx, mapy, roi


def undistort_frame(frame, mapx, mapy, roi, width, height):
    """تعديل الصورة واستعدال الأبعاد"""
    undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    x, y, w, h = roi
    undistorted = undistorted[y:y + h, x:x + w]
    undistorted = cv2.resize(undistorted, (width, height))
    return undistorted


def detect_target_fps(cap):
    """حساب متوسط السرعة الحقيقية للفيديو من الـ Timestamps تماماً مثل resize.py"""
    meta_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    last_ts_ms = 0
    frames_seen = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        ts = cap.get(cv2.CAP_PROP_POS_MSEC)
        if ts > 0:
            last_ts_ms = ts
        frames_seen += 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # إعادة الفيديو للبداية المعيارية

    if last_ts_ms > 0 and frames_seen > 1:
        real_fps = (frames_seen - 1) / (last_ts_ms / 1000.0)
        print(f"ℹ️ تم اكتشاف سرعة الفيديو الحقيقية: {real_fps:.2f} FPS (مقايل {meta_fps:.2f} في البيانات)")
        return real_fps

    print(f"⚠️ يتعذر قراءة Timestamps بدقة، اعتماد سرعة الميتا: {meta_fps:.2f} FPS")
    return meta_fps


def main():
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"🖥️ الجهاز المستخدم للمعالجة: {DEVICE} ({device_name})")

    # 1. تحميل الموديلين
    print("⏳ جاري تحميل الموديل الأول (Crop/Detection)...")
    model1 = YOLO(MODEL1_PATH)
    
    print("⏳ جاري تحميل الموديل الثاني (Defect Detection)...")
    model2 = YOLO(MODEL2_PATH)
    target_ids2 = [k for k, v in model2.names.items() if v in TARGET_CLASSES2] if TARGET_CLASSES2 else None

    # 2. فتح فيديو المدخلات
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print("❌ تعذر فتح الفيديو:", INPUT_VIDEO)
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # تحديد السرعة الطبيعية الصحيحة للفيديو (FPS)
    target_fps = TARGET_FPS or detect_target_fps(cap)
    frame_duration_ms = 1000.0 / target_fps
    wait_ms = max(1, int(1000.0 / target_fps))

    print(f"🎬 سرعة الفيديو المستهدفة: {target_fps:.2f} FPS (مزامنة {wait_ms}ms)")

    # إعداد خرائط تعديل الأبعاد (Resize/Undistort)
    mapx, mapy, roi = build_undistort_maps(CAMERA_MATRIX, DIST_COEFFS, width, height)

    # 3. إعداد ملف إخراج الفيديو بالسرعة المظبوطة
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, target_fps, (width, height))

    # 4. إعداد نافذة المعاينة المباشرة
    WINDOW_NAME = "AllInOne Pipeline - Live Detection & PIP Overlay"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    box_history = []
    target_crop_w, target_crop_h = None, None
    next_output_time_ms = 0.0
    frames_processed = 0
    frames_written = 0

    print("🚀 بدء المعالجة الشاملة الدقيقة (Undistort -> Detect1 & Crop -> Detect2 -> Overlay)...")

    with torch.no_grad():
        while cap.isOpened():
            ret, raw_frame = cap.read()
            if not ret:
                break

            frames_processed += 1
            ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

            # ==================================================
            # المرحلة 1: Resize & Undistort (الكود الأول)
            # ==================================================
            main_frame = undistort_frame(raw_frame, mapx, mapy, roi, width, height)

            # ==================================================
            # المرحلة 2: Detect 1 & Crop (الكود الثاني)
            # ==================================================
            results1 = model1(main_frame, conf=MODEL1_CONF, half=USE_HALF, device=DEVICE)[0]

            crop = None
            cou_detected = False

            for box in results1.boxes:
                cls_id = int(box.cls[0])
                class_name = model1.names[cls_id]

                if class_name == TARGET_CLASS1:
                    cou_detected = True
                    current_box = list(map(int, box.xyxy[0]))
                    box_history.append(current_box)

                    # حساب المتوسط التراكمي لثبات البوكس
                    avg_x1 = int(sum(b[0] for b in box_history) / len(box_history))
                    avg_y1 = int(sum(b[1] for b in box_history) / len(box_history))
                    avg_x2 = int(sum(b[2] for b in box_history) / len(box_history))
                    avg_y2 = int(sum(b[3] for b in box_history) / len(box_history))

                    x1, y1 = max(0, avg_x1), max(0, avg_y1)
                    x2, y2 = min(width, avg_x2), min(height, avg_y2)

                    if x2 > x1 and y2 > y1:
                        crop = main_frame[y1:y2, x1:x2].copy()
                        
                        # رسم البوكس الأول على الفريم الرئيسي
                        conf_val = float(box.conf[0])
                        cv2.rectangle(main_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(main_frame, f"{class_name} {conf_val:.2f}", (x1, max(25, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    break

            if not cou_detected:
                box_history.clear()

            # ==================================================
            # المرحلة 3: Detect 2 على الجزء المقصوص (الكود الثالث)
            # ==================================================
            if crop is not None and crop.size > 0:
                # ضبط أبعاد قطعة القص لتطابق الأبعاد المعيارية التي تدرب عليها الموديل الثاني (مثل الكود الثاني)
                if target_crop_w is None or target_crop_h is None:
                    target_crop_h, target_crop_w = crop.shape[:2]

                crop_for_model2 = cv2.resize(crop, (target_crop_w, target_crop_h), interpolation=cv2.INTER_LINEAR)

                results2 = model2(
                    crop_for_model2, 
                    conf=MODEL2_CONF, 
                    iou=MODEL2_IOU, 
                    agnostic_nms=AGNOSTIC_NMS, 
                    classes=target_ids2, 
                    device=DEVICE
                )[0]

                crop_annotated = results2.plot()

                # ==================================================
                # المرحلة 4: تركيب فيديو القص في المنتصف على اليسار (Middle-Left)
                # ==================================================
                ch, cw = crop_annotated.shape[:2]

                if ch > 0 and cw > 0:
                    # تكبير الحجم بناءً على نسبة الارتفاع مع الحفاظ على نسبة العرض والارتفاع (Aspect Ratio)
                    target_h = int(height * PIP_SCALE_HEIGHT)
                    target_w = int(cw * (target_h / ch))

                    max_w = int(width * 0.45)
                    if target_w > max_w:
                        target_w = max_w
                        target_h = int(ch * (target_w / cw))

                    pip_resized = cv2.resize(crop_annotated, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                    # الموضع: على الشمال وفي منتصف الارتفاع تماماً (Middle-Left)
                    x1_pip = PIP_PADDING
                    x2_pip = x1_pip + target_w
                    y1_pip = max(0, (height - target_h) // 2)
                    y2_pip = y1_pip + target_h

                    if y2_pip <= height and x2_pip <= width:
                        # رسم حد إطار أبيض وعنوان أنيق
                        cv2.rectangle(main_frame, (x1_pip - 2, y1_pip - 2), (x2_pip + 2, y2_pip + 2), (255, 255, 255), 2)
                        cv2.putText(main_frame, "Cropped Detection (Model 2)", (x1_pip, max(18, y1_pip - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        main_frame[y1_pip:y2_pip, x1_pip:x2_pip] = pip_resized

            # تسجيل الفريم بالاعتماد على الفوارق الزمنية الحقيقية (Timestamps) لمنع التسريع ومطابقة resize.py
            if ts_ms > 0:
                while ts_ms >= next_output_time_ms:
                    writer.write(main_frame)
                    frames_written += 1
                    next_output_time_ms += frame_duration_ms
            else:
                writer.write(main_frame)
                frames_written += 1

            cv2.imshow(WINDOW_NAME, main_frame)

            if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                print("🛑 تم إيقاف التشغيل بواسطة المستخدم.")
                break

            if frames_processed % 50 == 0 or frames_processed == total_frames:
                print(f"⏳ تم معالجة {frames_processed}/{total_frames} إطار...")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print("==================================================")
    print("✅ تم الانتهاء بنجاح وحفظ الفيديو المدمج!")
    print(f"🎥 الفيديو النهائي (allcode): {OUTPUT_VIDEO}")
    print("==================================================")

if __name__ == "__main__":
    main()
