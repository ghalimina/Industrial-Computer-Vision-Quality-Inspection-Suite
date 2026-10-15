import cv2
import torch
import os
from ultralytics import YOLO

# ==================================================
# PATHS & CONFIG (إعدادات المسارات)
# ==================================================
MODEL_PATH = r"bestv1.11.pt"
INPUT_VIDEO = r"E:\elecrto pi\co\cam5re_detectco_crop_avgonlybox.mp4"
OUTPUT_VIDEO = r"E:\elecrto pi\co\cam5re_detect_fol_hol.mp4"

# مجلد واحد فقط لحفظ جميع الصور (بالـ Detect وبدون)
OUTPUT_DIR = r"E:\elecrto pi\co\cam1re_detect_fol_hol_frames"

# إنشاء المجلد إذا لم يكن موجوداً
os.makedirs(OUTPUT_DIR, exist_ok=True)

# الكلاسات المطلوبة فقط
TARGET_CLASSES = ["cou", "fol"]

# نسبة الثقة المطلوب ضبطها
CONFIDENCE_THRESHOLD = 0.40
IOU_THRESHOLD = 0.25         # تقليل نسبة التداخل إلى 25% لحذف أي بوكس متداخل
AGNOSTIC_NMS = True          # دمج وحذف البوكسات المتداخلة حتى لو كانت لكلاسات مختلفة

# أبعاد نافذة العرض المباشر
DISPLAY_WIDTH = 1024
DISPLAY_HEIGHT = 576

# تحديد جهاز المعالجة
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print(f"🖥️ الجهاز المستخدم للمعالجة: {DEVICE}")
    
    # 1. تحميل نموذج YOLO
    print("⏳ جاري تحميل النموذج...")
    model = YOLO(MODEL_PATH)

    # 💡 استخراج أرقام (IDs) الكلاسات المطلوبة فقط (fol و hol)
    target_ids = [k for k, v in model.names.items() if v in TARGET_CLASSES]

    # 2. فتح فيديو المدخلات
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print("❌ تعذر فتح الفيديو:", INPUT_VIDEO)
        return

    # قراءة أبعاد وسرعة الفيديو
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 3. إعداد ملف إخراج الفيديو
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    # 4. إعداد نافذة المعاينة المباشرة (Live Preview Window)
    WINDOW_NAME = "YOLO Live Detection (fol & hol) - Press 'q' to exit"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    frames_processed = 0
    
    print(f"🚀 بدء عملية الكشف واستخراج جميع الفريمات...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frames_processed += 1
        
        # 5. أخذ نسخة أصلية من الفريم (بدون أي تعديل) لحفظها
        raw_frame = frame.copy()
        
        # 6. تشغيل عملية الكشف (مع تمرير target_ids و agnostic_nms لمنع التداخل)
        results = model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, agnostic_nms=AGNOSTIC_NMS, classes=target_ids, device=DEVICE)[0]

        # 7. الرسم التلقائي الجاهز من مكتبة YOLO (سيرسم fol و hol فقط بناءً على الفلتر السابق)
        annotated_frame = results.plot()

        # 8. حفظ *كل فريم* بصورتين (أصلية ومعالجة) في نفس المجلد
        # الترقيم zfill(5) يجعل الأسماء مرتبة 00001, 00002...
        frame_number = str(frames_processed).zfill(5)
        
        raw_img_path = os.path.join(OUTPUT_DIR, f"frame_{frame_number}_a_raw.jpg")
        det_img_path = os.path.join(OUTPUT_DIR, f"frame_{frame_number}_b_detect.jpg")
        
        # حفظ الصور (حتى لو لم يجد الموديل شيء، سيحفظ الصورة كما هي)
        cv2.imwrite(raw_img_path, raw_frame)
        cv2.imwrite(det_img_path, annotated_frame)

        # 9. إضافة الإطار للفيديو النهائي
        writer.write(annotated_frame)

        # 10. عرض الإطار Live (المشاهدة المباشرة)
        cv2.imshow(WINDOW_NAME, annotated_frame)

        # الضغط على زر 'q' للخروج والتوقف
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 تم إيقاف التشغيل بواسطة المستخدم.")
            break

        # عرض تفاصيل التقدم في الكونسول
        if frames_processed % 50 == 0 or frames_processed == total_frames:
            print(f"⏳ تم معالجة وحفظ {frames_processed}/{total_frames} إطار (بواقع {frames_processed * 2} صورة)...")

    # إغلاق الملفات والنوافذ
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    
    print("==================================================")
    print("✅ تم الانتهاء بنجاح!")
    print(f"🎥 الفيديو النهائي المحفوظ: {OUTPUT_VIDEO}")
    print(f"🖼️ جميع الصور (الأصلية والـ Detect) محفوظة في: {OUTPUT_DIR}")
    print("==================================================")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ حدث خطأ غير متوقع: {e}")
        input("اضغط على زر Enter للإغلاق...")