import cv2
import numpy as np
import os
from ultralytics import YOLO

# ==========================================
# Paths (المسارات)
# ==========================================
model_path = r"E:\elecrto pi\lebt\bestv22.pt" 
video_path = r"E:\elecrto pi\lebt\test y25.mkv"
output_frames_dir = r"E:\elecrto pi\lebt\test y25_frames"
output_video_path = r"E:\elecrto pi\lebt\test y25_outputdetect.mp4"

if not os.path.exists(output_frames_dir):
    os.makedirs(output_frames_dir)

# ==========================================
# Load Model & Video
# ==========================================
model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0 or np.isnan(fps):
    fps = 30.0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height_vid))

# ==========================================
# تعريف الألوان الخاصة بالكلاسات المطلوبة (BGR)
# ==========================================
class_colors = {
    "Bad": (0, 0, 255),      # أحمر
    "good": (0, 255, 0),     # أخضر
    "red": (0, 165, 255),    # برتقالي
    "Text": (255, 0, 0)      # أزرق
}
default_color = (255, 255, 255)

# الكلاسات المستهدفة للفلترة (تم مسح best)
target_classes = ["Bad", "good", "red", "Text"]

cv2.namedWindow("Industrial Inspection AI - Text Detect", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Industrial Inspection AI - Text Detect", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# متغيّرات إحصائية
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    print(f"Processing frame {frame_count}")

    # --------------------------------------------------
    # 1. حفظ الفريم الأصلي النقي (من غير detect)
    # --------------------------------------------------
    original_frame_path = os.path.join(output_frames_dir, f"frame_{frame_count:05d}_A_original.jpg")
    cv2.imwrite(original_frame_path, frame)

    # تشغيل الموديل
    results = model(frame, conf=0.60, iou=0.45)
    annotated_frame = frame.copy()
    
    # قاموس لعد الكلاسات المكتشفة في الفريم الحالي ديناميكياً
    current_detections = {cls_name: 0 for cls_name in target_classes}

    # --------------------------------------------------
    # معالجة واكتشاف الكلاسات المطلوبة فقط
    # --------------------------------------------------
    for result in results:
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy() 
            classes = result.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confs, classes):
                class_name = model.names[int(cls)]
                
                # فلترة الكلاسات بناءً على القائمة المطلوبة
                if class_name in target_classes:
                    xb1, yb1, xb2, yb2 = map(int, box)
                    
                    # زيادة عداد الكائن المكتشف حالياً
                    current_detections[class_name] += 1

                    # رسم البوكس والكتابة فوق الكائن المكتشف
                    color = class_colors.get(class_name, default_color)
                    cv2.rectangle(annotated_frame, (xb1, yb1), (xb2, yb2), color, 3)
                    
                    label_text = f"{class_name} {conf:.2f}"
                    cv2.putText(annotated_frame, label_text, (xb1, yb1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # ==========================================
    # تصميم لوحة البيانات الرقمية (تحت على الشمال)
    # ==========================================
    panel_width = 500
    panel_height = 200  # تم تقليل الارتفاع لأن الكلاسات أصبحت 4 بدلاً من 5
    margin = 20

    x1 = margin
    y1 = height_vid - panel_height - margin
    x2 = margin + panel_width
    y2 = height_vid - margin

    # رسم خلفية اللوحة
    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (200, 200, 200), 2)
    
    font_scale = 0.65  
    thickness_val = 2  
    
    # كتابة رقم الفريم الحالي
    cv2.putText(annotated_frame, f"Frame: {frame_count}", (x1 + 20, y1 + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    
    # كتابة إحصائيات الكلاسات الأربعة المتبقية داخل اللوحة السفلية
    y_offset = y1 + 65
    for cls_name in target_classes:
        count_text = f"Current {cls_name}: {current_detections[cls_name]}"
        color = class_colors.get(cls_name, default_color)
        
        cv2.putText(annotated_frame, count_text, (x1 + 20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness_val, cv2.LINE_AA)
        y_offset += 35 # مسافة رأسية بين كل سطر وسطر

    # ==========================================
    # 2. Save & Show final annotated frame (بـ detect)
    # ==========================================
    out_video.write(annotated_frame)

    annotated_frame_path = os.path.join(output_frames_dir, f"frame_{frame_count:05d}_B_annotated.jpg")
    cv2.imwrite(annotated_frame_path, annotated_frame)

    screen = cv2.resize(annotated_frame, (width, height_vid))
    cv2.imshow("Industrial Inspection AI - Text Detect", screen)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out_video.release()  
cv2.destroyAllWindows()
print(f"Done! Video processing finished. Output saved to: {output_video_path}")