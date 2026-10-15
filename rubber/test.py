import cv2
import numpy as np
import os
from ultralytics import YOLO

# ==========================================
# Paths (المسارات)
# ==========================================
model_path = r"E:\elecrto pi\rubber\bestv2.pt" 
video_path = r"E:\elecrto pi\rubber\test.mp4"
output_frames_dir = r"E:\elecrto pi\rubber\test_frames"
output_video_path = r"E:\elecrto pi\rubber\test_outputdetect.mp4"

if not os.path.exists(output_frames_dir):
    os.makedirs(output_frames_dir)

# ==========================================
# [تحكم الشفافية]: نسبة ظهور اللون الداخلي (Fill)
# 0 = شفاف تماماً (لا يوجد لون داخلي)
# 100 = لون مصمت وثقيل بالكامل
# ==========================================
fill_percentage = 40  

alpha = fill_percentage / 100.0
beta = 1.0 - alpha

# ==========================================
# Load Model & Video
# ==========================================
model = YOLO(model_path).to('cuda') 
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
# تعريف الألوان الخاصة بالكلاسات الجديدة (BGR)
# ==========================================
class_colors = {
    "wire": (255, 255, 0),    # أزرق فاتح / سماوي (Cyan)
    "Rubber": (0, 165, 255)   # برتقالي للـ Rubber
}
default_color = (255, 255, 255)

target_classes = ["wire", "Rubber"]

cv2.namedWindow("Industrial Inspection AI - Wire & Rubber Detect", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Industrial Inspection AI - Wire & Rubber Detect", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    print(f"Processing frame {frame_count}")

    # 1. حفظ الفريم الأصلي النقي
    original_frame_path = os.path.join(output_frames_dir, f"frame_{frame_count:05d}_A_original.jpg")
    cv2.imwrite(original_frame_path, frame)

    results = model(frame, conf=0.60, iou=0.45, device='cuda', verbose=False)
    
    annotated_frame = frame.copy()
    # طبقة منفصلة لرسم الأقنعة الممتلئة عليها
    mask_overlay = frame.copy()
    
    current_detections = {cls_name: 0 for cls_name in target_classes}

    # --------------------------------------------------
    # معالجة واكتشاف الكلاسات (Detection + Segmentation)
    # --------------------------------------------------
    for result in results:
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy() 
            classes = result.boxes.cls.cpu().numpy()

            for idx, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
                class_name = model.names[int(cls)]
                
                if class_name in target_classes:
                    current_detections[class_name] += 1
                    color = class_colors.get(class_name, default_color)
                    xb1, yb1, xb2, yb2 = map(int, box)
                    
                    # --------------------------------------------------
                    # حالة كلاس الـ Rubber: معالجة ذكية بالـ Binary Mask لمنع الخيوط المتقاطعة
                    # --------------------------------------------------
                    if class_name == "Rubber" and result.masks is not None:
                        # استخراج قناع الكائن الحالي بدقة بكسلية وتكبيره لحجم الفيديو المستهدف
                        binary_mask = result.masks.data[idx].cpu().numpy()
                        binary_mask = cv2.resize(binary_mask, (width, height_vid))
                        
                        # إنشاء قناع منطقي (True/False) للبكسلات التي تخص الـ Rubber فقط
                        mask_bool = binary_mask > 0.5
                        
                        # 1. تلوين البكسلات الداخلية مباشرة دون رسم خطوط (تأثير متماسك ومصمت)
                        mask_overlay[mask_bool] = color
                        
                        # 2. استخراج الحواف الخارجية الحقيقية فقط (External Contour) لرسم إطار نظيف
                        mask_uint8 = (mask_bool * 255).astype(np.uint8)
                        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(annotated_frame, contours, -1, color, thickness=3)
                        
                        # كتابة النص فوق الـ Rubber
                        label_text = f"{class_name} {conf:.2f}"
                        cv2.putText(annotated_frame, label_text, (xb1, yb1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                    
                    # --------------------------------------------------
                    # حالة كلاس الـ wire: رسم صندوق كلاسيكي (الاسم تحت البوكس)
                    # --------------------------------------------------
                    else:
                        cv2.rectangle(annotated_frame, (xb1, yb1), (xb2, yb2), color, 3)
                        
                        label_text = f"{class_name} {conf:.2f}"
                        cv2.putText(annotated_frame, label_text, (xb1, yb2 + 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # دمج طبقة الـ Fill الشفافة والنظيفة مع الفريم الأساسي
    cv2.addWeighted(mask_overlay, alpha, annotated_frame, beta, 0, annotated_frame)

    # ==========================================
    # تصميم لوحة البيانات الرقمية (تحت على الشمال)
    # ==========================================
    panel_width = 500
    panel_height = 140  
    margin = 20

    x1 = margin
    y1 = height_vid - panel_height - margin
    x2 = margin + panel_width
    y2 = height_vid - margin

    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (200, 200, 200), 2)
    
    font_scale = 0.65  
    thickness_val = 2  
    
    cv2.putText(annotated_frame, f"Frame: {frame_count}", (x1 + 20, y1 + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    
    y_offset = y1 + 70
    for cls_name in target_classes:
        count_text = f"Current {cls_name}: {current_detections[cls_name]}"
        color = class_colors.get(cls_name, default_color)
        
        cv2.putText(annotated_frame, count_text, (x1 + 20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness_val, cv2.LINE_AA)
        y_offset += 40 

    # ==========================================
    # Save & Show final annotated frame
    # ==========================================
    out_video.write(annotated_frame)

    annotated_frame_path = os.path.join(output_frames_dir, f"frame_{frame_count:05d}_B_annotated.jpg")
    cv2.imwrite(annotated_frame_path, annotated_frame)

    screen = cv2.resize(annotated_frame, (width, height_vid))
    cv2.imshow("Industrial Inspection AI - Wire & Rubber Detect", screen)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out_video.release()  
cv2.destroyAllWindows()
print(f"Done! Video processing finished. Output saved to: {output_video_path}")