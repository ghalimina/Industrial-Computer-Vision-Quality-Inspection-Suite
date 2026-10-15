import cv2
import numpy as np
from ultralytics import YOLO

# 1. تحميل الموديل
model = YOLO(r"E:\elecrto pi\front\bestv1.2.pt")

# فتح الفيديو الأصلي
video_path = r"E:\elecrto pi\front\front.mp4"
cap = cv2.VideoCapture(video_path)

# ---- إعدادات حفظ الفيديو الجديد ----
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 25

# مسار ملف الخرج
output_path = "front _Result.mp4" 
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # كود الضغط للملف
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
# ----------------------------------

window_name = "Electro Pi - Ultra Clear Quality Control Dashboard"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1366, 768)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # تقليل الـ iou وزيادة الثقة لتنظيف التداخل من جذوره
    results = model(frame, conf=0.65, iou=0.15, agnostic_nms=True)[0]
    
    temp_clamps = []
    temp_reg = []
    temp_head = []
    temp_tail = []

    # 1. جمع وتصنيف كل الكائنات المكتشفة مع نسبة الثقة
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id].lower()
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf_score = float(box.conf[0])
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        if cls_name == "clamp":
            temp_clamps.append((center_x, center_y, conf_score, x1, y1, x2, y2))
        elif cls_name == "reg":
            temp_reg.append((x1, y1, x2, y2, conf_score))
        elif cls_name == "head":
            temp_head.append((center_x, center_y, conf_score, x1, y1, x2, y2))
        elif cls_name == "tail":
            temp_tail.append((center_x, center_y, conf_score, x1, y1, x2, y2))

    # ==========================================
    # الفلتر الذكي للقضاء على التداخلات (Smart Filtering)
    # ==========================================
    
    # ترتيب الكائنات حسب الأقوى ثقة
    temp_clamps.sort(key=lambda x: x[2], reverse=True)
    temp_reg.sort(key=lambda x: x[4], reverse=True)
    temp_head.sort(key=lambda x: x[2], reverse=True)
    temp_tail.sort(key=lambda x: x[2], reverse=True)

    filtered_clamps = []
    MIN_DISTANCE = 30  # المسافة بالبكسل التي تحدد إذا كان المربعان لنفس الكائن أم لا

    for c in temp_clamps:
        is_duplicate = False
        for fc in filtered_clamps:
            # إذا كان هناك Clamp آخر قريب جداً منه، اعتبره تداخل (Duplicate) وتجاهله
            dist = np.sqrt((c[0] - fc[0])**2 + (c[1] - fc[1])**2)
            if dist < MIN_DISTANCE:
                is_duplicate = True
                break
        if not is_duplicate:
            filtered_clamps.append(c)

    # اختيار أفضل كائن واحد فقط لكل من البقية
    reg_box = temp_reg[0][:4] if len(temp_reg) > 0 else None
    
    head_center = temp_head[0][:2] if len(temp_head) > 0 else None
    head_box = temp_head[0][3:7] if len(temp_head) > 0 else None
    
    tail_center = temp_tail[0][:2] if len(temp_tail) > 0 else None
    tail_box = temp_tail[0][3:7] if len(temp_tail) > 0 else None

    # ==========================================
    # تطبيق الرسم للبيانات المفلترة فقط
    # ==========================================
    clamps = []
    for c in filtered_clamps:
        cx, cy, conf, x1, y1, x2, y2 = c
        clamps.append((cx, cy))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 128), 2)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    if head_box:
        cv2.rectangle(frame, (head_box[0], head_box[1]), (head_box[2], head_box[3]), (255, 0, 255), 2) 
    if tail_box:
        cv2.rectangle(frame, (tail_box[0], tail_box[1]), (tail_box[2], tail_box[3]), (0, 255, 255), 2) 

    # 1. لوحة البيانات الرئيسية (UI Dashboard - Left)
    cv2.rectangle(frame, (10, 10), (580, 260), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (580, 260), (150, 150, 150), 3) 
    
    total_clamps = len(clamps)
    clamp_status_color = (0, 255, 0) if total_clamps == 14 else (0, 0, 255)
    cv2.putText(frame, f"1. Total Clamps: {total_clamps}/14", (25, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, clamp_status_color, 2)

    if reg_box is not None:
        rx1, ry1, rx2, ry2 = reg_box
        center_circle_x = (rx1 + rx2) // 2
        center_circle_y = (ry1 + ry2) // 2
        cv2.circle(frame, (center_circle_x, center_circle_y), 8, (0, 255, 255), -1) 

        def get_angle(pt):
            angle = np.degrees(np.arctan2(pt[1] - center_circle_y, pt[0] - center_circle_x))
            return angle + 360 if angle < 0 else angle

        clamp_data = []
        radii = [] 
        for c in clamps:
            angle = get_angle(c)
            r_pixel = np.sqrt((c[0] - center_circle_x)**2 + (c[1] - center_circle_y)**2)
            radii.append(r_pixel)
            clamp_data.append({'coords': c, 'angle': angle, 'radius': r_pixel})
        
        clamp_data.sort(key=lambda x: x['angle'])
        avg_radius = np.mean(radii) if len(radii) > 0 else 0

        # 2. حساب التداخل (Overlap Arc)
        if head_center is not None and tail_center is not None:
            head_angle = get_angle(head_center)
            tail_angle = get_angle(tail_center)

            cv2.line(frame, (center_circle_x, center_circle_y), head_center, (255, 0, 255), 3)
            cv2.line(frame, (center_circle_x, center_circle_y), tail_center, (0, 255, 255), 3)

            overlap_angle = head_angle - tail_angle
            if overlap_angle < 0:
                overlap_angle += 360

            clamps_in_overlap = 0
            for c in clamp_data:
                cx, cy = c['coords']
                a = c['angle']
                
                is_inside_head = False
                if head_box is not None:
                    hx1, hy1, hx2, hy2 = head_box
                    if (hx1 - 15) <= cx <= (hx2 + 15) and (hy1 - 15) <= cy <= (hy2 + 15):
                        is_inside_head = True

                is_inside_angle_arc = False
                if tail_angle <= head_angle:
                    if tail_angle <= a <= head_angle:
                        is_inside_angle_arc = True
                else: 
                    if a >= tail_angle or a <= head_angle:
                        is_inside_angle_arc = True

                if is_inside_angle_arc or is_inside_head:
                    clamps_in_overlap += 1
                    cv2.circle(frame, (cx, cy), 10, (0, 0, 255), 3)

            overlap_4_color = (0, 255, 0) if clamps_in_overlap == 4 else (0, 0, 255)
            cv2.putText(frame, "2. Overlap Segment: DETECTED", (25, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
            cv2.putText(frame, f"   -> Angle: {overlap_angle:.1f} DEG | R: {avg_radius:.1f} px", (25, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(frame, f"4. Clamps in Overlap: {clamps_in_overlap}/4", (25, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.85, overlap_4_color, 2)
        else:
            cv2.putText(frame, "2. Overlap Segment: MISSING DATA", (25, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
            cv2.putText(frame, f"   -> R (Circle): {avg_radius:.1f} px", (25, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(frame, "4. Clamps in Overlap: N/A", (25, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)

        # 3. حساب المسافات الفعلية (بالبكسل) للتأكد من التساوي
        pixel_distances = []
        is_equal_spacing = True
        tolerance_px = 15.0  # نسبة السماحية بالبكسل لتجنب الاهتزاز
        
        if total_clamps > 1:
            for i in range(len(clamp_data)):
                current_clamp = clamp_data[i]
                next_clamp = clamp_data[(i + 1) % len(clamp_data)]
                
                cx1, cy1 = current_clamp['coords']
                cx2, cy2 = next_clamp['coords']
                dist_px = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
                pixel_distances.append(round(dist_px, 1))

                dx = cx1 - center_circle_x
                dy = cy1 - center_circle_y
                dist_center = np.sqrt(dx*dx + dy*dy) if np.sqrt(dx*dx + dy*dy) > 0 else 1
                
                text_x = int(cx1 + (dx / dist_center) * 45)
                text_y = int(cy1 + (dy / dist_center) * 45)

                cv2.circle(frame, (text_x + 15, text_y - 5), 18, (0, 0, 0), -1)
                cv2.circle(frame, (text_x + 15, text_y - 5), 18, (255, 255, 255), 1)
                
                text_str = f"{dist_px:.0f}" 
                cv2.putText(frame, text_str, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            avg_dist = np.mean(pixel_distances)
            
            for d in pixel_distances:
                if abs(d - avg_dist) > tolerance_px:
                    is_equal_spacing = False
            
            panel_x = frame.shape[1] - 320 
            cv2.rectangle(frame, (panel_x - 15, 10), (frame.shape[1] - 10, 340), (20, 20, 20), -1)
            cv2.rectangle(frame, (panel_x - 15, 10), (frame.shape[1] - 10, 340), (255, 255, 0), 2)
            
            cv2.putText(frame, "LIVE DISTANCES (px)", (panel_x, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.line(frame, (panel_x, 45), (frame.shape[1] - 25, 45), (150, 150, 150), 1)
            
            for idx, d in enumerate(pixel_distances):
                y_pos = 75 + (idx * 19)
                text_color = (0, 255, 0) if abs(d - avg_dist) <= tolerance_px else (0, 0, 255)
                
                cv2.putText(frame, f"Dist {idx+1}->{idx+2 if idx+2<=len(pixel_distances) else 1}: {d:.1f}", 
                            (panel_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2)

            if is_equal_spacing and total_clamps == 14:
                cv2.putText(frame, "3. Spacing: EQUAL (PASSED)", (25, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "3. Spacing: NOT EQUAL (FAILED)", (25, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "3. Spacing: Waiting for Reg...", (25, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)

    # ---- حفظ الفريم الحالي داخل الفيديو الجديد ----
    out.write(frame)
    # --------------------------------------------

    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# إغلاق وحفظ كل شيء بشكل سليم
cap.release()
out.release()
cv2.destroyAllWindows()