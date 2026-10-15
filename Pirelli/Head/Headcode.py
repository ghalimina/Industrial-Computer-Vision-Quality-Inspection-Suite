import cv2
import numpy as np
import os
from ultralytics import YOLO

# ==========================================
# Paths
# ==========================================
model_path = r"E:\elecrto pi\aitest\best.pt"
video_path = r"E:\elecrto pi\aitest\right 3.mp4"
output_frames_dir = r"E:\elecrto pi\aitest\right 3_frames"
output_video_path = r"E:\elecrto pi\aitest\right 3last_output.mp4"

if not os.path.exists(output_frames_dir):
    os.makedirs(output_frames_dir)

# ==========================================
# متغير التحكم في خط الحجب (الإزاحة بكسل من أعلى بوكس الـ Bead)
# ==========================================
Y_BLOCK_OFFSET = 50

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
# تعريف الألوان الخاصة لكل كائن والمسافات (BGR)
# ==========================================
class_colors = {
    "Bead": (0, 255, 255),        # أصفر
    "Reg 1": (0, 255, 0),         # أخضر
    "Reg 2": (255, 200, 0),       # أزرق فاتح / سماوي
    "Height": (0, 165, 255),      # برتقالي
    "Wire_Long": (255, 0, 255),   # بنفسجي غامق للواير الطويل
    "Wire_Short": (255, 100, 255),# بنفسجي فاتح للواير القصير
    "Wire_General": (200, 50, 150),# لون مميز لباقي الوايرات المتداخلة
    "Distances": (0, 255, 128),   # أخضر فستقي لخطوط ونصوص المسافات
    "ThresholdLine": (0, 0, 255), # أحمر لخط الحجب
    "WireCount": (0, 140, 255)    # لون مميز لإحصائية عدد الوايرات
}
default_color = (255, 255, 255)

cv2.namedWindow("Industrial Inspection AI", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Industrial Inspection AI", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# متغيّرات إحصائية
frame_count = 0
all_angles = []
all_reg2_widths = []
all_heights = []
all_wire_distances = []    
all_reg1_short_distances = []  # مصفوفة إحصائية للمسافة الجديدة

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    print(f"Processing frame {frame_count}")

    original_frame_path = os.path.join(output_frames_dir, f"frame_{frame_count:05d}_A_original.jpg")
    cv2.imwrite(original_frame_path, frame)

    results = model(frame, conf=0.50, iou=0.30)
    annotated_frame = frame.copy()
    
    current_angle = None
    current_reg2_width = None
    current_height_val = None
    dist_wires = None
    dist_short_wire_reg1 = None  # قيمة المسافة الجديدة في الفريم الحالي
    
    bead_data = []  
    detected_wires = []
    detected_reg1 = []          # لتخزين كائنات Reg 1 المكتشفة
    all_objects_to_check = [] 
    
    dynamic_y_threshold = 100

    # --------------------------------------------------
    # Phase 1: Collect and classify all detected items
    # --------------------------------------------------
    for result in results:
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy() 
            classes = result.boxes.cls.cpu().numpy()
            masks = result.masks.xy if result.masks is not None else None

            for idx, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
                class_name = model.names[int(cls)]
                xb1, yb1, xb2, yb2 = map(int, box)
                
                if class_name == "Wire":
                    detected_wires.append((box, conf))
                    all_objects_to_check.append((box, class_name))
                elif class_name == "Bead":
                    bead_data.append((box, masks[idx] if masks is not None and idx < len(masks) else None, conf))
                    dynamic_y_threshold = yb1 + Y_BLOCK_OFFSET
                elif class_name in ["Reg 1", "Reg 2", "Height"]:
                    all_objects_to_check.append((box, class_name))
                    if class_name == "Reg 1":
                        detected_reg1.append(box) # حفظ الصندوق للبحث اللاحق
                    elif class_name == "Reg 2":
                        current_reg2_width = xb2 - xb1
                        all_reg2_widths.append(current_reg2_width)
                    elif class_name == "Height":
                        current_height_val = yb2 - yb1
                        all_heights.append(current_height_val)

                    color = class_colors.get(class_name, default_color)
                    cv2.rectangle(annotated_frame, (xb1, yb1), (xb2, yb2), color, 2)
                    label_text = f"{class_name} {conf:.2f}"
                    cv2.putText(annotated_frame, label_text, (xb1, yb1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    # حساب إجمالي عدد الوايرات المكتشفة في الفريم الحالي
    total_wires_count = len(detected_wires)

    # --------------------------------------------------
    # Phase 2: Check Blocking Threshold line
    # --------------------------------------------------
    object_blocking_at_top = False 
    for obj_box, c_name in all_objects_to_check:
        ox1, oy1, ox2, oy2 = map(int, obj_box)
        if oy1 <= dynamic_y_threshold:
            object_blocking_at_top = True
            break

    # --------------------------------------------------
    # Phase 3: Draw Intersecting Lines & Distances
    # --------------------------------------------------
    short_wire_box = None
    long_wire_box = None

    if len(detected_wires) >= 1:
        wire_y_centers = [(b[0][1] + b[0][3]) / 2 for b in detected_wires]
        sorted_indices = np.argsort(wire_y_centers)
        
        # 1. الواير الأقصر/الأعلى موقعاً (Wire Short)
        short_wire_box = detected_wires[sorted_indices[0]][0]
        sw_conf = detected_wires[sorted_indices[0]][1]
        sw_x1, sw_y1, sw_x2, sw_y2 = map(int, short_wire_box)
        cv2.rectangle(annotated_frame, (sw_x1, sw_y1), (sw_x2, sw_y2), class_colors["Wire_Short"], 2)
        cv2.putText(annotated_frame, f"Wire Short {sw_conf:.2f}", (sw_x1, sw_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_colors["Wire_Short"], 2, cv2.LINE_AA)
        
        center_sw_x = int((sw_x1 + sw_x2) / 2)
        center_sw_y = int((sw_y1 + sw_y2) / 2)

        # البحث عن أقرب Reg 1 فوق الواير القصير
        best_reg1_box = None
        min_reg1_dist = float('inf')

        for r_box in detected_reg1:
            rx1, ry1, rx2, ry2 = map(int, r_box)
            if ry2 < center_sw_y:
                current_dist = np.sqrt((rx2 - center_sw_x)**2 + (ry2 - center_sw_y)**2)
                if current_dist < min_reg1_dist:
                    min_reg1_dist = current_dist
                    best_reg1_box = r_box

        if best_reg1_box is not None:
            brx1, bry1, brx2, bry2 = map(int, best_reg1_box)
            reg1_point = (brx2, bry2)
            
            dist_short_wire_reg1 = abs(center_sw_y - bry2)
            all_reg1_short_distances.append(dist_short_wire_reg1)
            
            inter_reg1_sw_point = (brx2, center_sw_y)
            
            cv2.line(annotated_frame, reg1_point, inter_reg1_sw_point, class_colors["Distances"], 2, cv2.LINE_AA)
            cv2.line(annotated_frame, (center_sw_x, center_sw_y), inter_reg1_sw_point, class_colors["Distances"], 2, cv2.LINE_AA)
            cv2.circle(annotated_frame, inter_reg1_sw_point, 5, (0, 0, 255), -1)

        if len(detected_wires) > 1:
            # 2. الواير الأطول/الأسفل موقعاً (Wire Long)
            long_wire_box = detected_wires[sorted_indices[-1]][0]
            lw_conf = detected_wires[sorted_indices[-1]][1]
            lw_x1, lw_y1, lw_x2, lw_y2 = map(int, long_wire_box)
            cv2.rectangle(annotated_frame, (lw_x1, lw_y1), (lw_x2, lw_y2), class_colors["Wire_Long"], 2)
            cv2.putText(annotated_frame, f"Wire Long {lw_conf:.2f}", (lw_x1, lw_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_colors["Wire_Long"], 2, cv2.LINE_AA)
            
            # 3. رسم بقية الوايرات العامة المكتشفة وتلوينها
            for mid_idx in sorted_indices[1:-1]:
                m_box = detected_wires[mid_idx][0]
                m_conf = detected_wires[mid_idx][1]
                mx1, my1, mx2, my2 = map(int, m_box)
                cv2.rectangle(annotated_frame, (mx1, my1), (mx2, my2), class_colors["Wire_General"], 2)
                cv2.putText(annotated_frame, f"Wire {m_conf:.2f}", (mx1, my1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_colors["Wire_General"], 2, cv2.LINE_AA)

            center_lw_x = int((lw_x1 + lw_x2) / 2)
            center_lw_y = int((lw_y1 + lw_y2) / 2)
            
            # حساب المسافة العمودية الصافية فقط (الفارق على محور Y)
            dist_wires = abs(center_sw_y - center_lw_y)
            all_wire_distances.append(dist_wires)
            
            # تعديل نقطة الالتقاء: لتصبح نقطة الزاوية القائمة بين خط العمودي وخط الأفقي
            # الإحداثيات: x الخاص بالواير الطويل، و y الخاص بالواير القصير
            intersection_point = (center_lw_x, center_sw_y)
            
            # رسم الخط العمودي من الواير الطويل إلى نقطة الالتقاء الجديدة
            cv2.line(annotated_frame, (center_lw_x, center_lw_y), intersection_point, class_colors["Distances"], 2, cv2.LINE_AA)
            # رسم الخط الأفقي من الواير القصير إلى نقطة الالتقاء الجديدة
            cv2.line(annotated_frame, (center_sw_x, center_sw_y), intersection_point, class_colors["Distances"], 2, cv2.LINE_AA)
            # وضع النقطة الحمراء عند زاوية التقاطع والتقاء الخطين تماماً
            cv2.circle(annotated_frame, intersection_point, 5, (0, 0, 255), -1)

    # --------------------------------------------------
    # Phase 4: Bead Processing & Angle Calculation
    # --------------------------------------------------
    bead_color = class_colors["Bead"]
    for (b_box, mask_val, b_conf) in bead_data:
        if mask_val is not None:
            polygon = np.array(mask_val, dtype=np.int32)
            if len(polygon) > 0:
                hull_polygon = cv2.convexHull(polygon)
                xb, yb, wb, hb = cv2.boundingRect(hull_polygon)
                
                line_color = class_colors["ThresholdLine"]
                cv2.line(annotated_frame, (xb, dynamic_y_threshold), (xb + wb, dynamic_y_threshold), line_color, 2, cv2.LINE_AA)

                hull_points = hull_polygon.reshape(-1, 2)
                target_right_x = xb + wb
                snap_threshold_px = 15 
                close_to_right_mask = np.abs(hull_points[:, 0] - target_right_x) <= snap_threshold_px
                hull_points[close_to_right_mask, 0] = target_right_x
                hull_polygon = hull_points.reshape(-1, 1, 2)

                cv2.rectangle(annotated_frame, (xb, yb), (xb + wb, yb + hb), bead_color, 3)
                cv2.polylines(annotated_frame, [hull_polygon], True, bead_color, 2)
                
                label_bead = f"Bead {b_conf:.2f}"
                cv2.putText(annotated_frame, label_bead, (xb, yb - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, bead_color, 2, cv2.LINE_AA)

                if object_blocking_at_top:
                    continue

                p1 = (xb + wb, yb)
                y_distances = np.abs(hull_points[:, 1] - yb)
                top_touching_indices = np.where(y_distances <= 3)[0]
                if len(top_touching_indices) > 0:
                    p3 = tuple(hull_points[top_touching_indices[np.argmin(hull_points[top_touching_indices, 0])]])
                else:
                    p3 = tuple(hull_points[np.argmin(hull_points[:, 1])])
                x_distances = np.abs(hull_points[:, 0] - target_right_x)
                close_to_x_indices = np.where(x_distances <= 3)[0]
                if len(close_to_x_indices) > 0:
                    p2 = tuple(hull_points[close_to_x_indices[np.argmin(np.abs(hull_points[close_to_x_indices, 1] - yb))]])
                else:
                    p2 = (xb + wb, p3[1])

                v1 = (p1[0] - p3[0], p1[1] - p3[1])
                v2 = (p2[0] - p3[0], p2[1] - p3[1])
                angle_rad = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
                angle_deg = np.abs(np.degrees(angle_rad))
                if angle_deg > 180:
                    angle_deg = 360 - angle_deg
                
                current_angle = angle_deg
                all_angles.append(current_angle) 

                cv2.line(annotated_frame, p3, p1, (0, 0, 255), 2)
                cv2.line(annotated_frame, p3, p2, (0, 255, 0), 2)

    # المتوسطات الحسابية
    avg_angle = np.mean(all_angles) if len(all_angles) > 0 else 0.0
    avg_reg2_width = np.mean(all_reg2_widths) if len(all_reg2_widths) > 0 else 0.0
    avg_height = np.mean(all_heights) if len(all_heights) > 0 else 0.0
    avg_wire_dist = np.mean(all_wire_distances) if len(all_wire_distances) > 0 else 0.0
    avg_reg1_short_dist = np.mean(all_reg1_short_distances) if len(all_reg1_short_distances) > 0 else 0.0

    # ==========================================
    # تصميم لوحة البيانات الرقمية (تحت على الشمال)
    # ==========================================
    panel_width = 680
    panel_height = 500  # تم زيادة الارتفاع قليلاً لاستيعاب إحصائية الـ Wires Count المضافة
    margin = 20

    x1 = margin
    y1 = height_vid - panel_height - margin
    x2 = margin + panel_width
    y2 = height_vid - margin

    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (200, 200, 200), 2)
    
    font_scale = 0.75  
    thickness_val = 2  
    
    # 1. إحصائيات زاوية الـ Bead
    if object_blocking_at_top:
        cv2.putText(annotated_frame, "Angle: BLOCKED", (x1 + 20, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, class_colors["ThresholdLine"], thickness_val, cv2.LINE_AA)
    else:
        ui_angle = f"Angle: {current_angle:.2f} Deg" if current_angle is not None else "Angle: N/A"
        cv2.putText(annotated_frame, ui_angle, (x1 + 20, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, class_colors["Bead"], thickness_val, cv2.LINE_AA)
    cv2.putText(annotated_frame, f"Avg Angle: {avg_angle:.2f} Deg", (x1 + 20, y1 + 75), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    
    # 2. إحصائيات العرض والارتفاع
    ui_reg2 = f"Reg2 Width: {current_reg2_width} px" if current_reg2_width is not None else "Reg2 Width: N/A"
    cv2.putText(annotated_frame, ui_reg2, (x1 + 20, y1 + 125), cv2.FONT_HERSHEY_SIMPLEX, font_scale, class_colors["Reg 2"], thickness_val, cv2.LINE_AA)
    cv2.putText(annotated_frame, f"Avg Reg2 Width: {avg_reg2_width:.2f} px", (x1 + 20, y1 + 160), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    
    ui_h = f"Height: {current_height_val} px" if current_height_val is not None else "Height: N/A"
    cv2.putText(annotated_frame, ui_h, (x1 + 20, y1 + 210), cv2.FONT_HERSHEY_SIMPLEX, font_scale, class_colors["Height"], thickness_val, cv2.LINE_AA)
    cv2.putText(annotated_frame, f"Avg Height: {avg_height:.2f} px", (x1 + 20, y1 + 245), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    
    # 3. المسافات الرأسية للـ Wires والـ Reg1
    ui_w_dist = f"Wires Vertical Dist: {dist_wires} px" if dist_wires is not None else "Wires Vertical Dist: N/A"
    cv2.putText(annotated_frame, ui_w_dist, (x1 + 20, y1 + 295), cv2.FONT_HERSHEY_SIMPLEX, font_scale, class_colors["Distances"], thickness_val, cv2.LINE_AA)
    cv2.putText(annotated_frame, f"Avg Wires Vertical Dist: {avg_wire_dist:.2f} px", (x1 + 20, y1 + 330), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    ui_reg1_sw = f"Short Wire to Reg1 Dist: {dist_short_wire_reg1} px" if dist_short_wire_reg1 is not None else "Short Wire to Reg1 Dist: N/A"
    cv2.putText(annotated_frame, ui_reg1_sw, (x1 + 20, y1 + 380), cv2.FONT_HERSHEY_SIMPLEX, font_scale, class_colors["Reg 1"], thickness_val, cv2.LINE_AA)
    cv2.putText(annotated_frame, f"Avg Short Wire to Reg1 Dist: {avg_reg1_short_dist:.2f} px", (x1 + 20, y1 + 415), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    # 4. إضافة طباعة عداد الـ Wires الفعلي والمستهدف (7)
    # اللون يتحول للأخضر إذا تطابق مع الرقم 7، وإلى الأحمر إذا كان هناك نقص أو زيادة
    count_color = (0, 255, 0) if total_wires_count == 7 else (0, 0, 255)
    ui_wire_count = f"Wires Count: {total_wires_count} / 7"
    cv2.putText(annotated_frame, ui_wire_count, (x1 + 20, y1 + 465), cv2.FONT_HERSHEY_SIMPLEX, font_scale, count_color, thickness_val, cv2.LINE_AA)

    # ==========================================
    # Save & Show final annotated frame
    # ==========================================
    out_video.write(annotated_frame)

    annotated_frame_path = os.path.join(output_frames_dir, f"frame_{frame_count:05d}_B_annotated.jpg")
    cv2.imwrite(annotated_frame_path, annotated_frame)

    screen = cv2.resize(annotated_frame, (width, height_vid))
    cv2.imshow("Industrial Inspection AI", screen)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out_video.release()  
cv2.destroyAllWindows()
print(f"Done! All data processed. Video saved to: {output_video_path}")