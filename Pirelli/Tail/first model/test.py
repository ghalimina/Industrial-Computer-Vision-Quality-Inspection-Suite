import cv2
import math
from ultralytics import YOLO

# ==========================================
# 1. إعدادات المسارات والموديل (Tail)
# ==========================================
model_path = r"E:\elecrto pi\Tail\best.pt" 
video_path = r"E:\elecrto pi\Tail\Internal Head.mp4" 
output_path = r"E:\elecrto pi\Tail\Internal Head_output.mp4"

model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height_vid))

show_view = True
cv2.namedWindow("Tail Engineering Analysis", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tail Engineering Analysis", 1280, 720) 

def get_center(box):
    return (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))

frame_count = 0

# ==========================================
# 2. معالجة الفيديو فريم بفريم
# ==========================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frames_remaining = max(0, total_frames - frame_count)

    results = model.predict(frame, conf=0.60, verbose=False)
    boxes = results[0].boxes

    wires = []
    clamp_boxes = []

    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id].lower() 
        xyxy = box.xyxy[0].cpu().numpy()
        conf_score = float(box.conf[0])

        if "wire" in cls_name:
            wires.append(xyxy)
            color = (255, 0, 0) 
        elif "clamp" in cls_name:
            clamp_boxes.append(xyxy)
            color = (0, 255, 0) 
        else:
            color = (128, 128, 128)

        if show_view:
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
            label_text = f"{cls_name.upper()} {conf_score:.2f}"
            cv2.putText(frame, label_text, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    wire_count = len(wires)
    
    dist_s_l_val = "N/A"
    dist_s_c_val = "N/A"
    long_len_val = "N/A"
    short_len_val = "N/A"

    if wire_count >= 2:
        longest_wire = min(wires, key=lambda w: (get_center(w)[1] * 2) + get_center(w)[0])
        shortest_wire = max(wires, key=lambda w: (get_center(w)[1] * 2) + get_center(w)[0])

        c_short = get_center(shortest_wire)
        c_long = get_center(longest_wire)

        long_len = int(math.hypot(longest_wire[2] - longest_wire[0], longest_wire[3] - longest_wire[1]))
        short_len = int(math.hypot(shortest_wire[2] - shortest_wire[0], shortest_wire[3] - shortest_wire[1]))
        long_len_val = f"{long_len}px"
        short_len_val = f"{short_len}px"

        dist_vertical_sl = abs(c_short[1] - c_long[1])
        dist_s_l_val = f"{dist_vertical_sl}px"

        if show_view:
            cv2.circle(frame, c_short, 6, (0, 0, 255), -1)
            cv2.circle(frame, c_long, 6, (0, 0, 255), -1)
            cv2.putText(frame, "Shortest", (c_short[0] + 15, c_short[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, "Longest", (c_long[0] + 15, c_long[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            p_intersect_sl = (c_long[0], c_short[1]) 
            cv2.line(frame, c_long, p_intersect_sl, (0, 255, 255), 2)  
            cv2.line(frame, p_intersect_sl, c_short, (0, 255, 255), 2) 
            
            cv2.putText(frame, f"V-Dist: {dist_vertical_sl}px", 
                        (c_long[0] + 10, int((c_long[1] + c_short[1])/2)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # ==========================================
        # الجزء المصحح: تعديل نسبة النزول لضبط موقع الشفة بالملي
        # ==========================================
        if len(clamp_boxes) > 0:
            valid_clamps = [c for c in clamp_boxes if get_center(c)[1] >= (c_short[1] - 30) or get_center(c)[0] >= (c_short[0] - 30)]
            
            if len(valid_clamps) > 0:
                nearest_clamp = min(valid_clamps, key=lambda c: math.dist(c_short, get_center(c)))
                
                clamp_box_height = nearest_clamp[3] - nearest_clamp[1]
                
                # 🎯 جرب النسبة ديه (0.45 يعني هينزل لقرب المنتصف بالظبط)
                # لو عايز تنزل زيادة خليها 0.50، لو عايز ترفعها شوية خليها 0.35
                offset_y = int(clamp_box_height * 0.70) 
                
                c_clamp_top = (int((nearest_clamp[0] + nearest_clamp[2]) / 2), int(nearest_clamp[1] + offset_y))
                
                dist_vertical_sc = abs(c_short[1] - c_clamp_top[1])
                dist_s_c_val = f"{dist_vertical_sc}px"
                
                if show_view:
                    cv2.circle(frame, c_clamp_top, 5, (255, 0, 255), -1)
                    
                    p_intersect_clamp = (c_clamp_top[0], c_short[1]) 
                    cv2.line(frame, c_clamp_top, p_intersect_clamp, (255, 0, 255), 2)  
                    cv2.line(frame, c_short, p_intersect_clamp, (255, 0, 255), 2)  
                    
                    cv2.putText(frame, f"V-Clamp: {dist_vertical_sc}px", 
                                (c_clamp_top[0] + 10, int((c_clamp_top[1] + c_short[1])/2)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # ==========================================
    # 3. عرض لوحة الـ UI (Dashboard)
    # ==========================================
    if show_view:
        rect_top_left = (10, height_vid - 250)
        rect_bottom_right = (460, height_vid - 20)
        cv2.rectangle(frame, rect_top_left, rect_bottom_right, (0, 0, 0), -1)
        
        y_offset = height_vid - 220
        
        cv2.putText(frame, f"Wires Detected: {wire_count}/6", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += 30
        cv2.putText(frame, f"Longest Wire Len: {long_len_val}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(frame, f"Shortest Wire Len: {short_len_val}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(frame, f"Vertical Dist (S-L): {dist_s_l_val}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += 30
        cv2.putText(frame, f"Vertical Clamp Dist: {dist_s_c_val}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        y_offset += 30
        cv2.putText(frame, f"Frames Remaining: {frames_remaining}/{total_frames}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
        cv2.putText(frame, "Press 'Z' to toggle View (Speed up)", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    print(f"Frame: {frame_count}/{total_frames} | Wires: {wire_count}/6 | V-Dist: {dist_s_l_val} | V-Clamp: {dist_s_c_val}")
    out.write(frame)

    if show_view:
        cv2.imshow("Tail Engineering Analysis", frame)
    else:
        try:
            cv2.destroyWindow("Tail Engineering Analysis")
        except:
            pass

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break
    elif key == ord('z'): 
        show_view = not show_view
        if show_view:
            cv2.namedWindow("Tail Engineering Analysis", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Tail Engineering Analysis", 1280, 720)

cap.release()
out.release()
cv2.destroyAllWindows()