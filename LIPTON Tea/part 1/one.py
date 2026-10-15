import cv2
import numpy as np
import os
from ultralytics import YOLO

# ==========================================
# Paths
# ==========================================
model_path = r"E:\elecrto pi\lebt\bestv23.pt"
video_path = r"E:\elecrto pi\lebt\test y25.mkv"
output_frames_dir = r"E:\elecrto pi\lebt\test y25_framesone"
output_video_path = r"E:\elecrto pi\lebt\test y25_outputdetectone.mp4"

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
# Colors
# ==========================================
class_colors = {
    "Bad": (0, 0, 255),
    "good": (0, 255, 0),
    "red": (0, 165, 255),
    "Text": (255, 0, 0)
}

default_color = (255, 255, 255)
target_classes = ["Bad", "good", "red", "Text"]

# ==========================================
# Window
# ==========================================
cv2.namedWindow("Industrial Inspection AI", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "Industrial Inspection AI",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    print(f"Processing frame {frame_count}")

    # حفظ الفريم الأصلي
    original_frame_path = os.path.join(
        output_frames_dir,
        f"frame_{frame_count:05d}_A_original.jpg"
    )
    cv2.imwrite(original_frame_path, frame)

    # ==========================================
    # Run Model
    # ==========================================
    results = model.predict(
        frame,
        conf=0.60,
        iou=0.45,
        verbose=False
    )

    annotated_frame = frame.copy()
    current_detections = {cls_name: 0 for cls_name in target_classes}

    # ==========================================
    # Process Results
    # ==========================================
    for result in results:

        if result.boxes is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        filtered = []

        for box, conf, cls in zip(boxes, confs, classes):
            class_name = model.names[int(cls)]

            if class_name not in target_classes:
                continue

            should_add = True

            for i, (old_box, old_conf, old_class) in enumerate(filtered):

                # حساب IOU
                xA = max(box[0], old_box[0])
                yA = max(box[1], old_box[1])
                xB = min(box[2], old_box[2])
                yB = min(box[3], old_box[3])

                inter = max(0, xB - xA) * max(0, yB - yA)

                area1 = (box[2] - box[0]) * (box[3] - box[1])
                area2 = (old_box[2] - old_box[0]) * (old_box[3] - old_box[1])

                iou_score = inter / (area1 + area2 - inter + 1e-6)

                # لو نفس العلبة تقريباً
                if iou_score > 0.5:

                    # لو good و Bad ظهروا مع بعض
                    if {class_name, old_class} == {"good", "Bad"}:
                        if conf > old_conf:
                            filtered[i] = (box, conf, class_name)

                        should_add = False
                        break

            if should_add:
                filtered.append((box, conf, class_name))

        # ==========================================
        # Draw Final Filtered Objects
        # ==========================================
        for box, conf, class_name in filtered:
            xb1, yb1, xb2, yb2 = map(int, box)

            current_detections[class_name] += 1

            color = class_colors.get(class_name, default_color)

            cv2.rectangle(
                annotated_frame,
                (xb1, yb1),
                (xb2, yb2),
                color,
                3
            )

            label_text = f"{class_name} {conf:.2f}"

            cv2.putText(
                annotated_frame,
                label_text,
                (xb1, yb1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA
            )

    # ==========================================
    # Dashboard Panel
    # ==========================================
    panel_width = 500
    panel_height = 200
    margin = 20

    x1 = margin
    y1 = height_vid - panel_height - margin
    x2 = margin + panel_width
    y2 = height_vid - margin

    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (200, 200, 200), 2)

    cv2.putText(
        annotated_frame,
        f"Frame: {frame_count}",
        (x1 + 20, y1 + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1
    )

    y_offset = y1 + 65

    for cls_name in target_classes:
        count_text = f"Current {cls_name}: {current_detections[cls_name]}"
        color = class_colors.get(cls_name, default_color)

        cv2.putText(
            annotated_frame,
            count_text,
            (x1 + 20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

        y_offset += 35

    # ==========================================
    # Save Output
    # ==========================================
    out_video.write(annotated_frame)

    annotated_frame_path = os.path.join(
        output_frames_dir,
        f"frame_{frame_count:05d}_B_annotated.jpg"
    )

    cv2.imwrite(annotated_frame_path, annotated_frame)

    cv2.imshow("Industrial Inspection AI", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================================
# Release
# ==========================================
cap.release()
out_video.release()
cv2.destroyAllWindows()

print("Done!")
print(f"Output Video Saved: {output_video_path}")