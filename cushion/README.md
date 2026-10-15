# 🛡️ Calender Cushion Two-Stage Cascade Inspection System

[![YOLOv8](https://img.shields.io/badge/YOLO-v8%20%7C%20v11-green.svg)](https://ultralytics.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-Camera%20Calibration-red.svg)](https://opencv.org)
[![4K Vision](https://img.shields.io/badge/Resolution-4K%20UHD%20(3840x2160)-blue.svg)]()

> **Automated Optical Distortion Correction & Hierarchical Two-Stage Defect Localization on High-Speed Industrial Rubber Calender Production Lines.**

---

## 📌 1. Project Overview

During heavy rubber calendering, continuous rubber cushions are transported through giant rotating steel rolls at high speed. Wide-angle industrial cameras suffer from optical distortion, and detecting sub-millimeter defects (such as folds, punctures, and tears) across wide web surfaces poses a major computer vision challenge.

This project delivers an end-to-end vision pipeline:
1. **Camera Intrinsics Matrix & Distortion Rectification**: Eliminates fisheye/barrel distortion on 4K factory feeds.
2. **Two-Stage Cascade Architecture**: Coarse localization of the rubber web followed by fine-grained micro-defect detection on the cropped ROI.
3. **Picture-in-Picture (PiP) Operator HUD**: Synchronous high-magnification overlay for plant technicians.

![Calender Cushion Demo](../assets/cushion_calender_demo.gif)

---

## ⚙️ 2. Algorithmic Pipeline

```mermaid
graph LR
    A["Raw 4K Camera Feed (3840x2160)"] --> B["cv2.initUndistortRectifyMap (Intrinsic Matrix)"]
    B --> C["Undistorted Full-Frame Stream"]
    C --> D["Model 1: Coarse Cushion Localization (cou)"]
    D --> E["Dynamic Sub-Region Cropping (ROI)"]
    E --> F["Model 2: Fine Defect Detection (fol / hol)"]
    F --> G["Live PiP 65% Overlay & Metric Telemetry"]
```

### A. Optical Camera Calibration
Using pre-calibrated pinhole camera parameters:
* **Camera Matrix ($K$)**:
  $$\begin{bmatrix} 8842.95 & 0 & 1976.03 \\ 0 & 8061.43 & 1055.31 \\ 0 & 0 & 1 \end{bmatrix}$$
* **Distortion Coefficients ($D$)**:
  $$[-4.45, 57.30, -0.076, 0.098, -295.95]$$
Frames are dynamically undistorted using bilinear interpolation (`cv2.remap`) before feeding to the detection pipeline.

### B. Two-Stage Cascade Inference
* **Stage 1 (Web Localization)**: Model 1 detects the moving cushion with `conf=0.60` and extracts the bounding coordinates $[x_1, y_1, x_2, y_2]$.
* **Stage 2 (Micro-Defect Diagnosis)**: The cropped patch is passed to `bestv1.11.pt` with `conf=0.40`, `iou=0.25`, and `agnostic_nms=True` to isolate:
  - `cou`: Cushion body
  - `fol`: Surface fold / crease defect
  - `hol`: Material puncture / tear

---

## 🏗️ 3. Directory Structure

```text
cushion/
├── allcode.py                  # Main unified calibration + 2-stage inference pipeline
├── cam$re_detect_fol_hol.py    # Dedicated defect extraction module
├── bestv1.11.pt                # Trained weights for fold & hole detection
└── README.md                   # System documentation
```

---

## 🚀 4. How to Run

```bash
# Run the complete calibrated two-stage pipeline on input video
python allcode.py
```
