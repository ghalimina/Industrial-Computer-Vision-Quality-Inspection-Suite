# 🧪 Heavy Rubber Mill Instance Segmentation & Alpha Blending

[![YOLOv8-Seg](https://img.shields.io/badge/YOLO-v8%20Segmentation-green.svg)](https://ultralytics.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-Alpha%20Blending-red.svg)](https://opencv.org)
[![High Res](https://img.shields.io/badge/Resolution-3.2K%20(3200x1800)-blue.svg)]()

> **Real-Time Instance Segmentation of Steel Wire Reinforcement and Raw Rubber Mass on Heavy Industrial Calender Rolling Mills with Glare-Resistant Alpha Blending.**

---

## 📌 1. Project Overview

In rubber compounding and calender extrusion plants, high ambient illumination and mirror-finished rotating steel rollers produce blinding specular reflections (glare). Conventional thresholding and bounding boxes fail due to severe light clipping.

This system provides:
1. **Pixel-Level Instance Segmentation (`YOLOv8-Seg`)**: Accurately outlines the irregular contours of the rubber mass (`Rubber`) and fine reinforcement wires (`wire`).
2. **40% Alpha-Blended Visual Feedback**: Semi-transparent colored polygon masks that allow operators to inspect surface texture and material flow under the mask without obstruction.
3. **Automated Defect & Feed Telemetry**: Tracks wire feed alignment and frame-by-frame quality logging.

![Rubber Mill Demo](../assets/rubber_mill_demo.gif)

---

## ⚙️ 2. Key Engineering Innovations

### A. Alpha-Blending Mask Formulation
Rather than applying opaque overlays that obscure material flow, the segmentation mask is blended dynamically:
$$\text{Frame}_{\text{out}} = \alpha \times \text{Mask}_{\text{color}} + (1 - \alpha) \times \text{Frame}_{\text{original}}$$
Where $\alpha = 0.40$ (40% color fill) and $\beta = 0.60$ (60% background transparency), preserving sub-surface wire visibility.

### B. Detected Classes & Visual Palettes
| Class | BGR Color | Industrial Role |
| :--- | :--- | :--- |
| **`wire`** | `(255, 255, 0)` Cyan | Steel cord reinforcement feeding into the calender gap |
| **`Rubber`** | `(0, 165, 255)` Amber/Orange | Raw rubber compound mass undergoing plastic deformation |

---

## 🏗️ 3. Directory Structure

```text
rubber/
├── test.py             # Inference pipeline with alpha blending & frame logging
├── bestv2.pt           # Trained YOLOv8-Seg weights (Instance Segmentation)
└── README.md           # Documentation
```

---

## 🚀 4. How to Run

```bash
# Run real-time segmentation with alpha-blended visualization
python test.py
```
