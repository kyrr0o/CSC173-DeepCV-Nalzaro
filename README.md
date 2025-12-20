# DrainSight: Video-Based Drainage Risk Assessment Using Deep Computer Vision
**CSC173 Intelligent Systems Final Project**  
*Mindanao State University - Iligan Institute of Technology*  
**Student:** Kaycee T. Nalzaro, 2021-1756  
**Semester:** AY 2025-2026 Sem 1
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org)

## Abstract
Urban drainage systems are vulnerable to blockage due to improper waste disposal, often leading to flooding and environmental hazards, particularly in densely populated areas. 
This project presents DrainSight, a deep computer vision–based system for assessing drainage risk using video analysis. Instead of relying on supervised training with labeled datasets, 
the system operates in inference mode using a pretrained YOLOv8 model to detect visible objects from video frames. Waste accumulation is approximated through object area coverage, 
while water movement is estimated using optical flow analysis between consecutive frames. These visual cues are combined into a unified risk score that categorizes drainage conditions 
as low, moderate, or high risk. Experimental demonstrations show that the system produces interpretable, real-time outputs with annotated visualizations, including bounding boxes, 
flow statistics, risk levels, and frame rate. The primary contribution of this project is a lightweight, deployable deep computer vision pipeline that integrates object detection and 
motion analysis for practical environmental monitoring without requiring model retraining.

## Table of Contents
- [Introduction](#introduction)
- [Related Work](#related-work)
- [Methodology](#methodology)
- [Experiments & Results](#experiments--results)
- [Discussion](#discussion)
- [Ethical Considerations](#ethical-considerations)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [References](#references)

## Introduction
### Problem Statement
Drainage blockage caused by waste accumulation remains a persistent urban problem that can result in flooding, infrastructure damage, and health risks. In many local contexts, including urban 
areas in Mindanao, drainage inspection is often conducted manually and only after flooding incidents occur. With the increasing availability of surveillance and mobile video recordings, 
there is an opportunity to apply deep computer vision techniques to automatically analyze drainage conditions. This project explores a video-based approach to assessing drainage risk using 
object detection and motion analysis.

### Objectives
- Develop a video-based deep computer vision pipeline for drainage analysis
- Estimate waste accumulation using object detection
- Analyze water movement using optical flow
- Generate an interpretable drainage risk level in real time

![Problem Demo](images/problem_example.gif)

## Related Works
- Due to the frequent and sudden occurrence of urban waterlogging, targeted and rapid risk monitoring is extremely important for urban management [1]. Recent study develops a deep learning and computer vision method for flood scene understanding from continuous video. It integrates motion analysis (optical flow-based modules) with object tracking to analyze dynamic scenes like flooding [2].
- The Philippines has contributed to immense discharge of plastic waste into rivers. The development of an object detection model based on YOLOv8 to identify floating debris on a water surface accurately and in real-time, including garbage and invasive plants was proposed recently as an ooptimal solution to this problem [3].
- Several studies adopted the YOLOv8 framework for its modern, anchor-free design, decoupled heads, TensorRT export, and integrated tracking. It also delivers higher detection metrics at comparable inference times, facilitating real-time deployment under the hardware constraints [4].
- Optical flow methods, which track pixel-wise motion between image frames, have also been explored for river flow measurement. Optical flow algorithms can be combined with deep neural networks to capture complex, unstructured motion patterns in natural rivers [5].
- The framework serves as a powerful tool for understanding the dynamics of flood-affected areas and optimizing rescue strategies. This framework capability enhances the efficiency and effectiveness of humanitarian aid operations in flood-impacted regions [6].

## Reference
[1] Huang, H., Lei, X., Liao, W., Li, H., Wang, C., & Wang, H. (2023). A real-time detecting method for continuous urban flood scenarios based on computer vision on block scale. Remote Sensing, 15(6), 1696.
[2] Yan, X., Zhu, Y., Wang, Z., Xu, B., He, L., & Xia, R. (2025). Intelligent flood scene understanding using computer vision-based multi-object tracking. Water, 17(14), 2111.
[3] Tomas, J. P. Q., Tupas, J. E. E., Soniel, M. T., Caruz, C. H. M. E., & Babar, D. B. (2024). Real-time detection of floating debris in waterways using YOLOv8. In Proceedings of the 14th International Workshop on Computer Science and Engineering (WCSE 2024).
[4] Tikász, G., Gyalai-Korpos, M., Fleit, G., & Baranya, S. (2025). Real-time detection of macroplastic pollution in inland waters: development of a lightweight image recognition system. Frontiers in Environmental Science, 13, 1666271.
[5] Chen, W., Nguyen, K. A., & Lin, B.-S. (2025). Deep learning and optical flow for river velocity estimation: Insights from a field case study. Sustainability.
[6] Yan, X., Zhu, Y., Wang, Z., Xu, B., He, L., & Xia, R. (2025). Intelligent flood scene understanding using computer vision-based multi-object tracking. Water, 17(14), 2111.

## Methodology
### Dataset
This project operates on raw video data rather than a static labeled image dataset.
- Source: Publicly available and manually captured drainage and water-flow videos
- Data Type: Continuous video frames
- Preprocessing: Frame resizing to a fixed width of 640 pixels for real-time inference
Four initial video samples were used to represent different observed conditions:
- Clean drainage (minimal waste, normal flow)
- Light trash accumulation
- Heavy trash accumulation
- Clogged or near-clogged drainage
These categories are used for qualitative evaluation and are not treated as supervised class labels.

## Architecture
![Model Diagram](images/architecture.png)

The proposed system follows a frame-based video processing pipeline:
- A pretrained YOLOv8n model performs object detection on each frame.
- Detected bounding boxes are used to estimate trash density based on area coverage.
- Optical flow (Farneback method) estimates water movement between frames.
- Trash density and inverse flow magnitude are fused into a drainage risk score.
- Annotated frames are written to an output video with visual overlays

### Key Parameters
| Parameter            | Value                                    |
| -------------------- | ---------------------------------------- |
| YOLO Model           | YOLOv8n (pretrained)                     |
| Input Width          | 640 pixels                               |
| Detection Confidence | 0.05                                     |
| Trash Weight         | 0.8                                      |
| Flow Weight          | 0.2                                      |
| Risk Thresholds      | Low < 0.25, Moderate < 0.50, High ≥ 0.50 |

### Inference Code Snippet
```python
model = YOLO("yolov8n.pt")
results = model(frame, imgsz=640, conf=0.05, verbose=False)
```





















