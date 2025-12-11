# CanalGuard AI: A CCTV-Based Deep Learning Model for Monitoring Water Flow, Trash Accumulation, and Clogging Risk in Urban Drainage Systems

CSC173 Intelligent Systems Final Project
Mindanao State University - Iligan Institute of Technology
Student: Kaycee T. Nalzaro, 2021-1756
Semester: AY 2025-2026 Sem 1

## Abstract
Urban flooding in the Philippines is frequently caused not only by heavy rainfall but also by clogged drainage canals that accumulate trash and experience reduced water flow. Manual inspection is inconsistent, unsafe during bad weather, and cannot provide continuous monitoring.  
This project introduces **DrainSight**, a lightweight computer vision pipeline that uses:
- a **pretrained YOLOv8 model** to detect trash and debris,  
- **optical flow** to estimate water movement, and  
- a combined **clogging risk score** (LOW / MODERATE / HIGH)

This automatically assess drainage condition from short canal videos. The system processes four real-world canal scenarios—clean, moderate trash, heavy trash, and stagnant water—and generates annotated output videos showing detection boxes, scores, and final risk labels. The goal is to demonstrate how deep learning can support early clogging detection using accessible tools and minimal data.

# Table of Contents
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
Flooding remains one of the most persistent problems in Philippine cities.  
Even moderate rainfall can cause street-level flooding when drainage canals ("kanal") are clogged with accumulated waste or stagnant water. Current monitoring systems rely mostly on manual inspection and occasional CCTV review, which:
- cannot detect clogging early,
- do not provide continuous automated analysis, and
- are unsafe for LGU personnel during bad weather.
Deep learning–based computer vision has advanced significantly, enabling automatic detection and analysis of visual patterns. This project applies such techniques to the drainage maintenance domain through **DrainSight**, a lightweight system for early detection of trash buildup and water stagnation using short canal videos.

##Related Work

## Methodology

### 1. System Overview
1. **Trash Detection (YOLOv8n)**
2. **Water Flow Estimation (Optical Flow)**
3. **Clogging Risk Scoring**

Each input video is processed frame-by-frame, where the system computes:
- `TrashScore` → how much of the frame contains detected debris  
- `FlowScore` → how strong the water movement is  
- `RiskScore` → weighted combination determining LOW / MODERATE / HIGH clogging risk

### 2. Deep Learning Model  
- **Model:** YOLOv8n (Ultralytics)  
- **Initial Weights:** `yolov8n.pt` pretrained on COCO  
- These weights act as the **initial model parameters**, satisfying the requirement of “initial models/weights.”  
- No retraining was performed; the pretrained model is used directly for inference.

Internally, YOLO uses CNN layers involving:
- Convolution,
- Activation (ReLU/SiLU),
- Downsampling/pooling, and
- Detection heads.




















