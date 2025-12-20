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
