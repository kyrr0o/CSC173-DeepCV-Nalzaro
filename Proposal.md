# CSC173 Deep Computer Vision Project Proposal
**Student:** Kaycee T. Nalzaro, 2021-1756  
**Date:** December 15, 2025

## 1. Project Title
**DrainSight: Video-Based Drainage Risk Assessment Using Deep Computer Vision**

## 2. Problem Statement
Urban drainage systems are prone to blockage due to improper waste disposal, leading to flooding and environmental hazards, especially during heavy rainfall. In many local areas in Mindanao, 
drainage monitoring is still conducted manually and reactively, making early detection difficult. With the increasing availability of CCTV and mobile video recordings, there is an opportunity 
to leverage deep computer vision techniques to automatically analyze drainage conditions. This project aims to develop a vision-based system that can assess drainage risk by analyzing visible 
waste accumulation and water flow patterns from video data.

## 3. Objectives
- Develop a deep computer vision pipeline for analyzing drainage and water-flow videos  
- Utilize object detection to estimate waste accumulation from video frames  
- Integrate motion analysis to capture water flow behavior  
- Generate an interpretable risk score categorizing drainage conditions as low, moderate, or high risk  

## 4. Dataset Plan
- **Source:** Publicly available drainage and water-flow videos (online repositories and locally recorded samples)  
- **Type:** Video data (continuous frame sequences)  
- **Classes:** Detected objects from pretrained YOLOv8 model used as visual indicators of waste presence  
- **Acquisition:** Videos will be collected from open online sources such as Youtube and Vecteezy. 

## 5. Technical Approach
- **Architecture:** Frame-based video processing pipeline with object detection and motion analysis  
- **Model:** YOLOv8n (pretrained on COCO dataset) for real-time object detection  
- **Additional CV Technique:** Optical flow (Farneback method) for estimating water movement  
- **Framework:** PyTorch with Ultralytics YOLOv8 and OpenCV  
- **Hardware:** Local machine or Google Colab for video processing and experimentation  

## 6. Expected Challenges & Mitigations
- **Challenge:** Absence of labeled, domain-specific drainage datasets  
  - **Mitigation:** Use pretrained object detection models and rely on object density as a proxy for waste accumulation  
- **Challenge:** False detections from non-waste objects  
  - **Mitigation:** Apply confidence thresholds and refine risk scoring logic  
- **Challenge:** Variability in lighting and video quality  
  - **Mitigation:** Resize frames and normalize motion measurements to maintain stable performance  
